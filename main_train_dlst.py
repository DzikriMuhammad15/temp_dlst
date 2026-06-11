import argparse
import json
import os
import random
from dataclasses import asdict, is_dataclass

import numpy as np
import torch
import yaml

from agents.dlst_agent import DLSTNegotiator
from agents.dlst.strategy_template import StrategyTemplateSpec
from agents.dlst.opponent_model import FrequencyOpponentModel
from agents.rule_agent import RuleBasedAgent
from environments.domain_builder import build_domain
from environments.negmas_env import NegMASBilateralEnv
from evaluation.metrics import compute_episode_metrics
from models.ddpg_networks import DDPGActorThreshold, DDPGActorStrategy, DDPGCritic
from training.buffer import TrajectoryBuffer
from reward_shaper.reward_shaper import (
    ProposeRewardConfig,
    RespondRewardConfig,
    ThresholdRewardConfig,
    RewardShaper,
)
from trainers.ddpg_trainer import DDPGTrainer
from utils.io import ensure_dir, save_json, save_yaml, timestamp
from utils.logger import SimpleLogger
from utils.plotting import plot_training_history


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_episode_json(path: str, episode):
    ensure_dir(os.path.dirname(path))
    if is_dataclass(episode):
        episode = asdict(episode)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episode, f, indent=2, ensure_ascii=False, default=str)


def build_ddpg_trainer(reward_shaper, state_dim, actor, critic, lr, cfg, device, action_dim, state_for):
    """Helper: bangun satu DDPGTrainer dengan actor+critic dan attach models."""
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

    trainer = DDPGTrainer(
        gamma=cfg["training"]["gamma"],
        tau=cfg["training"].get("ddpg_tau", 0.005),
        device=device,
        epoch=cfg["training"].get("ddpg_epoch", 5),
        batch_size=cfg["training"].get("batch_size", 32),
        reward_shaper=reward_shaper,
        gradient_clip_norm=cfg["training"].get("gradient_clip_norm", 1.0),
        action_dim=action_dim,
    )
    # Beri tahu trainer jenis state apa yang harus dibangun saat rekonstruksi.
    trainer._state_for = state_for

    trainer.attach_models({
        "actor": {
            "model": actor,
            "optimizer": actor_opt,
            "config": _actor_config(actor, state_dim, action_dim),
        },
        "critic": {
            "model": critic,
            "optimizer": critic_opt,
            "config": {"state_dim": state_dim, "action_dim": action_dim, "hidden_dim": 128},
        },
    })
    return trainer


def _actor_config(actor, state_dim, action_dim):
    """Config untuk rekonstruksi actor saat load_checkpoint."""
    if isinstance(actor, DDPGActorThreshold):
        return {"state_dim": state_dim, "hidden_dim": 128}
    if isinstance(actor, DDPGActorStrategy):
        return {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "split_sizes": list(actor.split_sizes),
            "hidden_dim": 128,
        }
    return {"state_dim": state_dim, "action_dim": action_dim, "hidden_dim": 128}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_prefix", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    run_id = timestamp()
    prefix = (args.save_prefix or run_id) + "_DLST"

    trajectories_dir = os.path.join(cfg["paths"]["trajectories_dir"], prefix)
    models_dir = os.path.join(cfg["paths"]["models_dir"], prefix)
    runs_dir = os.path.join(cfg["paths"]["runs_dir"], prefix)
    ensure_dir(trajectories_dir)
    ensure_dir(models_dir)
    ensure_dir(runs_dir)

    save_yaml(os.path.join(runs_dir, "resolved_config.yaml"), cfg)

    logger = SimpleLogger(prefix=f"TRAIN-{prefix}")
    logger.log("Loading domain...")

    domain_spec, learner_ufun, opponent_ufun = build_domain(
        cfg["env"]["domain_path"],
        max_combinations=cfg["env"]["max_action_library"],
    )

    logger.log(f"Domain loaded: {domain_spec.domain_name}")
    logger.log(f"Issue count: {len(domain_spec.issues)}")
    logger.log(f"Action library size: {len(domain_spec.action_library)}")

    # ──────────────────────────────────────────────────────────────────
    # STRATEGY TEMPLATE SPEC
    # ──────────────────────────────────────────────────────────────────
    # Contoh konfigurasi 2 fase. Bisa diubah sesuai kebutuhan eksperimen.
    # Acceptance tactics (T_a): U_future, quantile, u_bar, u_fixed.
    # Bidding tactics (T_b): boulware, pareto, b_opp, random_above.
    acceptance_spec = StrategyTemplateSpec(
        n_phases=2,
        tactics_per_phase=[2, 2],
        tactic_names_per_phase=[
            ["quantile", "u_bar"],     # fase 1
            ["u_fixed", "quantile"],   # fase 2
        ],
        params_per_tactic=2,  # (a, b)
    )
    bidding_spec = StrategyTemplateSpec(
        n_phases=2,
        tactics_per_phase=[2, 2],
        tactic_names_per_phase=[
            ["boulware", "pareto"],        # fase 1
            ["pareto", "random_above"],    # fase 2
        ],
        params_per_tactic=2,
    )

    # ──────────────────────────────────────────────────────────────────
    # REWARD SHAPERS (3 jenis)
    # ──────────────────────────────────────────────────────────────────
    # Opponent utility approximation (frequency model) MENYATU dengan
    # reward shaper respond.
    opponent_model = FrequencyOpponentModel(
        issue_names=domain_spec.issue_names,
        value_lists=domain_spec.value_lists,
    )

    # Propose (bidding) reward = r_bid (Persamaan 9) → mode1 propose (sudah ada).
    propose_reward_cfg = ProposeRewardConfig(
        mode=cfg["training"].get("propose_reward_mode", "mode1"),
        utility_scale=cfg["training"]["utility_scale"],
    )
    reward_shaper_propose = RewardShaper(
        propose_reward_cfg,
        utility_min=domain_spec.utility_min,
        utility_max=domain_spec.utility_max,
        for_="propose",
    )

    # Respond (acceptance) reward = r_acc (Persamaan 10) → mode "dlst",
    # membutuhkan opponent_model untuk U_o.
    respond_reward_cfg = RespondRewardConfig(
        mode=cfg["training"].get("dlst_respond_reward_mode", "dlst"),
        utility_scale=cfg["training"]["utility_scale"],
    )
    reward_shaper_respond = RewardShaper(
        respond_reward_cfg,
        utility_min=domain_spec.utility_min,
        utility_max=domain_spec.utility_max,
        for_="respond",
        opponent_model=opponent_model,
    )

    # Threshold utility reward = r_ū_t (Persamaan 8) → for_="utility_threshold".
    threshold_reward_cfg = ThresholdRewardConfig(
        mode=cfg["training"].get("threshold_reward_mode", "mode1"),
        utility_scale=cfg["training"]["utility_scale"],
    )
    reward_shaper_threshold = RewardShaper(
        threshold_reward_cfg,
        utility_min=domain_spec.utility_min,
        utility_max=domain_spec.utility_max,
        for_="utility_threshold",
    )

    # ──────────────────────────────────────────────────────────────────
    # DIMENSIONS
    # ──────────────────────────────────────────────────────────────────
    state_dim_threshold = DLSTNegotiator.STATE_DIM_THRESHOLD
    state_dim_respond = DLSTNegotiator.STATE_DIM_RESPOND
    state_dim_propose = DLSTNegotiator.STATE_DIM_PROPOSE

    threshold_action_dim = 1
    bidding_action_dim = bidding_spec.action_dim
    acceptance_action_dim = acceptance_spec.action_dim

    lr = cfg["training"]["lr"]

    # ──────────────────────────────────────────────────────────────────
    # TRAINER 1: THRESHOLD (single-output actor)
    # ──────────────────────────────────────────────────────────────────
    threshold_actor = DDPGActorThreshold(state_dim_threshold)
    threshold_critic = DDPGCritic(state_dim_threshold, threshold_action_dim)
    threshold_trainer = build_ddpg_trainer(
        reward_shaper_threshold, state_dim_threshold,
        threshold_actor, threshold_critic, lr, cfg, args.device,
        action_dim=threshold_action_dim, state_for="utility_threshold",
    )

    # ──────────────────────────────────────────────────────────────────
    # TRAINER 2: PROPOSE / BIDDING (multi-output actor)
    # ──────────────────────────────────────────────────────────────────
    bidding_actor = DDPGActorStrategy(state_dim_propose, bidding_action_dim, bidding_spec.split_sizes)
    bidding_critic = DDPGCritic(state_dim_propose, bidding_action_dim)
    propose_trainer = build_ddpg_trainer(
        reward_shaper_propose, state_dim_propose,
        bidding_actor, bidding_critic, lr, cfg, args.device,
        action_dim=bidding_action_dim, state_for="propose",
    )

    # ──────────────────────────────────────────────────────────────────
    # TRAINER 3: RESPOND / ACCEPTANCE (multi-output actor)
    # ──────────────────────────────────────────────────────────────────
    acceptance_actor = DDPGActorStrategy(state_dim_respond, acceptance_action_dim, acceptance_spec.split_sizes)
    acceptance_critic = DDPGCritic(state_dim_respond, acceptance_action_dim)
    respond_trainer = build_ddpg_trainer(
        reward_shaper_respond, state_dim_respond,
        acceptance_actor, acceptance_critic, lr, cfg, args.device,
        action_dim=acceptance_action_dim, state_for="respond",
    )

    history = {
        "actor_loss_threshold": [], "critic_loss_threshold": [],
        "actor_loss_propose": [], "critic_loss_propose": [],
        "actor_loss_respond": [], "critic_loss_respond": [],
        "deal_rate": [], "u_learner": [], "u_opponent": [],
        "social_welfare": [], "utility_gap": [], "nash_product": [],
        "concession_learner": [], "concession_opponent": [], "episode_length": [],
    }

    trajectory_buffer = TrajectoryBuffer()
    logger.log("Starting training (DLST-ANESIA)...")

    for episode in range(1, cfg["training"]["episodes"] + 1):
        user_agent = DLSTNegotiator(
            name="learner_dlst",
            ufun=learner_ufun,
            domain_spec=domain_spec,
            bidding_spec=bidding_spec,
            acceptance_spec=acceptance_spec,
            u_fixed=cfg["training"].get("dlst_u_fixed", 0.6),
        )

        user_agent.attach_trainer({
            "threshold_trainer": threshold_trainer,
            "propose_trainer": propose_trainer,
            "respond_trainer": respond_trainer,
        })

        threshold_trainer.attach_agent(user_agent)
        propose_trainer.attach_agent(user_agent)
        respond_trainer.attach_agent(user_agent)

        opponent_agent = RuleBasedAgent(
            name="opponent_rule",
            ufun=opponent_ufun,
            domain_spec=domain_spec,
        )

        env = NegMASBilateralEnv(domain_spec=domain_spec, max_steps=cfg["training"]["max_steps"])
        agreement, _ = env.run(user_agent, opponent_agent)

        payload_propose = user_agent.build_episode_payload_propose(agreement=agreement)
        payload_respond = user_agent.build_episode_payload_respond(agreement=agreement)
        payload_threshold = user_agent.build_episode_payload_threshold(agreement=agreement)

        metrics = compute_episode_metrics(
            agreement=agreement,
            domain_spec=domain_spec,
            max_steps=cfg["training"]["max_steps"],
            learner_ufun=learner_ufun,
            opponent_ufun=opponent_ufun,
            user_agent=user_agent,
            opponent_agent=opponent_agent,
        )

        for payload in [payload_propose, payload_respond, payload_threshold]:
            if payload.meta is None:
                payload.meta = {}
            payload.meta.update(metrics)
            payload.meta["episode"] = episode + 1
            payload.meta["domain_name"] = domain_spec.domain_name
            payload.meta["mode"] = "dlst"

        trajectory_buffer.add_episode(payload_propose)

        ep_path = os.path.join(trajectories_dir, f"episode_{episode+1:04d}.json")
        save_episode_json(ep_path, payload_propose)

        # ── Update (DDPG) ────────────────────────────────────────────
        if episode % cfg["training"]["update_every"] == 0:
            info_threshold = threshold_trainer.update()
            info_propose = propose_trainer.update()
            info_respond = respond_trainer.update()
            threshold_trainer.reset_episodes()
            propose_trainer.reset_episodes()
            respond_trainer.reset_episodes()
        else:
            info_threshold = info_propose = info_respond = {}

        if info_threshold:
            history["actor_loss_threshold"].append(info_threshold.get("actor_loss", 0.0))
            history["critic_loss_threshold"].append(info_threshold.get("critic_loss", 0.0))
        if info_propose:
            history["actor_loss_propose"].append(info_propose.get("actor_loss", 0.0))
            history["critic_loss_propose"].append(info_propose.get("critic_loss", 0.0))
        if info_respond:
            history["actor_loss_respond"].append(info_respond.get("actor_loss", 0.0))
            history["critic_loss_respond"].append(info_respond.get("critic_loss", 0.0))

        history["deal_rate"].append(metrics["deal"])
        history["u_learner"].append(metrics["u_learner"])
        history["u_opponent"].append(metrics["u_opponent"])
        history["social_welfare"].append(metrics["social_welfare"])
        history["utility_gap"].append(metrics["utility_gap"])
        history["nash_product"].append(metrics["nash_product"])
        history["concession_learner"].append(metrics["concession_learner"])
        history["concession_opponent"].append(metrics["concession_opponent"])
        history["episode_length"].append(metrics["episode_length"])

        # ── Checkpoint ───────────────────────────────────────────────
        if episode % cfg["training"]["eval_every"] == 0:
            threshold_trainer.save_checkpoint(
                os.path.join(models_dir, f"checkpoint_threshold_ep{episode}.pt"),
                extra={"episode": episode, "domain_name": domain_spec.domain_name},
            )
            propose_trainer.save_checkpoint(
                os.path.join(models_dir, f"checkpoint_propose_ep{episode}.pt"),
                extra={"episode": episode, "domain_name": domain_spec.domain_name},
            )
            respond_trainer.save_checkpoint(
                os.path.join(models_dir, f"checkpoint_respond_ep{episode}.pt"),
                extra={"episode": episode, "domain_name": domain_spec.domain_name},
            )
            logger.log(f"Checkpoints saved at episode {episode}")

        user_agent.reset_episode()

    # ── Final checkpoints ────────────────────────────────────────────
    threshold_trainer.save_checkpoint(os.path.join(models_dir, "final_checkpoint_threshold.pt"))
    propose_trainer.save_checkpoint(os.path.join(models_dir, "final_checkpoint_propose.pt"))
    respond_trainer.save_checkpoint(os.path.join(models_dir, "final_checkpoint_respond.pt"))

    dataset_path = os.path.join(trajectories_dir, "dataset.jsonl")
    trajectory_buffer.save_jsonl(dataset_path)

    save_json(os.path.join(runs_dir, "training_history.json"), history)

    if cfg["evaluation"]["save_plots"]:
        plot_training_history(history, save_dir=runs_dir)

    logger.log("Training finished (DLST-ANESIA).")
    logger.log(f"Trajectory dataset saved: {dataset_path}")


if __name__ == "__main__":
    main()

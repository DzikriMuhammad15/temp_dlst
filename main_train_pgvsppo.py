import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml
import sys

from agents.rl_agent import RLNegotiator
from agents.rule_agent import RuleBasedAgent
from environments.domain_builder import build_domain
from environments.negmas_env import NegMASBilateralEnv
from evaluation.metrics import compute_episode_metrics
from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork
from training.buffer import TrajectoryBuffer
from reward_shaper.reward_shaper import ProposeRewardConfig, RespondRewardConfig, RewardShaper
from trainers.ppo_trainer import PPOTrainer
from utils.io import ensure_dir, save_json, save_yaml, timestamp
from utils.logger import SimpleLogger
from utils.plotting import plot_training_history
from utils.utility import compute_utility
from training.loss_fns import *
from dataclasses import asdict, is_dataclass
from trainers.policy_gradient_trainer import PolicyGradientTrainer

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

    # convert dataclass -> dict
    if is_dataclass(episode):
        episode = asdict(episode)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(episode, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--offline_dataset", type=str, default=None)
    parser.add_argument("--offline_algo", type=str, default=None, choices=["bc", "pg", "ppo", "mixed"])
    parser.add_argument("--save_prefix", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    set_seed(cfg["project"]["seed"])

    run_id = timestamp()
    prefix = args.save_prefix or run_id

    prefix = prefix + "_ActorCritic"

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

    # ── Reward shaper untuk user (propose) ───────────────────────────
    # Gunakan ProposeRewardConfig untuk menentukan mode reward propose.
    # mode1 = r_bid (Persamaan 9): reward = utility saat sepakat, -1 selain itu.
    propose_reward_cfg_user = ProposeRewardConfig(
        mode=cfg["training"].get("propose_reward_mode", "mode1"),
        utility_scale=cfg["training"]["utility_scale"],
    )
    reward_shaper_propose_user = RewardShaper(
        propose_reward_cfg_user,
        utility_min=domain_spec.utility_min,
        utility_max=domain_spec.utility_max,
        for_="propose",
    )

    # ── Reward shaper untuk user (respond) ───────────────────────────
    # Gunakan RespondRewardConfig untuk menentukan mode reward respond.
    # mode1 = (accept_a, offer_a, penalty=-1): kombinasi dasar dari paper.
    # Lihat RespondRewardConfig docstring untuk daftar lengkap mode.
    respond_reward_cfg_user = RespondRewardConfig(
        mode=cfg["training"].get("respond_reward_mode", "mode1"),
        utility_scale=cfg["training"]["utility_scale"],
    )
    reward_shaper_respond_user = RewardShaper(
        respond_reward_cfg_user,
        utility_min=domain_spec.utility_min,
        utility_max=domain_spec.utility_max,
        for_="respond",
    )

    state_dim_user = 3
    propose_action_dim_user = len(domain_spec.action_library)  # propose: offer only
    respond_action_dim_user = 2                                  # respond: 0=accept, 1=reject

    # ── User: PPO propose trainer ─────────────────────────────────────
    propose_policy_net_user = PolicyNetwork(state_dim_user, propose_action_dim_user).to(args.device)
    propose_value_net_user  = ValueNetwork(state_dim_user).to(args.device)
    propose_policy_optimizer_user = torch.optim.Adam(propose_policy_net_user.parameters(), lr=cfg["training"]["lr"])
    propose_value_optimizer_user  = torch.optim.Adam(propose_value_net_user.parameters(), lr=cfg["training"]["lr"])

    propose_trainer_user = PPOTrainer(
        gamma=cfg["training"]["gamma"],
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gradient_clip_norm=0.5,
        device="cpu",
        epoch=10,
        reward_shaper=reward_shaper_propose_user,
        batch_size=32,
    )
    propose_trainer_user.attach_models({
        "policy": {
            "model": propose_policy_net_user,
            "optimizer": propose_policy_optimizer_user,
            "config": {
                "state_dim": state_dim_user,
                "hidden_dim": 128,
                "action_dim": propose_action_dim_user,
            }
        },
        "value": {
            "model": propose_value_net_user,
            "optimizer": propose_value_optimizer_user,
            "config": {
                "state_dim": state_dim_user,
                "hidden_dim": 128,
            }
        }
    })

    # ── User: PPO respond trainer ─────────────────────────────────────
    respond_policy_net_user = PolicyNetwork(state_dim_user, respond_action_dim_user).to(args.device)
    respond_value_net_user  = ValueNetwork(state_dim_user).to(args.device)
    respond_policy_optimizer_user = torch.optim.Adam(respond_policy_net_user.parameters(), lr=cfg["training"]["lr"])
    respond_value_optimizer_user  = torch.optim.Adam(respond_value_net_user.parameters(), lr=cfg["training"]["lr"])

    respond_trainer_user = PPOTrainer(
        gamma=cfg["training"]["gamma"],
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gradient_clip_norm=0.5,
        device="cpu",
        epoch=10,
        reward_shaper=reward_shaper_respond_user,
        batch_size=32,
    )
    respond_trainer_user.attach_models({
        "policy": {
            "model": respond_policy_net_user,
            "optimizer": respond_policy_optimizer_user,
            "config": {
                "state_dim": state_dim_user,
                "hidden_dim": 128,
                "action_dim": respond_action_dim_user,
            }
        },
        "value": {
            "model": respond_value_net_user,
            "optimizer": respond_value_optimizer_user,
            "config": {
                "state_dim": state_dim_user,
                "hidden_dim": 128,
            }
        }
    })

    # ── Reward shaper untuk opponent (propose) ───────────────────────
    propose_reward_cfg_opponent = ProposeRewardConfig(
        mode=cfg["training"].get("propose_reward_mode", "mode1"),
        utility_scale=cfg["training"]["utility_scale"],
    )
    reward_shaper_propose_opponent = RewardShaper(
        propose_reward_cfg_opponent,
        utility_min=domain_spec.utility_min,
        utility_max=domain_spec.utility_max,
        for_="propose",
    )

    # ── Reward shaper untuk opponent (respond) ───────────────────────
    respond_reward_cfg_opponent = RespondRewardConfig(
        mode=cfg["training"].get("respond_reward_mode", "mode1"),
        utility_scale=cfg["training"]["utility_scale"],
    )
    reward_shaper_respond_opponent = RewardShaper(
        respond_reward_cfg_opponent,
        utility_min=domain_spec.utility_min,
        utility_max=domain_spec.utility_max,
        for_="respond",
    )

    state_dim_opponent = 3
    propose_action_dim_opponent = len(domain_spec.action_library)
    respond_action_dim_opponent = 2

    # ── Opponent: PG propose trainer ─────────────────────────────────
    propose_policy_net_opponent = PolicyNetwork(state_dim_opponent, propose_action_dim_opponent).to(args.device)
    propose_policy_optimizer_opponent = torch.optim.Adam(propose_policy_net_opponent.parameters(), lr=cfg["training"]["lr"])

    propose_trainer_opponent = PolicyGradientTrainer(
        gamma=cfg["training"]["gamma"],
        epoch=1,
        device="cpu",
        batch_size=32,
        reward_shaper=reward_shaper_propose_opponent,
    )
    propose_trainer_opponent.attach_models(
        {
            "policy": {
                "model": propose_policy_net_opponent,
                "optimizer": propose_policy_optimizer_opponent,
                "config": {
                    "state_dim": state_dim_opponent,
                    "hidden_dim": 128,
                    "action_dim": propose_action_dim_opponent,
                }
            }
        }
    )

    # ── Opponent: PG respond trainer ─────────────────────────────────
    respond_policy_net_opponent = PolicyNetwork(state_dim_opponent, respond_action_dim_opponent).to(args.device)
    respond_policy_optimizer_opponent = torch.optim.Adam(respond_policy_net_opponent.parameters(), lr=cfg["training"]["lr"])

    respond_trainer_opponent = PolicyGradientTrainer(
        gamma=cfg["training"]["gamma"],
        epoch=1,
        device="cpu",
        batch_size=32,
        reward_shaper=reward_shaper_respond_opponent,
    )
    respond_trainer_opponent.attach_models(
        {
            "policy": {
                "model": respond_policy_net_opponent,
                "optimizer": respond_policy_optimizer_opponent,
                "config": {
                    "state_dim": state_dim_opponent,
                    "hidden_dim": 128,
                    "action_dim": respond_action_dim_opponent,
                }
            }
        }
    )

    history = {
        "policy_loss_propose_user": [],
        "value_loss_propose_user": [],
        "loss_propose_opponent": [],
        "loss_respond_opponent": [],
        "deal_rate": [],
        "u_learner": [],
        "u_opponent": [],
        "social_welfare": [],
        "utility_gap": [],
        "nash_product": [],
        "concession_learner": [],
        "concession_opponent": [],
        "episode_length": [],
    }

    trajectory_buffer = TrajectoryBuffer()

    logger.log("Starting training...")

    for episode in range(1, cfg["training"]["episodes"] + 1):
        user_agent = RLNegotiator(
            name="learner_rl",
            ufun=learner_ufun,
            domain_spec=domain_spec,
            state_dim=state_dim_user,
        )

        user_agent.attach_trainer({
            "propose_trainer": propose_trainer_user,
            "respond_trainer": respond_trainer_user,
        })

        propose_trainer_user.attach_agent(user_agent)
        respond_trainer_user.attach_agent(user_agent)

        opponent_agent = RLNegotiator(
            name="opponent_rl",
            ufun=opponent_ufun,
            domain_spec=domain_spec,
            state_dim=state_dim_opponent,
        )

        opponent_agent.attach_trainer({
            "propose_trainer": propose_trainer_opponent,
            "respond_trainer": respond_trainer_opponent,
        })

        propose_trainer_opponent.attach_agent(opponent_agent)
        respond_trainer_opponent.attach_agent(opponent_agent)

        env = NegMASBilateralEnv(domain_spec=domain_spec, max_steps=cfg["training"]["max_steps"])
        agreement, _ = env.run(user_agent, opponent_agent)

        episode_payload_propose = user_agent.build_episode_payload_propose(agreement=agreement)
        episode_payload_respond = user_agent.build_episode_payload_respond(agreement=agreement)

        # Metric per episode
        metrics = compute_episode_metrics(
            agreement=agreement,
            domain_spec=domain_spec,
            max_steps=cfg["training"]["max_steps"],
            learner_ufun=learner_ufun,
            opponent_ufun=opponent_ufun,
            user_agent=user_agent,
            opponent_agent=opponent_agent,
        )

        # add meta info to payloads
        for payload in [episode_payload_propose, episode_payload_respond]:
            if payload.meta is None:
                payload.meta = {}
            payload.meta.update(metrics)
            payload.meta["episode"] = episode + 1
            payload.meta["domain_name"] = domain_spec.domain_name
            payload.meta["mode"] = cfg["training"]["mode"]

        # add to trajectory buffer (propose payload sebagai representasi episode)
        trajectory_buffer.add_episode(episode_payload_propose)

        # Simpan per-episode JSON
        ep_path = os.path.join(trajectories_dir, f"episode_{episode+1:04d}.json")
        save_episode_json(ep_path, episode_payload_propose)

        # update
        if (episode) % cfg["training"]["update_every"] == 0:
            update_info_propose_user    = propose_trainer_user.update()
            update_info_respond_user    = respond_trainer_user.update()
            update_info_propose_opp     = propose_trainer_opponent.update()
            update_info_respond_opp     = respond_trainer_opponent.update()
            propose_trainer_user.reset_episodes()
            respond_trainer_user.reset_episodes()
            propose_trainer_opponent.reset_episodes()
            respond_trainer_opponent.reset_episodes()
        else:
            update_info_propose_user = {}
            update_info_respond_user = {}
            update_info_propose_opp  = {}
            update_info_respond_opp  = {}

        if update_info_propose_user:
            if "policy_loss" in update_info_propose_user:
                history["policy_loss_propose_user"].append(update_info_propose_user["policy_loss"])
            if "value_loss" in update_info_propose_user:
                history["value_loss_propose_user"].append(update_info_propose_user["value_loss"])
        if update_info_propose_opp:
            if "loss" in update_info_propose_opp:
                history["loss_propose_opponent"].append(update_info_propose_opp["loss"])
        if update_info_respond_opp:
            if "loss" in update_info_respond_opp:
                history["loss_respond_opponent"].append(update_info_respond_opp["loss"])

        history["deal_rate"].append(metrics["deal"])
        history["u_learner"].append(metrics["u_learner"])
        history["u_opponent"].append(metrics["u_opponent"])
        history["social_welfare"].append(metrics["social_welfare"])
        history["utility_gap"].append(metrics["utility_gap"])
        history["nash_product"].append(metrics["nash_product"])
        history["concession_learner"].append(metrics["concession_learner"])
        history["concession_opponent"].append(metrics["concession_opponent"])
        history["episode_length"].append(metrics["episode_length"])

        # eval and checkpointing
        if (episode) % cfg["training"]["eval_every"] == 0:
            ckpt_path_propose = os.path.join(models_dir, f"checkpoint_propose_ep{episode}.pt")
            ckpt_path_respond = os.path.join(models_dir, f"checkpoint_respond_ep{episode}.pt")
            propose_trainer_user.save_checkpoint(
                ckpt_path_propose,
                extra={
                    "episode": episode,
                    "domain_name": domain_spec.domain_name,
                    "mode": cfg["training"]["mode"],
                    "state_dim": state_dim_user,
                    "propose_action_dim": propose_action_dim_user,
                },
            )
            respond_trainer_user.save_checkpoint(
                ckpt_path_respond,
                extra={
                    "episode": episode,
                    "domain_name": domain_spec.domain_name,
                    "mode": cfg["training"]["mode"],
                    "state_dim": state_dim_user,
                    "respond_action_dim": respond_action_dim_user,
                },
            )
            logger.log(f"Checkpoints saved: {ckpt_path_propose}, {ckpt_path_respond}")

        user_agent.reset_episode()
        opponent_agent.reset_episode()

    # Final checkpoints
    final_ckpt_propose = os.path.join(models_dir, "final_checkpoint_propose.pt")
    final_ckpt_respond = os.path.join(models_dir, "final_checkpoint_respond.pt")
    propose_trainer_user.save_checkpoint(
        final_ckpt_propose,
        extra={
            "episode": cfg["training"]["episodes"],
            "domain_name": domain_spec.domain_name,
            "mode": cfg["training"]["mode"],
            "state_dim": state_dim_user,
            "propose_action_dim": propose_action_dim_user,
        },
    )
    respond_trainer_user.save_checkpoint(
        final_ckpt_respond,
        extra={
            "episode": cfg["training"]["episodes"],
            "domain_name": domain_spec.domain_name,
            "mode": cfg["training"]["mode"],
            "state_dim": state_dim_user,
            "respond_action_dim": respond_action_dim_user,
        },
    )

    # Simpan dataset JSONL
    dataset_path = os.path.join(trajectories_dir, "dataset.jsonl")
    trajectory_buffer.save_jsonl(dataset_path)

    save_json(os.path.join(runs_dir, "training_history.json"), history)

    if cfg["evaluation"]["save_plots"]:
        plot_training_history(history, save_dir=runs_dir)
        for key, value in history.items():
            pass

    logger.log(f"Training finished. Final checkpoints: {final_ckpt_propose}, {final_ckpt_respond}")
    logger.log(f"Trajectory dataset saved: {dataset_path}")


if __name__ == "__main__":
    main()
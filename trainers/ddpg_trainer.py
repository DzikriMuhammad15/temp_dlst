"""
DDPGTrainer — Deep Deterministic Policy Gradient trainer untuk DLST-ANESIA.

Konsisten dengan interface trainer lain di project (store_episode, attach_models,
attach_agent, build_batch, update, reset_episodes, save/load_checkpoint) dan
menggunakan reward_shaper untuk menghitung reward per-step.

Rumus (paper Sec. 5.2):
  Critic loss (Persamaan 18-19):
    L = (1/K) Σ_i ( y_i - Q(s_i, a_i | θ^Q) )^2
    y_i = r_i + γ Q'(s_{i+1}, μ'(s_{i+1} | θ^{μ'}) | θ^{Q'})
  Actor update (Persamaan 16-17, gradient ASCENT atas J):
    ∇_{θμ} J ≈ (1/K) Σ_i ∇_a Q(s,a|θ^Q)|_{a=μ(s_i)} · ∇_{θμ} μ(s|θ^μ)|_{s_i}
  → diimplementasikan sebagai actor_loss = -mean( Q(s, μ(s)) ), lalu backward.

  Target networks μ', Q' di-update secara SOFT:
    θ' ← τ θ + (1 - τ) θ'

Action yang disimpan per-step (episode.actions) adalah VEKTOR continuous
(output actor). Untuk threshold, action berupa vektor berukuran 1 ([ū_t]).
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from trainers.trainer import Trainer

# =========================================================
# MODEL REGISTRY (untuk load_checkpoint)
# Berisi seluruh class network DDPG agar load_checkpoint bisa
# merekonstruksi model dari config tersimpan.
# =========================================================
from models.ddpg_networks import DDPGActorThreshold, DDPGActorStrategy, DDPGCritic

MODEL_REGISTRY = {
    "DDPGActorThreshold": DDPGActorThreshold,
    "DDPGActorStrategy": DDPGActorStrategy,
    "DDPGCritic": DDPGCritic,
}


class DDPGTrainer(Trainer):
    """
    DDPG trainer untuk satu pasang actor-critic (mis. salah satu dari:
    threshold / bidding / acceptance).

    self.models format (lewat attach_models):
    {
        "actor":  {"model": actor_net,  "optimizer": opt_actor,  "config": {...}},
        "critic": {"model": critic_net, "optimizer": opt_critic, "config": {...}},
    }
    """

    def __init__(
        self,
        gamma,
        tau,
        device,
        epoch,
        batch_size,
        reward_shaper,
        gradient_clip_norm=1.0,
        action_dim=None,
        models=None,
        optimizers=None,
        losses=None,
        episodes=None,
        agent=None,
    ):
        super().__init__(
            models=models,
            optimizers=optimizers,
            reward_shaper=reward_shaper,
            losses=losses,
            episodes=episodes,
            batch_size=batch_size,
            agent=agent,
        )
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.epoch = epoch
        self.gradient_clip_norm = gradient_clip_norm
        self.action_dim = action_dim

        # Target networks (diisi saat attach_models)
        self.target_models = {}

    # =====================================================
    # INTERFACE IMPLEMENTATIONS
    # =====================================================

    def store_episode(self, episode):
        self.episodes.append(episode)

    def attach_models(self, dict_of_models):
        """
        Membuat self.models, self.optimizers, dan target networks (copy dari main).
        """
        self.models = {}
        self.optimizers = {}
        self.target_models = {}

        for key, value in dict_of_models.items():
            model = value["model"].to(self.device)
            self.models[key] = {
                "model": model,
                "class_name": model.__class__.__name__,
                "config": value.get("config", {}),
            }
            self.optimizers[key] = value["optimizer"]

            # Target network hanya untuk actor & critic
            target = copy.deepcopy(model).to(self.device)
            for p in target.parameters():
                p.requires_grad_(False)
            self.target_models[key] = target

        if self.action_dim is None:
            # Coba infer dari config actor
            actor_cfg = self.models.get("actor", {}).get("config", {})
            self.action_dim = actor_cfg.get("action_dim", 1)

    def attach_losses(self, dict_of_losses):
        """
        Opsional override:
        {
            "critic_loss": Function(q_pred, y_target) -> scalar,
            "actor_loss":  Function(q_for_policy) -> scalar,
        }
        """
        self.losses = dict_of_losses

    def attach_agent(self, agent):
        self.agent = agent

    # =====================================================
    # DEFAULT LOSSES (paper Persamaan 16-19)
    # =====================================================

    def default_critic_loss(self, q_pred, y_target):
        """MSE antara Q(s,a) dan target y (Persamaan 18)."""
        return F.mse_loss(q_pred, y_target)

    def default_actor_loss(self, q_for_policy):
        """
        Actor memaksimalkan J = E[Q(s, μ(s))] (Persamaan 16).
        Diimplementasikan sebagai meminimalkan -mean(Q).
        """
        return -q_for_policy.mean()

    # =====================================================
    # SOFT UPDATE TARGET NETWORK
    # =====================================================

    def _soft_update(self, key):
        """θ' ← τ θ + (1 - τ) θ' untuk model `key`."""
        main = self.models[key]["model"]
        target = self.target_models[key]
        with torch.no_grad():
            for tp, mp in zip(target.parameters(), main.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * mp.data)

    # =====================================================
    # EPISODE → BATCH
    # =====================================================

    def _extract_states_actions(self, episode):
        """
        State direkonstruksi via agent.build_state(state) agar konsisten
        dengan trainer lain. Action berupa vektor continuous yang disimpan
        agent di step (episode.actions berisi list[vector]).
        """
        # Trainer dibangun per-jenis; agent.build_state dipanggil dengan for_
        # via closure di agent (lihat dlst_agent: build_state mendukung for_).
        state_vecs = [self._build_state(s) for s in episode.states]
        actions = episode.actions if episode.actions is not None else [step.action for step in episode.steps]
        return state_vecs, actions

    def _build_state(self, state):
        """
        Bangun state vector via agent. Memakai self._state_for (di-set oleh
        agent saat attach) supaya build_state tahu jenis arsitektur.
        """
        for_ = getattr(self, "_state_for", None)
        if for_ is not None:
            return self.agent.build_state(state, for_=for_)
        return self.agent.build_state(state)

    def episode_to_batch(self, episode):
        rewards = self.reward_shaper.compute_reward_sequence(episode)
        states, actions = self._extract_states_actions(episode)

        n = len(states)
        if n == 0:
            return None

        # next_states: geser satu; state terakhir → dirinya sendiri (done=True)
        next_states = states[1:] + [states[-1]]
        dones = [False] * n
        dones[-1] = True

        states_np = np.array(states, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)

        # actions bisa berupa skalar (threshold) atau vektor (strategy)
        actions_np = np.array(
            [np.atleast_1d(np.asarray(a, dtype=np.float32)) for a in actions],
            dtype=np.float32,
        )

        states_t = torch.from_numpy(states_np).to(self.device)
        next_states_t = torch.from_numpy(next_states_np).to(self.device)
        actions_t = torch.from_numpy(actions_np).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return {
            "states": states_t,
            "actions": actions_t,
            "rewards": rewards_t,
            "next_states": next_states_t,
            "dones": dones_t,
        }

    def build_batch(self):
        all_states, all_actions, all_rewards, all_next, all_dones = [], [], [], [], []

        for ep in self.episodes:
            pack = self.episode_to_batch(ep)
            if pack is None:
                continue
            all_states.append(pack["states"])
            all_actions.append(pack["actions"])
            all_rewards.append(pack["rewards"])
            all_next.append(pack["next_states"])
            all_dones.append(pack["dones"])

        if len(all_states) == 0:
            return None

        states = torch.cat(all_states, dim=0)
        actions = torch.cat(all_actions, dim=0)
        rewards = torch.cat(all_rewards, dim=0)
        next_states = torch.cat(all_next, dim=0)
        dones = torch.cat(all_dones, dim=0)

        if states.size(0) == 0:
            return None

        dataset = TensorDataset(states, actions, rewards, next_states, dones)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        return loader

    # =====================================================
    # UPDATE (DDPG)
    # =====================================================

    def update(self):
        actor_entry = self.models.get("actor")
        critic_entry = self.models.get("critic")
        if actor_entry is None or critic_entry is None:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "steps": 0}

        actor = actor_entry["model"]
        critic = critic_entry["model"]
        actor_target = self.target_models["actor"]
        critic_target = self.target_models["critic"]

        actor_opt = self.optimizers.get("actor")
        critic_opt = self.optimizers.get("critic")
        if actor_opt is None or critic_opt is None:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "steps": 0}

        critic_loss_fn = self.losses.get("critic_loss", self.default_critic_loss)
        actor_loss_fn = self.losses.get("actor_loss", self.default_actor_loss)

        loader = self.build_batch()
        if loader is None:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "steps": 0}

        actor.train()
        critic.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_steps = 0

        for _ in range(self.epoch):
            for batch in loader:
                states, actions, rewards, next_states, dones = batch
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)

                # ── Critic update (Persamaan 18-19) ──────────────────
                with torch.no_grad():
                    next_actions = actor_target(next_states)
                    q_next = critic_target(next_states, next_actions)
                    y = rewards + self.gamma * (1.0 - dones) * q_next

                q_pred = critic(states, actions)
                critic_loss = critic_loss_fn(q_pred, y)

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), self.gradient_clip_norm)
                critic_opt.step()

                # ── Actor update (Persamaan 16-17) ───────────────────
                pred_actions = actor(states)
                q_for_policy = critic(states, pred_actions)
                actor_loss = actor_loss_fn(q_for_policy)

                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), self.gradient_clip_norm)
                actor_opt.step()

                # ── Soft update target networks ──────────────────────
                self._soft_update("actor")
                self._soft_update("critic")

                bs = states.size(0)
                total_actor_loss += float(actor_loss.item()) * bs
                total_critic_loss += float(critic_loss.item()) * bs
                total_steps += bs

        if total_steps == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "steps": 0}

        return {
            "actor_loss": total_actor_loss / total_steps,
            "critic_loss": total_critic_loss / total_steps,
            "steps": total_steps,
        }

    def reset_episodes(self):
        self.episodes = []

    # =====================================================
    # CHECKPOINT
    # =====================================================

    def save_checkpoint(self, path, extra=None):
        checkpoint = {
            "trainer_state": {
                "gamma": self.gamma,
                "tau": self.tau,
                "epoch": self.epoch,
                "batch_size": self.batch_size,
                "gradient_clip_norm": self.gradient_clip_norm,
                "action_dim": self.action_dim,
            },
            "models": {},
            "optimizers": {},
            "target_models": {},
        }

        if self.models is not None:
            for key, entry in self.models.items():
                model = entry["model"]
                checkpoint["models"][key] = {
                    "class_name": entry["class_name"],
                    "config": entry["config"],
                    "state_dict": model.state_dict(),
                }

        for key, target in self.target_models.items():
            checkpoint["target_models"][key] = target.state_dict()

        if self.optimizers is not None:
            for key, optimizer in self.optimizers.items():
                checkpoint["optimizers"][key] = {
                    "class_name": optimizer.__class__.__name__,
                    "state_dict": optimizer.state_dict(),
                }

        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, path)
        print(f"[INFO] Checkpoint saved: {path}")

    def load_checkpoint(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)

        trainer_state = checkpoint.get("trainer_state", {})
        self.gamma = trainer_state.get("gamma", self.gamma)
        self.tau = trainer_state.get("tau", self.tau)
        self.epoch = trainer_state.get("epoch", self.epoch)
        self.batch_size = trainer_state.get("batch_size", self.batch_size)
        self.gradient_clip_norm = trainer_state.get("gradient_clip_norm", self.gradient_clip_norm)
        self.action_dim = trainer_state.get("action_dim", self.action_dim)

        self.models = {}
        self.target_models = {}

        for key, info in checkpoint.get("models", {}).items():
            class_name = info["class_name"]
            config = info["config"]
            state_dict = info["state_dict"]

            if class_name not in MODEL_REGISTRY:
                raise ValueError(
                    f"Model class '{class_name}' not found in MODEL_REGISTRY"
                )
            model_class = MODEL_REGISTRY[class_name]
            model = model_class(**config).to(self.device)
            model.load_state_dict(state_dict)

            self.models[key] = {
                "model": model,
                "class_name": class_name,
                "config": config,
            }

            # Target network
            target = copy.deepcopy(model).to(self.device)
            for p in target.parameters():
                p.requires_grad_(False)
            target_sd = checkpoint.get("target_models", {}).get(key)
            if target_sd is not None:
                target.load_state_dict(target_sd)
            self.target_models[key] = target

            print(f"[INFO] Loaded model: {key}")

        self.optimizers = {}
        if load_optimizer:
            for key, info in checkpoint.get("optimizers", {}).items():
                if key not in self.models:
                    print(f"[WARNING] No model found for optimizer '{key}'")
                    continue
                optimizer_class_name = info["class_name"]
                optimizer_state = info["state_dict"]
                model = self.models[key]["model"]
                if not hasattr(torch.optim, optimizer_class_name):
                    raise ValueError(
                        f"Optimizer class '{optimizer_class_name}' not found in torch.optim"
                    )
                optimizer_class = getattr(torch.optim, optimizer_class_name)
                optimizer = optimizer_class(model.parameters())
                optimizer.load_state_dict(optimizer_state)
                self.optimizers[key] = optimizer
                print(f"[INFO] Loaded optimizer: {key}")

        print(f"[INFO] Checkpoint loaded from: {path}")

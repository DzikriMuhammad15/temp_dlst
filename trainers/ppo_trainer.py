from trainers.trainer import Trainer

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (
    TensorDataset,
    DataLoader,
)

MODEL_REGISTRY = {}


class PPOTrainer(Trainer):
    def __init__(self, gamma, gae_lambda, clip_ratio, value_coef, entropy_coef, gradient_clip_norm, device, epoch, batch_size, reward_shaper, models=None, optimizers=None, losses=None, episodes=None, agent=None):
        super().__init__(models=models, optimizers=optimizers, reward_shaper=reward_shaper, losses=losses, episodes=episodes, batch_size=batch_size, agent=agent)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gradient_clip_norm = gradient_clip_norm
        self.device = device
        self.epoch = epoch

        # print(f"LOSSES: {self.losses}")

    # region PPO-specific Methods
    def default_policy_loss(self, logits, actions, old_log_probs, advantages, clip_ratio):
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs)

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages

        loss = -torch.min(unclipped, clipped).mean()
        return loss
    
    def default_value_loss(self, values, returns_t):
        return F.mse_loss(values, returns_t)

    def default_entropy_loss(self, logits):
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy().mean()  

    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        values = values + [0.0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values[:-1])]
        return returns, advantages
    # endregion

    #region Interface Implementations
    def store_episode(self, episode):
        # print("KEPANGGIL STORE EPISODE")
        self.episodes.append(episode)
        # print(f"isi self.episodes setelah store_episode: {self.episodes}")

    def attach_models(self, dict_of_models):
        """
        harus:

        {
            "policy (harus policy)": {
                "model": policy_model,
                "optimizer": optimizer,
                "config": {
                    "state_dim": 32,
                    "hidden_dim": 128,
                    "action_dim": 5,
                }
            }, 
            "value (harus value)": {
                "model": value_model,
                "optimizer": optimizer,
                "config": {
                    "state_dim": 32,
                    "hidden_dim": 128,
                }
            }
        }
        """
        self.models = {}
        self.optimizers = {}

        for key, value in dict_of_models.items():
            model = value["model"].to(self.device)

            self.models[key] = {
                "model": model,
                "class_name": model.__class__.__name__,
                "config": value.get("config", {}),
            }

            self.optimizers[key] = value["optimizer"]

    def attach_losses(self, dict_of_losses):
        """
        harus:
        {
            "policy_loss": Function,
            "value_loss": Function,
            "entropy_loss": Function
        }
        """
        self.losses = dict_of_losses

    def attach_agent(self, agent):
        self.agent = agent

    def _extract_states_actions(self, episode):
        state_vecs = [self.agent.build_state(state) for state in episode.states]
        states = state_vecs
        actions = episode.actions or [int(step.action) for step in episode.steps]
        log_probs = episode.log_probs or [step.log_prob for step in episode.steps]
        values = episode.values or [step.value for step in episode.steps]
        return states, actions, log_probs, values

    # TODO: BENERIN STATES NYA KARENA KALAU DARI DATA MASIH BERUPA NEGMAS STATE
    def episode_to_batch(self, episode):
        rewards = self.reward_shaper.compute_reward_sequence(episode)
        states, actions, log_probs, values = self._extract_states_actions(episode)
        dones = [False] * len(rewards)
        if len(dones) > 0:
            dones[-1] = True
        returns, advantages = self._compute_gae(rewards, values, dones)


        states = np.array(states, dtype=np.float32)

        states = torch.from_numpy(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        return_value = {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "returns": returns,
            "advantages": advantages,
        }
        # print(f"return_value_episode_to_batch: {return_value}")
        return return_value
    
    def build_batch(self):
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_advantages = []

        # print(f"isi episode di build_batch: {self.episodes}")

        for ep in self.episodes:
            pack = self.episode_to_batch(ep)

            if pack is None:
                continue

            all_states.append(pack["states"])
            all_actions.append(pack["actions"])
            all_log_probs.append(pack["log_probs"])
            all_returns.append(pack["returns"])
            all_advantages.append(pack["advantages"])
        if len(all_states) == 0:
            return None
        
        states = torch.cat(all_states, dim=0)
        actions = torch.cat(all_actions, dim=0)
        log_probs = torch.cat(all_log_probs, dim=0)
        returns = torch.cat(all_returns, dim=0)
        advantages = torch.cat(all_advantages, dim=0)

        if(len(states) == 0):
            return None

        dataset = TensorDataset(states, actions, log_probs, returns, advantages)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader
    
    def update(self):
        policy = self.models["policy"]["model"]
        value = self.models["value"]["model"]
        policy_optimizer = self.optimizers["policy"]
        value_optimizer = self.optimizers["value"]

        policy_loss_fn = self.losses.get("policy_loss", self.default_policy_loss)
        value_loss_fn = self.losses.get("value_loss", self.default_value_loss)
        entropy_loss_fn = self.losses.get("entropy_loss", self.default_entropy_loss)

        data_loader = self.build_batch()
        if data_loader is None:
            # print(f"data loader none, returning zeros")
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
            }
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_steps = 0

        policy.train()
        value.train()

        for epoch in range(self.epoch):
            # print(f"Epoch {epoch+1}/{self.epoch}")
            for batch in data_loader:
                states, actions, log_probs, returns, advantages = batch
                states = states.to(self.device)
                actions = actions.to(self.device)
                log_probs = log_probs.to(self.device)
                returns = returns.to(self.device)
                advantages = advantages.to(self.device)

                policy_optimizer.zero_grad()
                value_optimizer.zero_grad()

                logits = policy(states)
                values = value(states)
                # print(f"values: {values}")
                values = values.view(-1)
                # print(f"values after view: {values}")

                policy_loss = policy_loss_fn(logits, actions, log_probs, advantages, self.clip_ratio)
                value_loss = value_loss_fn(values, returns)
                entropy_loss = entropy_loss_fn(logits)

                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                # print(f"total loss: {total_loss.item()}")
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.gradient_clip_norm)
                torch.nn.utils.clip_grad_norm_(value.parameters(), self.gradient_clip_norm)

                policy_optimizer.step()
                value_optimizer.step()

                total_policy_loss += policy_loss.item() * states.size(0)
                total_value_loss += value_loss.item() * states.size(0)
                total_entropy_loss += entropy_loss.item() * states.size(0)
                total_steps += states.size(0)

        if total_steps == 0:
            # print(f"total steps is 0, returning zeros")
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
            }
        return {
            "policy_loss": total_policy_loss / total_steps,
            "value_loss": total_value_loss / total_steps,
            "entropy_loss": total_entropy_loss / total_steps,
        }

    def reset_episodes(self):
        self.episodes = []

    def save_checkpoint(self, path, extra=None):
        checkpoint = {
            "trainer_state": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_ratio": self.clip_ratio,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "gradient_clip_norm": self.gradient_clip_norm,
                "epoch": self.epoch,
            },
            "models": {},
            "optimizers": {},
        }
        # =================================================
        # MODELS
        # =================================================

        if self.models is not None:

            for key, entry in (
                self.models.items()
            ):

                model = entry["model"]

                checkpoint["models"][key] = {
                    "class_name": (
                        entry["class_name"]
                    ),
                    "config": (
                        entry["config"]
                    ),
                    "state_dict": (
                        model.state_dict()
                    ),
                }

        # =================================================
        # OPTIMIZERS
        # =================================================

        if self.optimizers is not None:

            for key, optimizer in (
                self.optimizers.items()
            ):

                checkpoint[
                    "optimizers"
                ][key] = {
                    "class_name": (
                        optimizer.__class__.__name__
                    ),
                    "state_dict": (
                        optimizer.state_dict()
                    ),
                }

        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(
            checkpoint,
            path,
        )

        print(
            f"[INFO] Checkpoint saved: "
            f"{path}"
        )

    def load_checkpoint(self, path, load_optimizer=True):
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        trainer_state = checkpoint.get("trainer_state", {})
        self.gamma = trainer_state.get("gamma", self.gamma)
        self.gae_lambda = trainer_state.get("gae_lambda", self.gae_lambda)
        self.clip_ratio = trainer_state.get("clip_ratio", self.clip_ratio)
        self.value_coef = trainer_state.get("value_coef", self.value_coef)
        self.entropy_coef = trainer_state.get("entropy_coef", self.entropy_coef)
        self.gradient_clip_norm = trainer_state.get("gradient_clip_norm", self.gradient_clip_norm)
        self.epoch = trainer_state.get("epoch", self.epoch)
        # =================================================
        # LOAD MODELS
        # =================================================

        self.models = {}

        for key, info in (
            checkpoint.get(
                "models",
                {},
            ).items()
        ):

            class_name = info[
                "class_name"
            ]

            config = info["config"]

            state_dict = info[
                "state_dict"
            ]

            if (
                class_name
                not in MODEL_REGISTRY
            ):
                raise ValueError(
                    f"Model class "
                    f"'{class_name}' "
                    f"not found in "
                    f"MODEL_REGISTRY"
                )

            model_class = (
                MODEL_REGISTRY[
                    class_name
                ]
            )

            model = model_class(
                **config
            ).to(self.device)

            model.load_state_dict(
                state_dict
            )

            self.models[key] = {
                "model": model,
                "class_name": class_name,
                "config": config,
            }

            print(
                f"[INFO] Loaded model: "
                f"{key}"
            )

        # =================================================
        # LOAD OPTIMIZERS
        # =================================================

        self.optimizers = {}

        if load_optimizer:

            for key, info in (
                checkpoint.get(
                    "optimizers",
                    {},
                ).items()
            ):

                if key not in self.models:

                    print(
                        f"[WARNING] "
                        f"No model found "
                        f"for optimizer "
                        f"'{key}'"
                    )

                    continue

                optimizer_class_name = (
                    info["class_name"]
                )

                optimizer_state = (
                    info["state_dict"]
                )

                model = (
                    self.models[key]
                    ["model"]
                )

                if not hasattr(
                    torch.optim,
                    optimizer_class_name,
                ):
                    raise ValueError(
                        f"Optimizer class "
                        f"'{optimizer_class_name}' "
                        f"not found "
                        f"in torch.optim"
                    )

                optimizer_class = getattr(
                    torch.optim,
                    optimizer_class_name,
                )

                optimizer = (
                    optimizer_class(
                        model.parameters()
                    )
                )

                optimizer.load_state_dict(
                    optimizer_state
                )

                self.optimizers[
                    key
                ] = optimizer

                print(
                    f"[INFO] Loaded optimizer: "
                    f"{key}"
                )

        print(
            f"[INFO] Checkpoint loaded "
            f"from: {path}"
        )




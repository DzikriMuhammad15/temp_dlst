from trainers.trainer import Trainer

import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import (
    TensorDataset,
    DataLoader,
)

# =========================================================
# MODEL REGISTRY
# =========================================================
# Tambahkan seluruh model class di sini
#
# Example:
#
# from models.policy_network import PolicyNetwork
#
# MODEL_REGISTRY = {
#     "PolicyNetwork": PolicyNetwork,
# }
#
# =========================================================

MODEL_REGISTRY = {}


class PolicyGradientTrainer(Trainer):
    """
    Simple REINFORCE / Policy Gradient Trainer

    self.models format:
    {
        "policy": {
            "model": policy_model,
            "config": {...}
        }
    }

    self.optimizers format:
    {
        "policy": optimizer
    }
    """

    def __init__(
        self,
        gamma,
        epoch,
        device,
        batch_size,
        reward_shaper,
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
        self.epochs = epoch
        self.device = device
        self.batch_size = batch_size

    def default_loss(self, logits, actions, returns):
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        loss = -(log_probs * returns).mean()
        return loss


    # =====================================================
    # ABSTRACT IMPLEMENTATIONS
    # =====================================================

    def store_episode(self, episode):
        self.episodes.append(episode)

    def attach_models(self, dict_of_models):
        """
        Example:

        {
            "policy (harus policy)": {
                "model": policy_model,
                "optimizer": optimizer,
                "config": {
                    "state_dim": 32,
                    "hidden_dim": 128,
                    "action_dim": 5,
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
                "config": value["config"],
            }

            self.optimizers[key] = value["optimizer"]

    def attach_losses(self, dict_of_losses):
        """
        harus: 
        {
            "policy_loss": Function
        }
        """
        self.losses = dict_of_losses

    def attach_agent(self, agent):
        self.agent = agent

    # =====================================================
    # ADVANTAGES / RETURNS
    # =====================================================

    def compute_advantages_and_returns(
        self,
        rewards,
    ):

        returns = []

        g = 0.0

        for r in reversed(rewards):

            g = r + self.gamma * g

            returns.append(g)

        returns.reverse()

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        advantages = (
            returns - returns.mean()
        )

        std = advantages.std(
            unbiased=False
        )

        if std > 1e-8:
            advantages /= std

        return advantages, returns

    # =====================================================
    # EPISODE PROCESSING
    # =====================================================

    def _extract_states_actions(
        self,
        episode,
    ):
        state_vecs = [self.agent.build_state(state) for state in episode.states]
        states = state_vecs
        actions = episode.actions or [int(step.action) for step in episode.steps]
        log_probs = episode.log_probs or [step.log_prob for step in episode.steps]


        return (
            states,
            actions,
            log_probs,
        )
    
    # TODO: BENERIN STATES NYA KARENA KALAU DARI DATA MASIH BERUPA NEGMAS STATE
    def _episode_to_batch(
        self,
        episode,
    ):

        rewards = self.reward_shaper.compute_reward_sequence(episode)
        states, actions, log_probs = self._extract_states_actions(episode)
        advantages, returns = self.compute_advantages_and_returns(rewards)

        if len(states) == 0 or len(actions) == 0:
            return None
        
        states = np.array(states, dtype=np.float32)

        states = torch.from_numpy(states).to(self.device)

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        log_probs = torch.tensor(
            log_probs,
            dtype=torch.float32,
            device=self.device,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        return {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards,
            "advantages": advantages,
            "returns": returns,
        }

    # =====================================================
    # BUILD BATCH
    # =====================================================

    def build_batch(self):

        all_states = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_advantages = []
        all_returns = []

        for ep in self.episodes:

            pack = self._episode_to_batch(ep)

            if pack is None:
                continue

            all_states.append(
                pack["states"]
            )

            all_actions.append(
                pack["actions"]
            )

            all_log_probs.append(
                pack["log_probs"]
            )

            all_rewards.append(
                pack["rewards"]
            )

            all_advantages.append(
                pack["advantages"]
            )

            all_returns.append(
                pack["returns"]
            )

        if len(all_states) == 0:
            return None

        states = torch.cat(
            all_states,
            dim=0,
        )

        actions = torch.cat(
            all_actions,
            dim=0,
        )

        log_probs = torch.cat(
            all_log_probs,
            dim=0,
        )

        rewards = torch.cat(
            all_rewards,
            dim=0,
        )

        advantages = torch.cat(
            all_advantages,
            dim=0,
        )

        returns = torch.cat(
            all_returns,
            dim=0,
        )

        if states.size(0) == 0:
            return None

        dataset = TensorDataset(
            states,
            actions,
            log_probs,
            rewards,
            advantages,
            returns,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        return loader

    # =====================================================
    # TRAINING
    # =====================================================

    def update(self):

        policy_entry = (
            self.models.get(
                "policy",
                None,
            )
        )

        if policy_entry is None:
            return {
                "loss": 0.0,
                "steps": 0,
            }

        policy = policy_entry["model"]

        optimizer = (
            self.optimizers.get(
                "policy",
                None,
            )
        )

        if optimizer is None:
            return {
                "loss": 0.0,
                "steps": 0,
            }

        loss_fn = self.losses.get(
            "policy_loss",
            self.default_loss,
        )

        if loss_fn is None:
            return {
                "loss": 0.0,
                "steps": 0,
            }

        loader = self.build_batch()

        if loader is None:
            return {
                "loss": 0.0,
                "steps": 0,
            }

        total_loss = 0.0
        total_steps = 0

        policy.train()

        for epoch in range(self.epochs):

            # print(
            #     f"Epoch "
            #     f"{epoch + 1}"
            #     f"/{self.epochs}"
            #     f" - Updating on "
            #     f"{len(loader.dataset)} samples..."
            # )

            for batch in loader:

                (
                    states,
                    actions,
                    log_probs,
                    rewards,
                    advantages,
                    returns,
                ) = batch

                logits = policy(states)

                optimizer.zero_grad()

                loss = loss_fn(
                    logits,
                    actions,
                    returns,
                )

                loss.backward()

                optimizer.step()

                total_loss += (
                    float(loss.item())
                    * states.size(0)
                )

                total_steps += (
                    states.size(0)
                )

        if total_steps == 0:
            return {
                "loss": 0.0,
                "steps": 0,
            }

        return {
            "loss": (
                total_loss
                / total_steps
            ),
            "steps": total_steps,
        }

    # =====================================================
    # RESET
    # =====================================================

    def reset_episodes(self):
        self.episodes = []

    # =====================================================
    # CHECKPOINT SAVE
    # =====================================================

    def save_checkpoint(
        self,
        path,
        extra=None
    ):

        checkpoint = {
            "trainer_state": {
                "gamma": self.gamma,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
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

    # =====================================================
    # CHECKPOINT LOAD
    # =====================================================

    def load_checkpoint(
        self,
        path,
        load_optimizers=True,
        strict=True,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        # =================================================
        # TRAINER STATE
        # =================================================

        trainer_state = checkpoint.get(
            "trainer_state",
            {},
        )

        self.gamma = trainer_state.get(
            "gamma",
            self.gamma,
        )

        self.epochs = trainer_state.get(
            "epochs",
            self.epochs,
        )

        self.batch_size = (
            trainer_state.get(
                "batch_size",
                self.batch_size,
            )
        )

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
                state_dict,
                strict=strict,
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

        if load_optimizers:

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
from trainers.trainer import Trainer

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (
    TensorDataset,
    DataLoader,
)
from data_type.step import Step

MODEL_REGISTRY = {}


class IOPTrainer(Trainer):
    def __init__(self, epoch, device, batch_size, reward_shaper, gradient_clip_norm, models=None, optimizers=None, losses=None, episodes=None, agent=None):
        super().__init__(models=models, optimizers=optimizers, reward_shaper=reward_shaper, losses=losses, episodes=episodes, batch_size=batch_size, agent=agent)
        self.epoch = epoch
        self.device = device
        self.memory = []
        self.agent = None
        self.gradient_clip_norm = gradient_clip_norm

    def reset_step(self):
        self.memory = []

    def add_step(self, step: Step):
        self.memory.append(step)

    def default_loss(self, logits: torch.Tensor, action: torch.Tensor):
        # Ubah logits menjadi log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Ambil log-probability dari action yang dipilih
        selected_log_probs = log_probs.gather(
            dim=1,
            index=action.unsqueeze(1)
        ).squeeze(1)

        # Negative log likelihood
        loss = -selected_log_probs.mean()

        return loss
    
    def store_episode(self, episode):
        pass

    def attach_models(self, dict_of_models):
        """
        Harus: 
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

        atau 

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
                "config": value.get("config", {}),
            }

            self.optimizers[key] = value["optimizer"]

    def attach_agent(self, agent):
        self.agent = agent

    def attach_losses(self, dict_of_losses):
        """
        harus:
        {
            "iop_loss": Function
        }
        """
        self.losses = dict_of_losses

    def _episode_to_batch(self, episode):
        pass

    def build_batch(self):
        all_states = []
        all_actions = []

        # print(f"isi episode di build_batch: {self.episodes}")

        for step in self.memory:
            # print(f"step: {step}")
            if isinstance(step.state, dict):
                state_vec = self.agent.build_state_basis(step.state) # step.state nya masih berupa object negmas state
            else:
                state_vec = step.state # step.state nya berupa state vec langsung
            all_states.append(state_vec)
            all_actions.append(step.action)

        if len(all_states) == 0:
            return None
        
        # print(f"all_states: {all_states}")
        # print(f"all_actions: {all_actions}")
        states = torch.tensor(all_states)
        actions = torch.tensor(all_actions)

        # print(f"states: {states}")
        # print(f"actions: {actions}")

        dataset = TensorDataset(states, actions)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader
    
    def update(self):
        value = None
        value_optimizer = None
        is_using_value = self.models.get("value")
        if(is_using_value):
            value = self.models["value"]["model"]
            value_optimizer = self.optimizers["value"]
        policy = self.models["policy"]["model"]
        policy_optimizer = self.optimizers["policy"]



        loss_fn = self.losses.get("iop_loss", self.default_loss)

        data_loader = self.build_batch()

        if data_loader is None:
            return {
                "loss": 0.0,
            }
        
        total_loss = 0.0
        total_steps = 0

        policy.train()

        for epoch in range(self.epoch):
            for batch in data_loader:
                states, actions = batch
                states = states.to(self.device)
                actions = actions.to(self.device)

                logits = policy(states)
                loss = loss_fn(logits, actions)
                policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.gradient_clip_norm)
                policy_optimizer.step()

                total_loss += loss
                total_steps += states.size(0)

        if total_steps == 0:
            return {
                "loss": 0.0
            }
        
        return {
            "loss": total_loss/total_steps
        }

    def reset_episodes(self):
        pass

    def save_checkpoint(self, path, extra=None):
        checkpoint = {
            "trainer_state": {
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

    def copy_iop_policy_weight(self, iop_policy):
        """
        policy harus dengan model yang sama dan parameter instansiasi object model yang sama (arsitektur yang sama)
        """
        self.models.get("policy").get("model").load_state_dict(iop_policy.state_dict())

    

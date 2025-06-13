# src/asys_i/components/ppo_host.py
"""
Core Philosophy: Separation (Observer-Subject).
Encapsulates the "Subject" model (LLM running PPO).
A-Sys-I observes this process without modifying its core logic.
High Cohesion: Purely PPO training loop.
"""
import logging
import time
from threading import Event
from typing import Callable, Optional

import torch
import torch.nn.functional as F

# Conditional import for TRL
try:
    from datasets import Dataset
    from transformers import AutoModelForCausalLMWithValueHead, AutoTokenizer
    from trl import PPOConfig, PPOTrainer

    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    AutoModelForCausalLMWithValueHead, AutoTokenizer, PPOTrainer, PPOConfig, Dataset = (None,) * 5  # type: ignore
    logging.warning(
        "TRL or Transformers/Datasets not installed. PPOHostProcess will be unavailable."
    )

from torch import nn

from asys_i.common.types import ComponentID, GlobalStep
from asys_i.components.activation_hooker import ActivationHooker  # Type hint only
from asys_i.hpc.resource_manager import bind_current_process
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.orchestration.config_loader import MasterConfig

log = logging.getLogger(__name__)


# ---- MOCK classes for simulation if TRL not available or for testing ----
class MockModel(nn.Module):
    def __init__(self, d_model=768, n_layers=12):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, 1000)  # dummy vocab

    def forward(self, x):
        hidden_states = []
        for layer in self.layers:
            x = F.relu(layer(x))
            hidden_states.append(x)  # Simulate activation points
        return self.head(x), hidden_states

    def generate(self, *args, **kwargs):
        return torch.randint(0, 100, (4, 10))  # Mock generation


class MockPPOTrainer:
    def __init__(self, *args, **kwargs):
        log.warning("MockPPOTrainer initialized")

    def step(self, *args, **kwargs):
        time.sleep(0.05)  # Simulate work
        return {"ppo/loss": torch.rand(1).item(), "ppo/reward": torch.rand(1).item()}

    @property
    def model(self):
        return MockModel()


# -----------------------------------------------------------------------


class PPOHostProcess:
    """
    Manages the lifecycle and execution of the host PPO training process.
    """

    def __init__(
        self,
        config: MasterConfig,
        monitor: BaseMonitor,
        stop_event: Event,
        component_id: ComponentID = "host_process",
    ):
        if not TRL_AVAILABLE:
            log.error("TRL unavailable. Using Mock PPO Host. Real training disabled.")
            # raise ImportError("TRL/Transformers required for PPOHostProcess")

        self.config = config
        self.ppo_config = config.ppo
        self.monitor = monitor
        self.component_id = component_id
        self._global_step: GlobalStep = 0
        self._stop_event = stop_event

        self.monitor.register_component(self.component_id)

        log.info(f"Loading host model: {self.ppo_config.model_name}")
        # TODO: Load model, tokenizer, reward model, setup dataset
        # self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.ppo_config.model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(...)
        # self.ppo_trainer_config = PPOConfig(...)
        # self.ppo_trainer = PPOTrainer(self.ppo_trainer_config, self.model,...)

        # --- MOCK ---
        d_model_val = config.sae_model.d_in
        if not isinstance(d_model_val, int):
            log.warning(
                f"SAE d_in is '{d_model_val}', PPOHost's MockModel will use default d_model=768."
            )
            d_model_val = 768  # Default for GPT2 if "auto" or other non-int

        self.model: nn.Module = MockModel(
            d_model=d_model_val,
            n_layers=(
                max(config.hook.layers_to_hook) + 1 if config.hook.layers_to_hook else 1
            ),
        )
        try:
            self.model.to(config.hardware.device)
            log.info(f"MockModel moved to {config.hardware.device}")
        except RuntimeError as e:
            log.warning(
                f"Failed to move MockModel to {config.hardware.device} ({e}). Using CPU instead."
            )
            self.config.hardware.device = "cpu"  # Fallback to CPU
            self.model.to("cpu")
        self.ppo_trainer = MockPPOTrainer()
        log.warning("PPOHostProcess is RUNNING IN MOCK MODE!")
        # ------------

        if config.hardware.compile_model and hasattr(torch, "compile"):
            log.info("Compiling host model with torch.compile...")
            self.model = torch.compile(self.model)

        log.info("PPOHostProcess initialized.")

    def get_model(self) -> nn.Module:
        """Provides the model instance for ActivationHooker to attach to."""
        # handle compiled model wrapper
        return getattr(self.model, "_orig_mod", self.model)

    def get_global_step(self) -> GlobalStep:
        """Provides the current training step count, for ActivationPacket timestamping."""
        return self._global_step

    def _get_step_callable(self) -> Callable[[], GlobalStep]:
        """Helper to pass a reference to the current step"""
        return lambda: self._global_step

    def run_training_loop(self, hooker: Optional[ActivationHooker] = None):
        """
        Executes the main PPO training loop.
        This method is blocking.
        """
        # Bind CPU cores if HPC
        bind_current_process(self.config, self.component_id, self.monitor)

        log.info(
            f"Starting PPO host training loop for {self.ppo_config.max_steps} steps."
        )
        if hooker:
            log.info("Attaching ActivationHooker...")
            # Pass the callable function, not the current value of _global_step
            hooker.attach(self._get_step_callable())

        last_heartbeat = time.time()
        HEARTBEAT_INTERVAL = 15.0

        try:
            # --- Real loop structure ---
            # for batch in self.dataloader:
            #    queries = ...
            #    responses = self.model.generate(...) # Hooks trigger here
            #    rewards = ...
            #    stats = self.ppo_trainer.step(queries, responses, rewards)
            # ---------------------------

            # --- MOCK Loop ---
            for step in range(self.ppo_config.max_steps):
                if self._stop_event.is_set():
                    log.info("Stop event detected. Exiting training loop.")
                    break

                # --- Simulate Forward Pass to Trigger Hooks ---
                dummy_input = torch.randn(
                    self.ppo_config.batch_size,
                    self.config.sae_model.d_in,
                    device=self.config.hardware.device,
                )
                with torch.no_grad():  # Hooks still trigger
                    _ = self.model(dummy_input)
                # -----------------------------------------------

                stats = self.ppo_trainer.step()  # Simulate PPO step
                self._global_step = step + 1

                self.monitor.log_metrics(
                    stats, step=self._global_step, tags={"source": "ppo"}
                )
                self.monitor.log_metric("ppo_global_step", self._global_step)

                if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                    self.monitor.heartbeat(self.component_id)
                    last_heartbeat = time.time()

                if (step + 1) % 100 == 0:
                    log.info(
                        f"PPO Step {self._global_step}/{self.ppo_config.max_steps} - Loss: {stats.get('ppo/loss',-1):.4f}"
                    )
            # -----------------
            log.info("PPO training loop finished.")

        except KeyboardInterrupt:
            log.warning("PPO loop interrupted by user.")
            self._stop_event.set()
        except Exception:
            log.exception("Exception in PPO training loop:")
            self.monitor.log_metric(
                "host_error_count", 1, tags={"component": self.component_id}
            )
            self._stop_event.set()  # Signal other components to stop
            raise  # Re-raise to be caught by pipeline
        finally:
            if hooker:
                log.info("Detaching ActivationHooker...")
                hooker.detach()
            self.monitor.heartbeat(self.component_id)  # Final heartbeat

    def shutdown(self):
        log.info("Shutting down PPOHostProcess (model/trainer cleanup).")
        self._stop_event.set()
        # Free GPU memory if needed
        self.model = None
        self.ppo_trainer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

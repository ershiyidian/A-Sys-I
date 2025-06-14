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

import random # Added for dummy rewards
import torch # Added for full import list, though already partially used via F

# Conditional import for TRL
try:
    from datasets import Dataset
    from transformers import AutoModelForCausalLMWithValueHead, AutoTokenizer
    from trl import PPOConfig as TRLPPOConfig, PPOTrainer # Aliased PPOConfig

    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    AutoModelForCausalLMWithValueHead, AutoTokenizer, PPOTrainer, TRLPPOConfig, Dataset = (None,) * 5  # type: ignore # Adjusted for alias
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


# MockModel and MockPPOTrainer removed
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

        if not TRL_AVAILABLE:
            log.critical(
                "TRL components are not available. PPOHostProcess cannot function in real mode."
            )
            # According to problem, assume TRL_AVAILABLE is True for modifications.
            # If it were False, an error should ideally be raised or handled gracefully.
            # For now, proceeding with the assumption it's available.
            # raise RuntimeError("TRL components unavailable, cannot initialize PPOHostProcess for real training.")

        log.info(f"Loading host model: {self.ppo_config.model_name}")

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.ppo_config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.ppo_config.model_name)

        # Add special tokens if they don't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            log.info(f"Set tokenizer.pad_token to tokenizer.eos_token: {self.tokenizer.eos_token}")

        try:
            self.model.to(config.hardware.device)
            log.info(f"HuggingFace model moved to {config.hardware.device}")
        except RuntimeError as e:
            log.warning(
                f"Failed to move HuggingFace model to {config.hardware.device} ({e}). Using CPU instead."
            )
            self.config.hardware.device = "cpu"  # Fallback to CPU
            self.model.to("cpu")

        # Initialize PPOConfig from trl (using the aliased TRLPPOConfig)
        ppo_trl_config_args = {
            "model_name": self.ppo_config.model_name,
            "learning_rate": self.ppo_config.learning_rate,
            "batch_size": self.ppo_config.batch_size,
            "mini_batch_size": self.ppo_config.batch_size, # Simplification for now
            "log_with": None, # Disable wandb/tensorboard logging from PPO trainer itself
            "ppo_epochs": 1, # Simplification
            "steps": self.ppo_config.max_steps,
            # Add other necessary PPOConfig fields if they become relevant
        }
        ppo_trl_config_obj = TRLPPOConfig(**ppo_trl_config_args)

        # Dummy dataset for initialization
        # Ensure that the tokenizer has a pad_token set.
        # It's crucial for PPO trainer that input_ids and attention_mask are present.
        if self.tokenizer.pad_token_id is None: # Double check specifically for pad_token_id for tokenizer calls
             self.tokenizer.pad_token = self.tokenizer.eos_token
             log.info(f"Re-checked and set tokenizer.pad_token_id via eos_token for dummy dataset tokenization.")


        dummy_texts = ["hello world", "this is a test"]
        # Tokenize texts and create a dataset
        tokenized_texts = self.tokenizer(dummy_texts, padding=True, truncation=True, return_tensors="pt")
        dummy_dataset_dict = {
            "input_ids": [t for t in tokenized_texts.input_ids],
            "attention_mask": [m for m in tokenized_texts.attention_mask],
            "query": [self.tokenizer.decode(t) for t in tokenized_texts.input_ids],
        }
        dummy_dataset = Dataset.from_dict(dummy_dataset_dict)

        # Initialize PPOTrainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_trl_config_obj,
            model=self.model,
            ref_model=None, # No separate reference model for simplicity
            tokenizer=self.tokenizer,
            dataset=dummy_dataset, # Provide the dummy dataset
        )
        # log.warning("PPOHostProcess is RUNNING IN MOCK MODE!") # This line is removed.

        if config.hardware.compile_model and hasattr(torch, "compile"):
            log.info("Compiling host model with torch.compile...")
            self.model = torch.compile(self.model)

        log.info("PPOHostProcess initialized.")

    def get_model(self) -> nn.Module:
        """Provides the model instance for ActivationHooker to attach to."""
        # PPOTrainer uses an Accelerator which might wrap the model.
        # We need the underlying nn.Module for hooks.
        # First, unwrap the model from the accelerator.
        if hasattr(self.ppo_trainer, 'accelerator'):
            unwrapped_model = self.ppo_trainer.accelerator.unwrap_model(self.ppo_trainer.model)
        else:
            # Fallback if somehow accelerator is not present (e.g. custom PPOTrainer version)
            unwrapped_model = self.ppo_trainer.model

        # Then, handle the torch.compile wrapper if it was applied.
        return getattr(unwrapped_model, "_orig_mod", unwrapped_model)

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

            # --- Real Loop Structure (Simplified) ---
            for step in range(self.ppo_config.max_steps): # Keep existing loop structure
                if self._stop_event.is_set():
                    log.info("Stop event detected. Exiting training loop.")
                    break

                # Generate dummy query tensors (batch of tokenized text)
                dummy_queries_str = ["simulate query 1"] * self.ppo_config.batch_size
                query_tokens = self.tokenizer(
                    dummy_queries_str, padding=True, truncation=True, return_tensors="pt"
                ).to(self.config.hardware.device)

                # query_tensors_list is a list of 1D tensors [batch_size, query_length]
                query_tensors_list = [q_ids for q_ids in query_tokens["input_ids"]]

                # Generate responses using ppo_trainer.generate
                # Hooks will trigger inside model.forward() called by generate() and step()
                generation_kwargs = {
                    "min_length": -1, # avoid warning if pad_token_id is eos_token_id
                    "top_k": 0.0,
                    "top_p": 1.0,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "max_new_tokens": 10, # Keep responses short for dummy data
                    # "output_scores": True, # Not needed for step, but useful for debugging generation
                }
                # PPOTrainer.generate expects List[torch.Tensor] (queries)
                # It returns List[torch.Tensor] (responses)
                response_tensors_list = self.ppo_trainer.generate(query_tensors_list, **generation_kwargs)
                # response_tensors_list are on the correct device as per PPOTrainer's internal handling.

                # Dummy rewards (batch_size list of scalars)
                dummy_rewards = [torch.tensor(random.random(), device=self.config.hardware.device) for _ in range(self.ppo_config.batch_size)]

                # The PPO step. Model's forward pass (and thus hooks) will be triggered internally by TRL.
                # PPOTrainer.step expects: List[torch.Tensor] queries, List[torch.Tensor] responses, List[torch.Tensor] scores
                stats = self.ppo_trainer.step(query_tensors_list, response_tensors_list, dummy_rewards)

                self._global_step = step + 1

                self.monitor.log_metrics(
                    stats, step=self._global_step, tags={"source": "ppo"}
                )
                self.monitor.log_metric("ppo_global_step", self._global_step)

                if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                    self.monitor.heartbeat(self.component_id)
                    last_heartbeat = time.time()

                if (step + 1) % 100 == 0:
                    # TRL PPOTrainer stats often include 'ppo/mean_scores' or 'ppo/rewards/mean' etc.
                    # Using a general name that might be present, with a fallback.
                    reward_metric = stats.get('ppo/mean_scores', stats.get('ppo/mean_reward', stats.get('ppo/rewards/mean', -1)))
                    log.info(
                        f"PPO Step {self._global_step}/{self.ppo_config.max_steps} - Mean PPO Reward: {reward_metric:.4f}"
                    )
            # --- End of Real Loop Structure (Simplified) ---
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

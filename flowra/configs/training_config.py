"""
Training Configuration

Configuration for the training loop, optimization, and learning rate scheduling.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
from enum import Enum


class OptimizerType(Enum):
    """Available optimizers."""
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    ADAFACTOR = "adafactor"
    LION = "lion"
    SOPHIA = "sophia"


class SchedulerType(Enum):
    """Available learning rate schedulers."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_warmup"
    POLYNOMIAL = "polynomial"
    INVERSE_SQRT = "inverse_sqrt"
    ONE_CYCLE = "one_cycle"


class LossType(Enum):
    """Available loss functions."""
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"
    NLL = "nll"
    CONTRASTIVE = "contrastive"
    CUSTOM = "custom"


@dataclass
class OptimizerConfig:
    """
    Optimizer configuration.
    """
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    
    # Common parameters
    lr: float = 1e-4
    weight_decay: float = 0.01
    
    # Adam/AdamW specific
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # SGD specific
    momentum: float = 0.9
    nesterov: bool = False
    
    # Gradient clipping
    max_grad_norm: Optional[float] = 1.0
    
    # Layer-wise learning rates
    layer_wise_lr_decay: Optional[float] = None  # e.g., 0.95 for decay
    
    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for optimizer constructor."""
        if self.optimizer_type in [OptimizerType.ADAMW, OptimizerType.ADAM]:
            return {
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            }
        elif self.optimizer_type == OptimizerType.SGD:
            return {
                "lr": self.lr,
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                "weight_decay": self.weight_decay,
            }
        else:
            return {"lr": self.lr, "weight_decay": self.weight_decay}


@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler configuration.
    """
    scheduler_type: SchedulerType = SchedulerType.COSINE
    
    # Warmup
    warmup_steps: int = 0
    warmup_ratio: float = 0.0  # Alternative: fraction of total steps
    
    # Cosine specific
    min_lr_ratio: float = 0.0  # Final LR as ratio of initial LR
    num_cycles: float = 0.5    # For cosine with restarts
    
    # Polynomial specific
    power: float = 1.0
    
    # One cycle specific
    max_lr: Optional[float] = None
    pct_start: float = 0.3
    
    def get_warmup_steps(self, total_steps: int) -> int:
        """Calculate warmup steps."""
        if self.warmup_steps > 0:
            return self.warmup_steps
        return int(total_steps * self.warmup_ratio)


@dataclass
class DataConfig:
    """
    Data loading configuration.
    """
    batch_size: int = 32
    eval_batch_size: Optional[int] = None  # Defaults to batch_size
    
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    
    # Sequence settings (for language models)
    max_length: Optional[int] = None
    padding: str = "max_length"
    truncation: bool = True
    
    # Augmentation
    use_augmentation: bool = False
    augmentation_config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def effective_eval_batch_size(self) -> int:
        return self.eval_batch_size or self.batch_size


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    
    Attributes
    ----------
    epochs : int
        Number of training epochs.
    
    max_steps : Optional[int]
        Maximum training steps (overrides epochs if set).
    
    optimizer : OptimizerConfig
        Optimizer configuration.
    
    scheduler : SchedulerConfig
        Learning rate scheduler configuration.
    
    data : DataConfig
        Data loading configuration.
    
    eval_strategy : str
        Evaluation strategy: "no", "steps", "epoch"
    
    eval_steps : int
        Evaluate every N steps if eval_strategy is "steps".
    
    save_strategy : str
        Checkpoint strategy: "no", "steps", "epoch", "best"
    
    save_steps : int
        Save every N steps if save_strategy is "steps".
    
    logging_steps : int
        Log metrics every N steps.
    
    use_dynamic_refinement : bool
        Whether to use progressive subspace refinement.
    
    refinement_interval : int
        Steps between refinement checks.
    
    early_stopping_patience : Optional[int]
        Stop after N evaluations without improvement.
    
    gradient_accumulation_steps : int
        Number of gradient accumulation steps.
    
    mixed_precision : bool
        Whether to use mixed precision training.
    
    compile_model : bool
        Whether to use torch.compile for speedup.
    """
    
    # Duration
    epochs: int = 3
    max_steps: Optional[int] = None
    
    # Optimization
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Data
    data: DataConfig = field(default_factory=DataConfig)
    
    # Loss
    loss_type: LossType = LossType.CROSS_ENTROPY
    label_smoothing: float = 0.0
    
    # Evaluation
    eval_strategy: str = "epoch"
    eval_steps: int = 500
    eval_delay: int = 0  # Skip first N steps
    
    # Checkpointing
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: Optional[int] = 3
    save_best_only: bool = False
    
    # Logging
    logging_steps: int = 10
    logging_first_step: bool = True
    report_to: List[str] = field(default_factory=lambda: ["console"])
    
    # FLOWRA specific
    use_dynamic_refinement: bool = False
    refinement_interval: int = 100
    refinement_warmup: int = 100
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
    
    # Gradient handling
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    
    # Performance
    mixed_precision: bool = False
    fp16: bool = False
    bf16: bool = False
    compile_model: bool = False
    
    # Reproducibility
    seed: Optional[int] = None
    deterministic: bool = False
    
    # Distributed training
    local_rank: int = -1
    ddp_find_unused_parameters: bool = False
    
    # Output
    output_dir: str = "./flowra_output"
    run_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.eval_batch_size is None:
            self.data.eval_batch_size = self.data.batch_size
        
        if self.fp16 or self.bf16:
            self.mixed_precision = True
    
    @property
    def eval_batch_size(self) -> int:
        return self.data.effective_eval_batch_size
    
    def get_total_steps(self, num_train_examples: int) -> int:
        """Calculate total training steps."""
        if self.max_steps is not None:
            return self.max_steps
        
        steps_per_epoch = num_train_examples // (
            self.data.batch_size * self.gradient_accumulation_steps
        )
        return steps_per_epoch * self.epochs
    
    def should_evaluate(self, step: int, epoch: int) -> bool:
        """Check if evaluation should run at current step."""
        if step < self.eval_delay:
            return False
        
        if self.eval_strategy == "no":
            return False
        elif self.eval_strategy == "epoch":
            return False  # Handled by epoch completion
        elif self.eval_strategy == "steps":
            return step > 0 and step % self.eval_steps == 0
        return False
    
    def should_save(self, step: int, epoch: int) -> bool:
        """Check if checkpoint should be saved at current step."""
        if self.save_strategy == "no":
            return False
        elif self.save_strategy == "epoch":
            return False  # Handled by epoch completion
        elif self.save_strategy == "steps":
            return step > 0 and step % self.save_steps == 0
        return False
    
    def should_log(self, step: int) -> bool:
        """Check if logging should occur at current step."""
        if step == 0 and self.logging_first_step:
            return True
        return step > 0 and step % self.logging_steps == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "optimizer": {
                "type": self.optimizer.optimizer_type.value,
                "lr": self.optimizer.lr,
                "weight_decay": self.optimizer.weight_decay,
            },
            "scheduler": {
                "type": self.scheduler.scheduler_type.value,
                "warmup_steps": self.scheduler.warmup_steps,
            },
            "data": {
                "batch_size": self.data.batch_size,
                "num_workers": self.data.num_workers,
            },
            "use_dynamic_refinement": self.use_dynamic_refinement,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
        }


# Preset configurations
def get_default_training_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_fast_training_config() -> TrainingConfig:
    """Get configuration for fast training (fewer epochs, larger batch)."""
    return TrainingConfig(
        epochs=1,
        data=DataConfig(batch_size=64),
        optimizer=OptimizerConfig(lr=3e-4),
        eval_strategy="no",
        save_strategy="no",
    )


def get_careful_training_config() -> TrainingConfig:
    """Get configuration for careful training (more evaluation, lower LR)."""
    return TrainingConfig(
        epochs=10,
        optimizer=OptimizerConfig(lr=5e-5),
        scheduler=SchedulerConfig(
            scheduler_type=SchedulerType.COSINE_WARMUP,
            warmup_ratio=0.1,
        ),
        eval_strategy="steps",
        eval_steps=100,
        early_stopping_patience=5,
        use_dynamic_refinement=True,
    )

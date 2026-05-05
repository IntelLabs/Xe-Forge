"""
Configuration Manager for Triton Optimizer
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from xe_forge.models import OptimizationStage


@dataclass
class LLMConfig:
    """LLM configuration"""

    model: str = "openai/gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 8192
    api_base: str | None = None
    api_key: str | None = None


@dataclass
class AgentConfig:
    """Agent configuration"""

    max_iterations: int = 5
    use_cover: bool = True
    strategy: str = "cover"  # cover, react, hybrid


@dataclass
class OptimizationConfig:
    """Optimization pipeline configuration"""

    enabled_stages: list[OptimizationStage] = field(
        default_factory=lambda: [
            OptimizationStage.ANALYSIS,
            OptimizationStage.ALGORITHMIC,
            OptimizationStage.DTYPE_FIX,
            OptimizationStage.FUSION,
            OptimizationStage.MEMORY_ACCESS,
            OptimizationStage.BLOCK_POINTERS,
            OptimizationStage.PERSISTENT_KERNEL,
            OptimizationStage.XPU_SPECIFIC,
            OptimizationStage.AUTOTUNING,
        ]
    )
    max_attempts_per_stage: int = 3
    validate_each_stage: bool = True
    best_k: int = 1

    # Correctness validation settings
    require_correctness: bool = True  # If False, skip output comparison
    correctness_rtol: float = 1e-2  # Relative tolerance for correctness check
    correctness_atol: float = 1e-5  # Absolute tolerance for correctness check

    target_speedup: float = 2.0  # Minimum acceptable speedup
    target_dtype: str | None = None  # "float16", "bfloat16", etc.


@dataclass
class XPUConfig:
    """Intel XPU specific configuration"""

    device: str = "xpu"
    grf_mode: str = "256"  # XPU register file mode
    default_num_warps: int = 32  # XPU prefers 32 warps
    default_num_stages: int = 2
    preferred_tile_m: int = 256
    preferred_tile_n: int = 256
    preferred_tile_k: int = 32
    group_size_m: int = 4


@dataclass
class KnowledgeConfig:
    """Knowledge base configuration.

    Set enabled=True and point knowledge_dir at a directory containing
    the YAML pattern files to load the knowledge base at startup.
    When enabled=False (default) the pipeline relies entirely on the
    LLM's built-in knowledge — no YAML files are required.
    """

    enabled: bool = False
    knowledge_dir: str = "./knowledge_base"
    # auto_load kept for backward compatibility; ignored when enabled=False
    auto_load: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: str = "INFO"
    log_dir: str = "./outputs/logs"
    kernel_dir: str = "./outputs/kernels"
    save_intermediate: bool = True


@dataclass
class TrialConfig:
    """Trial-tree exploration configuration.

    Drives the new tree-structured search that replaces the flat best_k loop.
    Each trial is a node in a tree; the search policy picks the parent to
    branch from and when to stop, and the writer produces the child kernel.
    """

    max_trials: int = 10
    early_stop_speedup: float = 5.0
    plateau_window: int = 2
    search: str = "tree_walk"       # tree_walk, best_first, beam, mcts
    writer: str = "stage_sequence"  # stage_sequence, cover, react, explorer
    trials_dir: str = "./trials"
    output_dir: str = "./output"


@dataclass
class ProfilerConfig:
    """VTune profiler configuration.

    When vtune_enabled is True and the trial search signals a plateau, the
    runner calls xpu_profiler.profile() on the current best trial and the
    resulting markdown report is forwarded into OptimizerAgent.vtune_report.
    """

    vtune_enabled: bool = False
    vtune_bin: str | None = None


@dataclass
class Config:
    """Master configuration"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    xpu: XPUConfig = field(default_factory=XPUConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    trial: TrialConfig = field(default_factory=TrialConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)


class ConfigManager:
    """Centralized configuration manager that loads from .env and allows overrides"""

    def __init__(self, env_path: Path | None = None):
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv(verbose=True)
        self.config = self._load_config()

    def _get_env(self, key: str, default: Any = None, cast_type: type = str) -> Any:
        """Get environment variable with type casting"""
        value = os.getenv(key, default)
        if value is None:
            return default
        if cast_type is bool:
            return str(value).lower() in ("true", "1", "yes", "on")
        if cast_type is list:
            return [x.strip() for x in str(value).split(",")]
        return cast_type(value)

    def _load_config(self) -> Config:
        """Load configuration from environment variables"""

        # LLM Configuration
        llm = LLMConfig(
            model=self._get_env("LLM_MODEL", "openai/gpt-4o"),
            temperature=self._get_env("LLM_TEMPERATURE", 0.1, float),  # GPT need 1.0
            max_tokens=self._get_env("LLM_MAX_TOKENS", 8192, int),  # 16k for gpt
            api_base=self._get_env("OPENAI_API_BASE"),
            api_key=self._get_env("OPENAI_API_KEY"),
        )

        # Agent Configuration
        agent = AgentConfig(
            max_iterations=self._get_env("AGENT_MAX_ITERATIONS", 5, int),
            use_cover=self._get_env("USE_COVER", True, bool),
            strategy=self._get_env("AGENT_STRATEGY", "cover"),
        )

        # Optimization Configuration
        optimization = OptimizationConfig(
            max_attempts_per_stage=self._get_env("MAX_ATTEMPTS_PER_STAGE", 3, int),
            validate_each_stage=self._get_env("VALIDATE_EACH_STAGE", True, bool),
            best_k=self._get_env("BEST_K", 1, int),
            require_correctness=self._get_env("REQUIRE_CORRECTNESS", True, bool),
            correctness_rtol=self._get_env("CORRECTNESS_RTOL", 1e-2, float),
            correctness_atol=self._get_env("CORRECTNESS_ATOL", 1e-5, float),
            target_speedup=self._get_env("TARGET_SPEEDUP", 2.0, float),
            target_dtype=self._get_env("TARGET_DTYPE", None),
        )

        # Trial Exploration Configuration
        # BEST_K aliases onto TRIAL_MAX_TRIALS when TRIAL_MAX_TRIALS is unset,
        # so existing BEST_K=N users keep working during the transition.
        trial_max_trials = self._get_env("TRIAL_MAX_TRIALS", None, int)
        if trial_max_trials is None:
            trial_max_trials = optimization.best_k if optimization.best_k > 1 else 10
        trial = TrialConfig(
            max_trials=trial_max_trials,
            early_stop_speedup=self._get_env("TRIAL_EARLY_STOP_SPEEDUP", 5.0, float),
            plateau_window=self._get_env("TRIAL_PLATEAU_WINDOW", 2, int),
            search=self._get_env("TRIAL_SEARCH", "tree_walk"),
            writer=self._get_env("TRIAL_WRITER", "stage_sequence"),
            trials_dir=self._get_env("TRIALS_DIR", "./trials"),
            output_dir=self._get_env("OUTPUT_DIR", "./output"),
        )

        # Profiler Configuration
        profiler = ProfilerConfig(
            vtune_enabled=self._get_env("VTUNE_ENABLED", False, bool),
            vtune_bin=self._get_env("VTUNE_BIN", None),
        )

        # XPU Configuration
        xpu = XPUConfig(
            device=self._get_env("XPU_DEVICE", "xpu"),
            grf_mode=self._get_env("GRF_MODE", "256"),
            default_num_warps=self._get_env("DEFAULT_NUM_WARPS", 32, int),
            default_num_stages=self._get_env("DEFAULT_NUM_STAGES", 2, int),
            preferred_tile_m=self._get_env("PREFERRED_TILE_M", 256, int),
            preferred_tile_n=self._get_env("PREFERRED_TILE_N", 256, int),
            preferred_tile_k=self._get_env("PREFERRED_TILE_K", 32, int),
            group_size_m=self._get_env("GROUP_SIZE_M", 4, int),
        )

        # Knowledge Configuration
        knowledge = KnowledgeConfig(
            enabled=self._get_env("KNOWLEDGE_BASE_ENABLED", False, bool),
            knowledge_dir=self._get_env("KNOWLEDGE_DIR", "./knowledge_base"),
            auto_load=self._get_env("AUTO_LOAD_KNOWLEDGE", True, bool),
        )

        # Logging Configuration
        logging_cfg = LoggingConfig(
            level=self._get_env("LOG_LEVEL", "INFO"),
            log_dir=self._get_env("LOG_DIR", "./outputs/logs"),
            kernel_dir=self._get_env("KERNEL_DIR", "./outputs/kernels"),
            save_intermediate=self._get_env("SAVE_INTERMEDIATE", True, bool),
        )

        return Config(
            llm=llm,
            agent=agent,
            optimization=optimization,
            xpu=xpu,
            knowledge=knowledge,
            logging=logging_cfg,
            trial=trial,
            profiler=profiler,
        )

    def override(self, **kwargs) -> "ConfigManager":
        """Override configuration values programmatically"""
        for key, value in kwargs.items():
            parts = key.split("_", 1)
            if len(parts) == 2:
                section, attr = parts
                if hasattr(self.config, section):
                    section_obj = getattr(self.config, section)
                    if hasattr(section_obj, attr):
                        setattr(section_obj, attr, value)
        return self

    def get(self) -> Config:
        """Get the loaded configuration"""
        return self.config


# Global config instance
_config_manager: ConfigManager | None = None


def get_config(env_path: Path | None = None, reload: bool = False) -> Config:
    """Get global configuration instance"""
    global _config_manager
    if _config_manager is None or reload:
        _config_manager = ConfigManager(env_path)
    return _config_manager.get()


def override_config(**kwargs) -> Config:
    """Override configuration values"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    _config_manager.override(**kwargs)
    return _config_manager.get()

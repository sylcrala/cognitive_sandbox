"""
Configuration management for the Cognitive Sandbox.
Uses a singleton pattern to ensure consistent configuration across the application.
"""
from pathlib import Path
import datetime as dt
import uuid
from typing import Set


class AppConfig:
    """
    Centralized configuration singleton for the Cognitive Sandbox.
    
    Attributes:
        session_id (str): Unique identifier for the current session.
        BASE_DIR (Path): Root directory of the main package.
        MEM_DIR (Path): Directory for particle memory storage.
        LOGS_DIR (Path): Directory for application logs.
        max_file_size (int): Maximum size for individual memory files (~5MB).
        simulated_input (bool): Whether to simulate environment events.
        rolling_reflection_window (int): Window size for reflection analysis.
        ERROR_flag (Set): Set to track application errors.
    """
    _instance = None

    def __new__(cls):
        """Implement singleton pattern - only one instance per application."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize configuration only once."""
        if self._initialized:
            return
        
        # unique session identifier
        self.session_id = "session_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

        # path configurations
        self.BASE_DIR = Path(__file__).resolve().parent
        self.MEM_DIR = self.BASE_DIR / "memories"
        self.LOGS_DIR = self.BASE_DIR / "logs"
        self.MEM_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # file size limits
        self.max_file_size = 5_000_000  # ~5MB

        # simulation flag for particles
        self.simulated_input = True

        # Rolling window for tracking "new" reflections (in aggregate_reflections calls)
        # Higher = new reflections show longer before being counted as "seen"
        self.rolling_reflection_window = 100

        # ==================== REPRODUCTION & POPULATION ====================
        self.max_population = 250  # Hard cap on particle count
        self.energy_threshold_reproduction = 0.75  # Min energy to reproduce (lowered from 0.85)
        self.rhythm_sync_window = 0.3  # Max rhythm diff for sync (|diff| < window)
        self.reproduction_energy_cost_base = 0.5  # Base energy cost for reproduction
        self.offspring_initial_energy = 0.4  # Offspring starting energy
        self.reproduction_check_interval = 600  # Ticks between reproduction checks
        
        # ==================== ENERGY & DECAY ====================
        self.max_energy = 1.0  # Global cap for particle energy
        self.max_activation = 1.0  # Global cap for particle activation
        self.base_energy_decay = 0.995  # Baseline energy decay per tick (was 0.98)
        self.valence_energy_adjust_factor = 0.02  # How much valence affects decay (was 0.05)
        self.min_energy_decay = 0.99  # Floor on energy decay (was 0.95)
        self.base_activation_decay = 0.995  # Baseline activation decay (was 0.98)
        self.valence_activation_boost = 0.05  # Acceleration of decay with negative valence (was 0.1)
        self.energy_regen_base = 0.006  # Base energy regeneration per tick (was 0.00325)
        self.energy_regen_rhythm_bonus = 0.004  # Regen bonus from rhythm alignment (was 0.0035)
        self.energy_regen_valence_bonus = 0.005  # Regen bonus from positive valence (was 0.0045)
        
        # ==================== ACTIVATION & VITALITY ====================
        self.activation_gain_threshold = 0.5  # Vitality threshold for activation gain (lowered from 0.8)
        self.activation_gain_factor = 0.25  # Multiplier for activation gain
        self.social_activation_boost = 0.015  # Boost per neighbor (capped at 0.08)
        self.social_activation_cap = 0.08  # Max social boost per tick
        self.isolation_activation_penalty = 0.002  # Activation loss when isolated
        self.min_activation = 0.1  # Minimum activation floor (prevents instant death)
        self.vitality_energy_weight = 0.5  # Energy contribution to vitality
        self.vitality_activation_weight = 0.3  # Activation contribution to vitality
        self.vitality_rhythm_weight = 0.2  # Rhythm alignment contribution to vitality
        
        # ==================== PHYSICS & FORCES ====================
        self.velocity_damping = 0.87  # Velocity decay factor per tick
        self.force_application_scale = 0.8  # Scale applied to all calculated forces
        self.long_range_force_scale = 0.002  # Inverse-square force magnitude
        self.repulsion_force_scale = 0.015  # Collision repulsion magnitude
        self.rhythm_drift_coefficient = 0.0015  # Temporal anchor strength
        self.rhythm_fluctuation_stddev = 0.008  # Random rhythm jitter
        self.temporal_anchor_increment = 0.001  # How much position[4] increases per tick
        self.energy_exchange_rate = 0.012  # Rate of energy transfer between neighbors
        self.energy_exchange_cap = 0.005  # Max energy exchanged per interaction
        self.emotional_contagion_rate = 0.015  # Rate of rhythm/emotion spreading
        
        # ==================== ENVIRONMENTAL RHYTHM ====================
        self.env_rhythm_base = 0.3  # Base rhythm level
        self.env_rhythm_amplitude = 0.3  # Oscillation amplitude
        self.env_rhythm_frequency = 0.01  # Oscillation frequency (tick_count * this)
        
        # ==================== LINGUISTIC GENERATION ====================
        self.word_memory_reuse_chance = 0.6  # Chance to reuse/mutate existing word
        self.word_memory_generate_chance = 0.4  # Chance to generate entirely new word
        self.word_mutation_chance = 0.7  # Chance to mutate word slightly vs reuse exactly
        self.word_reuse_exact_chance = 0.3  # Chance to reuse word exactly
        self.word_length_intensity_range = 0.4  # How much intensity affects word length
        self.word_generation_cost_base = 0.01  # Base energy cost for word generation
        self.word_generation_cost_per_length = 0.015  # Cost multiplier per word length
        self.word_generation_cost_valence_discount = 0.08  # Discount for positive valence
        self.syntax_pattern_inheritance_chance = 0.9  # Chance to inherit syntax pattern
        self.type_mutation_chance = 0.1  # Chance to mutate particle type on reproduction
        
        # ==================== STATE PERSISTENCE ====================
        self.memory_save_interval = 300  # Ticks between memory saves
        self.full_state_backup_interval = 600  # Ticks between full state backups
        
        # ==================== ENVIRONMENTAL EVENTS ====================
        self.environmental_event_interval = 100  # Ticks between random inspirations
        self.rhythm_adjustment_interval = 150  # Ticks between rhythm adjustments
        self.random_reflection_interval = 60  # Ticks for random reflection events
        self.random_reflection_chance = 0.5  # Chance per trigger for random reflection
        
        self.ERROR_flag: Set[str] = set()
        
        self._initialized = True


def get_config() -> AppConfig:
    """
    Get the global AppConfig singleton instance.
    
    Returns:
        AppConfig: The single global configuration instance.
    """
    return AppConfig()

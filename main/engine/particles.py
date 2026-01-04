"""
The main particle class utilized by all particles/agents in the Cognitive Sandbox.
Manages individual particle state, physics, memory, and behavior.
"""
import uuid
import datetime as dt
import numpy as np
import random
import math
import time
from typing import List, Dict, Any, Optional


class Particle:
    """
    Individual autonomous particle/agent in the simulation.
    
    Represents an entity with:
    - 11-dimensional state (spatial + temporal + emotional)
    - Energy and activation levels
    - Memory bank for reflections and interactions
    - Type-based behavioral variations
    - Dynamic emotional state (rhythm and valence)
    
    Attributes:
        field (ParticleField): Reference to the parent field.
        id (uuid.UUID): Unique identifier.
        name (str): Human-readable name (agent-XXXX).
        position (np.ndarray): 11D state vector.
        velocity (np.ndarray): 11D velocity vector.
        energy (float): Energy level [0.0, 1.03].
        activation (float): Activation level [0.0, 1.0].
        type (str): Behavior type (cooperative, avoidant, etc.).
        alive (bool): Whether particle is still active.
        memory_bank (List[Dict]): List of memories and reflections.
    """
    
    def __init__(
        self,
        field: Optional[Any] = None,
        id: Optional[uuid.UUID] = None,
        name: Optional[str] = None,
        energy: float = 0.0,
        activation: float = 0.0
    ) -> None:
        """
        Initialize a new particle.
        
        Args:
            field: Reference to ParticleField instance.
            id: UUID (generated if not provided).
            name: Human-readable name (generated if not provided).
            energy: Initial energy level (default: random [0.5, 1.0]).
            activation: Initial activation (default: random [0.1, 0.4]).
        """
        self.field = field

        self.id = uuid.uuid4() if id is None else id
        self.logger = self.field.logger
        self.name = f"agent-{str(self.id)[:4]}" if name is None else name

        self.type = random.choice(["cooperative", "avoidant", "chaotic", "inquisitive", "dormant", "resonant"])

        self.memory_bank = []

        self.position = np.zeros(11)
        self.velocity = np.random.uniform(-0.01, 0.01, 11)

        vec3 = np.random.uniform(0, 1, 3)                           # random 3D vector
        self.position[0] = vec3[0]                                  # x
        self.position[1] = vec3[1]                                  # y
        self.position[2] = vec3[2]                                  # z
        self.position[3] = dt.datetime.now().timestamp()            # w
        self.position[4] = 0.0                                      # t ( current time )
        self.position[5] = 0.0                                      # age
        self.position[6] = np.random.uniform(-1,1)                  # emotional rhythm
        self.position[8] = np.random.uniform(-1,1)                  # valence

        self.activation = activation or random.uniform(0.1,0.4)
        self.activation = max(self.activation, self.field.config.min_activation)  # Enforce minimum
        self.energy = energy or random.uniform(0.7, 1.0)  # Start closer to cap
        self.t = self.position[4]
        
        # individual caps (can vary by type or inheritance)
        self.max_energy = self.field.config.max_energy
        self.max_activation = self.field.config.max_activation

        self.embeddings = {}
        self.policies = {}
        self.interaction_weights = {}

        # genealogy tracking
        self.parent_ids: List[uuid.UUID] = []  # List of parent UUIDs (max 2)
        self.child_ids: List[uuid.UUID] = []   # List of offspring UUIDs
        self.generation = 0  # Generation number (0 = original)

        # linguistic emergence system - track word usage and syntax preferences
        self.word_context_frequencies: Dict[str, Dict[str, int]] = {}  # {context: {word: count}}
        self.preferred_syntax_patterns: Dict[str, Dict[str, int]] = {}  # {context: {syntax_type: count}}
        
        # voice profile shapes phonetic generation
        self.voice_profile = {
            "consonant_preference": random.uniform(0.3, 0.7),   # [0.3, 0.7] = harsh vs soft
            "vowel_openness": random.uniform(0.2, 0.8),         # [0.2, 0.8] = open(a,o) vs closed(e,i)
            "syllable_tendency": random.uniform(0.5, 2.5),      # [0.5, 2.5] = short vs long words
            "expressiveness": random.uniform(0.4, 1.2),         # [0.4, 1.2] = variation magnitude
        }

        self.alive = True
        self.last_reflection = None



    def update(self, env_rhythm: float, neighbors: List["Particle"]) -> None:
        """
        Update particle state for one simulation tick.
        
        Handles:
        - Position and velocity updates
        - Energy and activation decay
        - Rhythm-based vitality scoring
        - Energy regeneration
        
        Args:
            env_rhythm: Current environmental rhythm value [-1.0, 1.0].
            neighbors: List of nearby particles for context.
        """
        for i in range(11):
            self.position[i] += self.velocity[i] * 0.05
            self.velocity[i] *= 0.95
        # Activation decay handled below via config.base_activation_decay

        now = dt.datetime.now().timestamp()
        self.position[4] = now  # update localized time
        self.position[5] = now - self.position[3] # update age
        self.last_updated = now

        vitality = self.vitality_score(env_rhythm)

        if not math.isfinite(self.activation):
            self.logger.error(f"Invalid activation for particle {self.name} | {self.id} at {now}")
            self.activation = 0.0

        if not math.isfinite(self.energy) or self.energy < 0:
            self.logger.error(f"Invalid energy for particle {self.name} | {self.id} at {now}")
            self.energy = 0.0

        if not math.isfinite(vitality):
            self.logger.error(f"Invalid vitality for particle {self.name} | {self.id} at {now}")
            vitality = 0.0

        ## particle energy and activation decay + energy check
        if self.activation < 0.001:
            self.alive = False

        # valence influence on decay
        # positive valence = less decay (keep more energy)
        # negative valence = more decay (lose more energy)
        valence = self.position[8]
        valence_norm = (valence + 1) / 2  # 0 = negative, 1 = positive

        base_decay = self.field.config.base_energy_decay
        # Positive valence reduces decay (closer to 1.0 = keep more)
        # valence_norm=1 (happy): decay approaches base_decay (minimal penalty)
        # valence_norm=0 (sad): decay reduced by valence_energy_adjust_factor
        valence_decay_penalty = self.field.config.valence_energy_adjust_factor * (1 - valence_norm)
        energy_decay = base_decay - valence_decay_penalty
        
        activation_decay = self.field.config.base_activation_decay

        # clamping decay to floor
        energy_decay = max(energy_decay, self.field.config.min_energy_decay)

        # negative valence accelerates activation decay
        if valence < 0:
            activation_decay *= (1 + self.field.config.valence_activation_boost * abs(valence))

        # applying decay
        self.energy *= energy_decay
        self.activation *= activation_decay


        if vitality > self.field.config.activation_gain_threshold:
            gain = min((vitality - self.field.config.activation_gain_threshold), 0.5) * self.field.config.activation_gain_factor * self.energy
            self.activation += gain
            self.energy -= gain * 0.1
        
        rhythm_align = 1.0 - abs(self.position[6] - env_rhythm)  # in [0, 1]
        valence_norm = (self.position[8] + 1) / 2  # in [0, 1]

        regen = (self.field.config.energy_regen_base + 
                (self.field.config.energy_regen_rhythm_bonus * rhythm_align) + 
                (self.field.config.energy_regen_valence_bonus * valence_norm))
        self.energy += regen

        self.clamp_state()

    def adjust_behavior(self, neighbors: List["Particle"], particle_context: Dict[str, Any]) -> None:
        """
        Adjust particle behavior based on neighbors and environment.
        
        Implements:
        - Spatial attraction with weak temporal "present moment" anchor
        - Collision repulsion to prevent stacking
        - Emotional contagion (both positive and negative rhythm spread)
        - Symmetric energy exchange (not gravity-based)
        - Type-based behavior variations
        - Minimal temporal drift using only active time (position[4])
        
        Args:
            neighbors: List of nearby particles.
            particle_context: Dict with "all_particles" and "total_energy".
        """
        # ==================== SPATIAL DRIFT + MINIMAL TEMPORAL ANCHOR ====================
        # Spatial noise for natural exploration
        spatial_noise = np.random.normal(0, 0.015, 3)  # Only spatial dims (x, y, z)
        drift_force = np.zeros(11)
        drift_force[0:3] = spatial_noise
        
        # Weak "present moment" anchor using only current time (position[4])
        # Uses configurable temporal drift coefficient
        avg_current_time = np.mean([p.position[4] for p in particle_context["all_particles"]])
        temporal_drift = (avg_current_time - self.position[4]) * self.field.config.rhythm_drift_coefficient
        drift_force[4] = temporal_drift + np.random.normal(0, self.field.config.rhythm_fluctuation_stddev)
        
        # Weak boundary repulsion to keep particles in [0,1] bounds
        boundary_force = np.zeros(11)
        for i in range(3):
            if self.position[i] < 0.1:
                boundary_force[i] += 0.02
            elif self.position[i] > 0.9:
                boundary_force[i] -= 0.02
        
        repulsion_force = np.zeros(11)
        attraction_force = np.zeros(11)
        
        if not neighbors and self.activation < 0.2:
            self.activation += 0.01  # slow self-wake
        
        # ==================== NEIGHBOR INTERACTIONS ====================
        if neighbors:
            # Only use spatial dimensions for local center
            spatial_center = np.zeros(3)
            for n in neighbors:
                spatial_center += n.position[0:3]
            spatial_center /= len(neighbors)
            
            # Weak attraction to local spatial center (reduced from 0.05 to 0.02)
            spatial_diff = spatial_center - self.position[0:3]
            attraction_force[0:3] = spatial_diff * 0.02
            
            # Type-based behavior variations
            if self.type == "avoidant":
                # Avoidant particles actively avoid neighbors
                attraction_force[0:3] *= 0.5  # Reduce attraction
            elif self.type == "cooperative":
                # Cooperative particles are more attracted
                attraction_force[0:3] *= 1.2
            
            # ==================== COLLISION REPULSION ====================
            # Hard repulsion when particles get too close (prevents stacking)
            for neighbor in neighbors:
                spatial_diff = self.position[0:3] - neighbor.position[0:3]
                spatial_dist = np.linalg.norm(spatial_diff)
                
                # Strong repulsion at very close range
                if spatial_dist < 0.12:
                    if spatial_dist < 1e-6:
                        spatial_diff = np.array([np.random.randn(), np.random.randn(), np.random.randn()])
                        spatial_dist = 1e-3
                    
                    repulsion_strength = max(0.08, (0.12 - spatial_dist) / 0.12)
                    repulsion_force[0:3] += (spatial_diff / spatial_dist) * repulsion_strength
                
                if neighbor is self:
                    continue
                
                # ==================== SYMMETRIC ENERGY EXCHANGE ====================
                # Energy exchange only between direct neighbors (not group-based)
                # Make it symmetric and minimal to prevent energy centralization
                energy_diff = (self.energy - neighbor.energy) * self.field.config.energy_exchange_rate
                energy_diff = np.clip(energy_diff, -self.field.config.energy_exchange_cap, self.field.config.energy_exchange_cap)
                
                self.energy -= energy_diff
                neighbor.energy += energy_diff
                
                # ==================== EMOTIONAL CONTAGION ====================
                # Both positive AND negative rhythm can spread
                # Stronger from high-energy neighbors
                rhythm_diff = self.position[6] - neighbor.position[6]
                neighbor_influence = neighbor.energy * self.field.config.emotional_contagion_rate
                
                # Both high and low rhythms spread (contagion)
                adjustment = rhythm_diff * neighbor_influence
                adjustment = np.clip(adjustment, -0.04, 0.04)
                
                self.position[6] -= adjustment * 0.5
                neighbor.position[6] += adjustment * 0.5
                
                # Add random rhythm fluctuation to prevent convergence
                self.position[6] += np.random.normal(0, self.field.config.rhythm_fluctuation_stddev)
        
        else:
            attraction_force = np.zeros(11)
        
        # ==================== ACTIVATION BOOST ====================
        # Social interaction boosts activation, isolation penalizes it
        if neighbors:
            social_boost = min(len(neighbors) * self.field.config.social_activation_boost, self.field.config.social_activation_cap)
            self.activation += social_boost * 0.5
        else:
            # Isolation penalty - particles lose activation when alone
            self.activation -= self.field.config.isolation_activation_penalty
        
        # ==================== VELOCITY UPDATE ====================
        # Apply configurable damping and force scaling
        self.velocity = [
            self.velocity[i] * self.field.config.velocity_damping + (attraction_force[i] + repulsion_force[i] + drift_force[i] + boundary_force[i]) * self.field.config.force_application_scale
            for i in range(11)
        ]
        
        self.clamp_state()


    def clamp_state(self) -> None:
        """
        Clamp particle state values to valid ranges.
        
        Ensures:
        - energy: [0.0, max_energy]
        - activation: [0.0, max_activation]
        - emotional rhythm: [-1.0, 1.0]
        - valence: [-1.0, 1.0]
        """
        # Safety fallback for particles from old saves without max attributes
        max_e = getattr(self, 'max_energy', self.field.config.max_energy)
        max_a = getattr(self, 'max_activation', self.field.config.max_activation)
        
        self.energy = min(max(self.energy, 0.0), max_e)
        self.activation = min(max(self.activation, 0.0), max_a)


    def vitality_score(self, env_rhythm: float) -> float:
        """
        Calculate particle vitality based on energy, activation, and rhythm alignment.
        
        Vitality represents "health" or "aliveness" of the particle.
        Uses configurable weights for each component.
        
        Args:
            env_rhythm: Current environmental rhythm [-1.0, 1.0].
            
        Returns:
            Vitality score (typically [0.0, 1.0]).
        """
        rhythm_alignment = 1.0 - abs(self.position[6] - env_rhythm)  # [0, 1]
        
        # Weighted combination of attributes
        energy_contrib = self.energy * self.field.config.vitality_energy_weight
        activation_contrib = self.activation * self.field.config.vitality_activation_weight
        rhythm_contrib = rhythm_alignment * self.field.config.vitality_rhythm_weight
        
        return energy_contrib + activation_contrib + rhythm_contrib

    def distance_to(self, other: "Particle") -> float:
        """
        Calculate Euclidean distance to another particle in 11D space.
        
        Args:
            other: Another Particle instance.
            
        Returns:
            Distance value.
        """
        return math.sqrt(sum(
            (self.position[i] - other.position[i]) ** 2 for i in range(11)
        ))

    def freeze(self) -> bool:
        """
        Check if particle is frozen (dead and stationary).
        
        Returns:
            True if particle is no longer alive and not moving.
        """
        return not self.alive and np.allclose(self.velocity, 0.0)
    
    def reflect(self, neighbors: List["Particle"]) -> None:
        """
        Generate fully procedural reflection from particle state and memory.
        
        Uses emergent syntax based on emotional/energetic context.
        No hardcoded templates - all expression derived from particle metrics.
        
        Args:
            neighbors: List of nearby particles at reflection time.
        """
        now = dt.datetime.now().timestamp()
        
        # Determine emotional context from internal state
        context = self._get_emotional_context()
        
        # Generate core emotive words using voice profile
        primary_word = self.generate_emotive_string(context)
        secondary_word = self.generate_emotive_string(context)
        
        # Determine syntax structure from particle state
        syntax_structure = self._determine_syntax_structure(context, neighbors)
        
        # Assemble reflection from procedural components
        reflection_parts = self._assemble_reflection_parts(
            primary_word,
            secondary_word,
            syntax_structure,
            context,
            neighbors
        )
        
        self.last_reflection = " ".join(reflection_parts)
        
        # Store with linguistic metadata
        self.memory_bank.append({
            "id": str(self.id),
            "name": self.name,
            "timestamp": now,
            "reflection": self.last_reflection,
            "context": context,
            "words_used": [primary_word, secondary_word],
            "syntax_type": syntax_structure["type"],
            "generation": self.generation,
            "persisted": False
        })
        
        # Limit memory bank size to prevent unbounded growth
        while len(self.memory_bank) > 20:
            self.memory_bank.pop(0)
        
        # Track word usage for future reflections
        self._track_word_usage(context, [primary_word, secondary_word])
        self._track_syntax_usage(context, syntax_structure["type"])
        
        self.logger.log(f"Reflection [{context}]: {self.last_reflection}")
    
    def _get_emotional_context(self) -> str:
        """
        Infer emotional context from particle state metrics.
        Determines which syntactic/linguistic patterns to use.
        """
        valence = self.position[8]
        rhythm = self.position[6]
        energy = self.energy
        activation = self.activation
        
        # Multi-factor emotional determination
        if valence > 0.6 and energy > 0.65:
            return "joyful_energetic"
        elif valence < -0.6 and activation < 0.3:
            return "despondent"
        elif energy < 0.2:
            return "exhausted"
        elif abs(rhythm) > 0.7:
            return "synchronized"
        elif len(self.child_ids) > 0 and energy > 0.5:
            return "generative"
        elif len(self.parent_ids) > 0 and activation > 0.6:
            return "inherited"
        elif len(self.memory_bank) > 5:
            return "nostalgic"
        else:
            return "neutral"
    
    def _determine_syntax_structure(self, context: str, neighbors: list) -> dict:
        """
        Derive syntax rules from particle state and emotional context.
        Different states produce different syntactic patterns.
        """
        metrics = {
            "energy": self.energy,
            "activation": self.activation,
            "has_neighbors": len(neighbors) > 0,
            "neighbor_count": len(neighbors),
            "has_children": len(self.child_ids) > 0,
            "has_parents": len(self.parent_ids) > 0,
            "memory_depth": len(self.memory_bank),
        }
        
        # High energy + activation = rapid staccato output
        if metrics["energy"] > 0.8 and metrics["activation"] > 0.6:
            return {
                "type": "rapid_series",
                "repetitions": random.randint(2, 4),
                "separator": ". ",
                "intensifier": True,
            }
        
        # Low energy = sparse, minimal output
        elif metrics["energy"] < 0.3:
            return {
                "type": "sparse",
                "repetitions": 1,
                "separator": " ",
                "intensifier": False,
            }
        
        # Synchronized with neighbors = connected chain
        elif metrics["has_neighbors"] and metrics["activation"] > 0.5:
            return {
                "type": "linked_chain",
                "repetitions": min(metrics["neighbor_count"], 3),
                "separator": " - ",
                "intensifier": False,
            }
        
        # Has offspring = branching structure
        elif metrics["has_children"]:
            return {
                "type": "branching",
                "repetitions": min(len(self.child_ids), 3),
                "separator": " :: ",
                "intensifier": True,
            }
        
        # Default balanced structure
        else:
            return {
                "type": "balanced",
                "repetitions": 2,
                "separator": " ",
                "intensifier": False,
            }
    
    def _assemble_reflection_parts(self, primary: str, secondary: str,
                                    syntax: dict, context: str, neighbors: list) -> list:
        """
        Procedurally assemble reflection using voice-generated words and syntax patterns.
        No hardcoded grammar - purely emergent assembly.
        """
        parts = []
        struct_type = syntax["type"]
        
        # === RAPID SERIES: staccato bursts with intensification ===
        if struct_type == "rapid_series":
            for i in range(syntax["repetitions"]):
                word = self.generate_emotive_string(context)
                if syntax["intensifier"] and i > 0:
                    word = self._morph_word(word, "intensify")
                parts.append(word)
            return parts
        
        # === SPARSE: minimal output ===
        elif struct_type == "sparse":
            parts.append(primary)
            if syntax["intensifier"]:
                parts.append(self._morph_word(primary, "diminish"))
            return parts
        
        # === LINKED CHAIN: connected neighbor references ===
        elif struct_type == "linked_chain":
            for neighbor in neighbors[:syntax["repetitions"]]:
                word = self.generate_emotive_string(context)
                # Mix neighbor initial + generated word for unique pattern
                neighbor_sig = neighbor.name[6:10].lower()  # Last 4 of UUID
                parts.append(f"{neighbor_sig}-{word}")
            return parts
        
        # === BRANCHING: offspring-like morphological variants ===
        elif struct_type == "branching":
            root_word = primary
            parts.append(root_word)
            for i in range(syntax["repetitions"]):
                child_word = self._morph_word(root_word, "derive_child")
                parts.append(child_word)
            return parts
        
        # === BALANCED: default two-word structure ===
        else:
            parts.append(primary)
            if syntax["repetitions"] > 1:
                parts.append(secondary)
            return parts
    
    def _track_word_usage(self, context: str, words: list) -> None:
        """Record which words were used in which contexts for future reference."""
        if context not in self.word_context_frequencies:
            self.word_context_frequencies[context] = {}
        
        for word in words:
            self.word_context_frequencies[context][word] = \
                self.word_context_frequencies[context].get(word, 0) + 1
        
        # Limit vocabulary size per context to top 50 words (prevent memory leak)
        if len(self.word_context_frequencies[context]) > 50:
            sorted_words = sorted(
                self.word_context_frequencies[context].items(),
                key=lambda x: x[1], reverse=True
            )[:50]
            self.word_context_frequencies[context] = dict(sorted_words)
    
    def _track_syntax_usage(self, context: str, syntax_type: str) -> None:
        """Record which syntax patterns are used in which contexts."""
        if context not in self.preferred_syntax_patterns:
            self.preferred_syntax_patterns[context] = {}
        
        self.preferred_syntax_patterns[context][syntax_type] = \
            self.preferred_syntax_patterns[context].get(syntax_type, 0) + 1
        
        # Limit syntax patterns per context to top 20 (prevent memory leak)
        if len(self.preferred_syntax_patterns[context]) > 20:
            sorted_patterns = sorted(
                self.preferred_syntax_patterns[context].items(),
                key=lambda x: x[1], reverse=True
            )[:20]
            self.preferred_syntax_patterns[context] = dict(sorted_patterns)
    
    def _morph_word(self, word: str, morphology_type: str) -> str:
        """
        Apply morphological transformation to create word variants.
        Creates word families and inflectional patterns.
        """
        if morphology_type == "intensify":
            # Double final syllable or add harsh consonant
            if len(word) > 2:
                return word + word[-2:] + "x"
            else:
                return word + "xx"
        
        elif morphology_type == "diminish":
            # Shorten or soften
            if len(word) > 3:
                return word[:-1]
            else:
                return word[0]
        
        elif morphology_type == "derive_child":
            # Shift one vowel for "offspring" derivation
            chars = list(word)
            if len(chars) > 1:
                for i, char in enumerate(chars):
                    if char in "aeiouy":
                        new_vowel = random.choice("aeiouy")
                        return "".join(chars[:i] + [new_vowel] + chars[i+1:])
            return word + "a"  # Fallback: add "a" suffix
        
        else:
            return word
    
    # DEPRECATED: Old hardcoded reflect() method - keeping for reference
    def _reflect_deprecated_old(self, neighbors: List["Particle"]) -> None:
        """DEPRECATED: This hardcoded template-based reflect is no longer used."""
        pass  # See new reflect() method above for current implementation


    def random_reflection(self):
        now = dt.datetime.now().timestamp()

        self.logger.log("Generating random reflection")

        seed_word = f"random_{self.type}_{int(self.energy * 100)}"
        emotive_word = str(self.generate_emotive_string(seed_word))

        self.last_reflection = f"{emotive_word}."

        self.memory_bank.append({
            "id": str(self.id),
            "name": f"{self.name} - SELF",
            "valence": float(self.position[8]),
            "timestamp": now,
            "reflection": self.last_reflection,
            "persisted": False
        })
        if len(self.memory_bank) > 10:
            self.memory_bank.pop(0)

        self.logger.log(f"Random reflection generated: {self.last_reflection}")

    def save_state(self):
        """
        Serialize particle state for persistence.
        
        Saves all particle attributes including genealogy and linguistic data
        to enable full state reconstruction on reload.
        
        Returns:
            Dict with complete particle state.
        """
        # Helper to convert numpy types to native Python types
        def to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            return obj
        
        # convert numpy arrays to lists if needed
        pos = self.position.tolist() if hasattr(self.position, "tolist") else self.position
        vel = self.velocity.tolist() if hasattr(self.velocity, "tolist") else self.velocity

        return {
            # Core identity and physics
            "id": str(self.id),
            "name": self.name,
            "type": self.type,
            "position": pos,
            "velocity": vel,
            "activation": float(self.activation),
            "energy": float(self.energy),
            "max_energy": float(self.max_energy),
            "max_activation": float(self.max_activation),
            "alive": self.alive,
            
            # Genealogy
            "generation": int(self.generation),
            "parent_ids": [str(p) for p in self.parent_ids],
            "child_ids": [str(c) for c in self.child_ids],
            
            # Linguistic attributes (convert any numpy types)
            "voice_profile": to_native(self.voice_profile),
            "word_context_frequencies": to_native(self.word_context_frequencies),
            "preferred_syntax_patterns": to_native(self.preferred_syntax_patterns),
            
            # Memory and reflections
            "last_reflection": self.last_reflection,
            "memory_bank": to_native(self.memory_bank)
        }
    


    def generate_emotive_string(self, emotional_context="neutral", from_memory=True) -> str:
        """
        Generate expressive word using voice profile and optionally memory.
        
        First tries to recall/mutate words from memory (60% chance),
        otherwise generates entirely new word from voice profile (40% chance).
        
        This creates linguistic continuity and word families while maintaining
        variation through mutation and emotional context modulation.
        
        Args:
            emotional_context: The emotional state context ("joyful_energetic", "despondent", etc).
            from_memory: Whether to check memory for previously used words first.
            
        Returns:
            Generated or recalled emotive word.
        """
        # Try to use or mutate words from memory first
        if from_memory and self.memory_bank and random.random() < self.field.config.word_memory_reuse_chance:
            past_words = []
            for mem in self.memory_bank:
                if mem.get("context") == emotional_context:
                    past_words.extend(mem.get("words_used", []))
            
            if past_words:
                # Mutate an old word slightly to create variation
                old_word = random.choice(past_words)
                if random.random() < self.field.config.word_mutation_chance:  # Configurable mutation chance
                    return self._mutate_word_slightly(old_word)
                else:
                    return old_word  # Reuse exactly (inverse of mutation chance)
        
        # Generate entirely new word from voice profile
        return self._generate_new_word(emotional_context)
    
    def _generate_new_word(self, emotional_context: str) -> str:
        """
        Synthetically generate new word using voice profile and emotional context.
        Voice profile shapes: consonant ratio, vowel selection, word length.
        """
        # Determine word length from voice profile + emotional intensity
        intensity = self.energy * self.activation  # [0, 1]
        base_length = int(3 + (self.voice_profile["syllable_tendency"] * 4))
        word_length = max(3, int(base_length * (0.8 + intensity * self.field.config.word_length_intensity_range)))
        
        # Get consonant/vowel ratios
        consonant_ratio = self.voice_profile["consonant_preference"]
        
        # Modulate by emotional context
        if "joyful" in emotional_context:
            consonant_ratio *= 0.8  # More vowels = softer
        elif "trouble" in emotional_context or "despondent" in emotional_context:
            consonant_ratio *= 1.2  # More consonants = harsher
        
        consonant_ratio = np.clip(consonant_ratio, 0.3, 0.8)
        
        # Select vowel set based on openness
        openness = self.voice_profile["vowel_openness"]
        if openness > 0.6:
            vowels = "aeiou"  # More open vowels
        elif openness < 0.4:
            vowels = "iy"  # More closed vowels
        else:
            vowels = "aeiouy"  # Balanced
        
        consonants = "bcdfghjklmnpqrstvwxz"
        
        # Generate phonetic word
        word = ""
        for i in range(word_length):
            if random.random() < consonant_ratio:
                word += random.choice(consonants)
            else:
                word += random.choice(vowels)
        
        # Smooth common phonetic violations
        word = word.replace("cc", "c").replace("vv", "v")
        
        # Apply energy cost for generation
        valence = self.position[8]
        valence_norm = (valence + 1) / 2
        cost = self.field.config.word_generation_cost_base + self.field.config.word_generation_cost_per_length * (word_length / 7)
        cost *= (1 - self.field.config.word_generation_cost_valence_discount * valence_norm)  # Slightly cheaper for positive valence
        self.energy = max(self.energy - cost, 0.0)
        
        return word.capitalize()
    
    def _mutate_word_slightly(self, word: str) -> str:
        """
        Apply minor phonetic variation to create word family.
        Preserves overall phonetic character while showing variation.
        """
        if len(word) < 2:
            return word
        
        mutation_type = random.choice(["vowel_shift", "suffix", "truncate"])
        
        if mutation_type == "vowel_shift":
            # Shift one vowel
            for i, char in enumerate(word):
                if char.lower() in "aeiouy":
                    new_vowel = random.choice("aeiouy")
                    return word[:i] + new_vowel + word[i+1:]
        
        elif mutation_type == "suffix":
            # Add common suffix
            suffixes = ["a", "o", "e", "ix", "th", "us"]
            return word + random.choice(suffixes)
        
        else:  # truncate
            # Remove last character
            return word[:-1] if len(word) > 2 else word
        
        return word
    
    # DEPRECATED: Old generate_emotive_string using hardcoded seed-based generation
    def _generate_emotive_string_deprecated(self, seed_word=None) -> str:
        """DEPRECATED: This seed-based approach is superseded by voice_profile generation."""
        pass  # See new generate_emotive_string() method above for current implementation


    def adaptive_component(self, other):
        vec_a = self.embeddings.get(self.id, np.zeros(3))
        vec_b = self.embeddings.get(other.id, np.zeros(3))
        dist = np.linalg.norm(vec_a - vec_b)

        score = self.get_interaction_weight(other.id)
        mod_factor = 1 - (score - 0.5)

        base_adaptive = dist * mod_factor
        policy_a = self.policies.get(self.id, lambda d: d)
        policy_b = self.policies.get(other.id, lambda d: d)

        return (policy_a(base_adaptive) + policy_b(base_adaptive)) / 2

    def set_policy(self, other_id, strategy="resonant"):
        strategies = {
            "cooperative": lambda d: d * 0.75,
            "avoidant": lambda d: d * 1.3,
            "chaotic": lambda d: d * random.uniform(0.8, 1.2),
            "inquisitive": lambda d: max(d * 0.6, 0.1),
            "dormant": lambda d: d * 1.0,
            "resonant": lambda d: math.sin(d * math.pi) + 1
        }
        self.policies[other_id] = strategies.get(strategy, lambda d: d)

    def get_interaction_weight(self, other_id):
        key = tuple(sorted((str(self.id), str(other_id))))
        return self.interaction_weights.get(key, 0.5)

    def attempt_reproduction(
        self,
        partner: "Particle",
        config: Any,
        all_particles: List["Particle"]
    ) -> Optional["Particle"]:
        """
        Attempt hybrid reproduction with a partner particle.
        
        Requires:
        - Both particles have energy >= config.energy_threshold_reproduction
        - Rhythm synchronization: |rhythm_diff| < config.rhythm_sync_window
        - Population below cap
        
        On success:
        - Creates offspring with inherited + mutated traits
        - Costs energy to both parents (scaled by their energy levels)
        - Offspring has reduced initial energy
        - Parent-child relationships tracked
        
        Args:
            partner: The other parent particle.
            config: AppConfig instance with reproduction settings.
            all_particles: List of all live particles (for population check).
            
        Returns:
            Offspring Particle if reproduction succeeds, None otherwise.
        """
        # Population cap check
        if len(all_particles) >= config.max_population:
            return None
        
        # Energy threshold check
        if self.energy < config.energy_threshold_reproduction or partner.energy < config.energy_threshold_reproduction:
            return None
        
        # Rhythm synchronization check
        rhythm_diff = abs(self.position[6] - partner.position[6])
        if rhythm_diff > config.rhythm_sync_window:
            return None
        
        # Energy cost scales with parent energy (lower energy = higher cost %)
        cost_multiplier_self = 1.0 + (0.5 * (1.0 - self.energy))  # [1.0, 1.5]
        cost_multiplier_partner = 1.0 + (0.5 * (1.0 - partner.energy))  # [1.0, 1.5]
        
        cost_self = config.reproduction_energy_cost_base * cost_multiplier_self
        cost_partner = config.reproduction_energy_cost_base * cost_multiplier_partner
        
        # Deduct energy from parents
        self.energy -= cost_self
        partner.energy -= cost_partner
        
        # Create offspring
        offspring = Particle(field=self.field)
        
        # Inherit type: from each parent, with configurable mutation chance
        if random.random() > self.field.config.type_mutation_chance:  # Most inherit
            offspring.type = random.choice([self.type, partner.type])
        else:  # Mutation
            all_types = ["cooperative", "avoidant", "chaotic", "inquisitive", "dormant", "resonant"]
            offspring.type = random.choice([t for t in all_types if t not in [self.type, partner.type]])
        
        # Inherit position traits (spatial spawning near midpoint + randomization)
        midpoint = (self.position[0:3] + partner.position[0:3]) / 2
        offspring.position[0:3] = midpoint + np.random.uniform(-0.08, 0.08, 3)
        offspring.position[0:3] = np.clip(offspring.position[0:3], 0.0, 1.0)  # Keep in bounds
        
        # Inherit temporal traits
        offspring.position[3] = dt.datetime.now().timestamp()  # Birth time = now
        offspring.position[4] = (self.position[4] + partner.position[4]) / 2  # Avg current time
        
        # Inherit emotional traits with mutation
        offspring.position[6] = self._inherit_with_mutation(  # rhythm
            self.position[6], partner.position[6], 0.15
        )
        offspring.position[8] = self._inherit_with_mutation(  # valence
            self.position[8], partner.position[8], 0.15
        )
        
        # Clamp emotional traits
        offspring.position[6] = np.clip(offspring.position[6], -1.0, 1.0)
        offspring.position[8] = np.clip(offspring.position[8], -1.0, 1.0)
        
        # Inherit energy/activation with mutation
        offspring.energy = self._inherit_with_mutation(
            self.energy, partner.energy, 0.15
        )
        offspring.energy = min(config.offspring_initial_energy, offspring.energy)
        
        offspring.activation = self._inherit_with_mutation(
            self.activation, partner.activation, 0.15
        )
        offspring.activation = float(max(config.min_activation, min(1.0, offspring.activation)))  # Floor + cap
        
        # Inherit max caps with small mutation (allows evolutionary pressure)
        offspring.max_energy = self._inherit_with_mutation(
            self.max_energy, partner.max_energy, 0.05
        )
        offspring.max_energy = float(max(0.5, min(1.5, offspring.max_energy)))  # Bounded evolution
        
        offspring.max_activation = self._inherit_with_mutation(
            self.max_activation, partner.max_activation, 0.05
        )
        offspring.max_activation = float(max(0.5, min(1.5, offspring.max_activation)))
        
        # Set generation
        max_parent_gen = max(self.generation, partner.generation)
        offspring.generation = max_parent_gen + 1
        
        # Inherit linguistic voice traits (voice_profile)
        # Blend voice characteristics from both parents with small mutation
        for key in offspring.voice_profile:
            blend = random.uniform(0, 1)
            parent_val = blend * self.voice_profile[key] + (1 - blend) * partner.voice_profile[key]
            
            # Small mutation: 0-5% variation (lighter than behavioral traits)
            mutation = random.uniform(-0.05, 0.05)
            offspring.voice_profile[key] = float(max(0.0, min(2.0, parent_val + mutation)))
        
        # Inherit word-context frequencies (vocabulary preferences)
        # Blend vocabulary from both parents
        combined_contexts = set(
            list(self.word_context_frequencies.keys()) + 
            list(partner.word_context_frequencies.keys())
        )
        
        for context in combined_contexts:
            parent_a_words = self.word_context_frequencies.get(context, {})
            parent_b_words = partner.word_context_frequencies.get(context, {})
            
            offspring.word_context_frequencies[context] = {}
            all_words = set(list(parent_a_words.keys()) + list(parent_b_words.keys()))
            
            for word in all_words:
                freq_a = parent_a_words.get(word, 0)
                freq_b = parent_b_words.get(word, 0)
                
                # Inherit 70% of combined frequency
                blend_freq = (freq_a + freq_b) * 0.7
                
                # Configurable chance to keep it, inverse is forgetting (mutation)
                if random.random() > (1 - self.field.config.syntax_pattern_inheritance_chance):
                    offspring.word_context_frequencies[context][word] = int(blend_freq)
        
        # Inherit syntax preferences similarly
        combined_syntax_contexts = set(
            list(self.preferred_syntax_patterns.keys()) +
            list(partner.preferred_syntax_patterns.keys())
        )
        
        for context in combined_syntax_contexts:
            parent_a_syntax = self.preferred_syntax_patterns.get(context, {})
            parent_b_syntax = partner.preferred_syntax_patterns.get(context, {})
            
            offspring.preferred_syntax_patterns[context] = {}
            all_syntax = set(
                list(parent_a_syntax.keys()) + list(parent_b_syntax.keys())
            )
            
            for syntax_type in all_syntax:
                freq_a = parent_a_syntax.get(syntax_type, 0)
                freq_b = parent_b_syntax.get(syntax_type, 0)
                blend_freq = (freq_a + freq_b) * 0.7
                
                if random.random() > 0.1:
                    offspring.preferred_syntax_patterns[context][syntax_type] = int(blend_freq)
        
        # Track genealogy
        offspring.parent_ids = [self.id, partner.id]
        self.child_ids.append(offspring.id)
        partner.child_ids.append(offspring.id)
        
        # Log reproduction event
        self.logger.log({
            "action": "reproduction",
            "parent_a": self.name,
            "parent_b": partner.name,
            "offspring": offspring.name,
            "offspring_type": offspring.type,
            "offspring_generation": offspring.generation,
            "cost_to_parent_a": cost_self,
            "cost_to_parent_b": cost_partner
        })
        
        return offspring

    def _inherit_with_mutation(self, trait_a: float, trait_b: float, mutation_intensity: float) -> float:
        """
        Inherit a trait from both parents with weighted random mutation.
        
        50/50 split from parents, then 0-10% random mutation applied
        (0% mutation most likely, 10% mutation least likely).
        
        Args:
            trait_a: Parent A's trait value.
            trait_b: Parent B's trait value.
            mutation_intensity: Max mutation magnitude (typically 0.15).
            
        Returns:
            Inherited and mutated trait value.
        """
        # 50/50 blend from parents
        blend = random.uniform(0, 1)
        inherited = blend * trait_a + (1 - blend) * trait_b
        
        # 0-10% mutation with weighted distribution (0% more likely)
        mutation_rate = random.betavariate(1.0, 9.0)  # Beta(1, 9) skews toward 0
        mutation_amount = random.uniform(-mutation_intensity, mutation_intensity) * mutation_rate
        
        return inherited + mutation_amount

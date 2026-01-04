"""
Particle field physics engine for the Cognitive Sandbox.
Manages particle interactions, environmental factors, and simulation dynamics.
"""
import json
from main.engine.particles import Particle
from main.memory import MemorySystem
import numpy as np
import uuid
import os
import random
import math
from main.utils.logger import Logger
from collections import deque, Counter
from typing import List, Dict, Optional, Tuple, Any


class ParticleField:
    """
    Manages the physical simulation environment and particle interactions.
    
    Handles neighbor detection, force calculations, environmental rhythms,
    memory persistence, and the main simulation tick loop.
    
    Attributes:
        config (AppConfig): Global configuration instance.
        logger (Logger): Logger for field-level events.
        memories (MemorySystem): SQLite memory management backend.
        rolling_reflection (deque): Rolling window of particle reflections.
    """
    
    def __init__(self, config: Any) -> None:
        """
        Initialize the particle field.
        
        Args:
            config: AppConfig instance with simulation settings.
            
        Raises:
            RuntimeError: If memory system initialization fails.
        """
        self.config = config
        self.logger = Logger().get_logger("ParticleField")
        self.memories = MemorySystem(self, self.config)
        self.rolling_reflection = deque(maxlen=self.config.rolling_reflection_window)

    def from_state(self, data: Dict[str, Any]) -> Particle:
        """
        Reconstruct a particle from saved state dict.
        
        Fully restores all particle attributes including genealogy,
        linguistic characteristics, and memory bank.
        
        Args:
            data: Dictionary containing particle state (from save_state()).
            
        Returns:
            Particle: Reconstructed particle with all saved properties.
            
        Raises:
            KeyError: If required fields are missing.
            ValueError: If particle ID is invalid.
        """
        try:
            p = Particle(
                field=self,
                id=uuid.UUID(data["id"]),
                name=data["name"],
                energy=data.get("energy", 0.5),
                activation=data.get("activation", 0.2)
            )
            
            # Core attributes
            p.type = data.get("type", "cooperative")
            p.position = np.array(data["position"], dtype=np.float32)
            p.velocity = np.array(data["velocity"], dtype=np.float32)
            p.alive = data.get("alive", True)
            p.last_reflection = data.get("last_reflection")
            
            # Per-particle caps (with config defaults for legacy saves)
            p.max_energy = data.get("max_energy", self.config.max_energy)
            p.max_activation = data.get("max_activation", self.config.max_activation)
            
            # Genealogy
            p.generation = data.get("generation", 0)
            parent_ids_str = data.get("parent_ids", [])
            p.parent_ids = [uuid.UUID(pid) for pid in parent_ids_str] if parent_ids_str else []
            child_ids_str = data.get("child_ids", [])
            p.child_ids = [uuid.UUID(cid) for cid in child_ids_str] if child_ids_str else []
            
            # Linguistic attributes (with safe defaults)
            if "voice_profile" in data:
                p.voice_profile = data["voice_profile"]
            if "word_context_frequencies" in data:
                p.word_context_frequencies = data["word_context_frequencies"]
            if "preferred_syntax_patterns" in data:
                p.preferred_syntax_patterns = data["preferred_syntax_patterns"]
            
            # Memory
            p.memory_bank = data.get("memory_bank", [])
            
            return p
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Failed to reconstruct particle from state: {e}")

    def load_latest_session_state(self) -> Optional[Tuple[List[Particle], int]]:
        """
        Load the most recent saved particle state from the database.
        
        Queries the field_states table for the latest session and returns
        the particles at that checkpoint along with the tick count.
        
        Returns:
            Tuple of (particles, tick_count) if found, None otherwise.
            
        Raises:
            ValueError: If state loading fails.
        """
        try:
            import sqlite3
            db_path = os.path.join(self.config.MEM_DIR, "memories.db")
            
            if not os.path.exists(db_path):
                self.logger.log("No saved state database found - starting fresh")
                return None
            
            conn = sqlite3.connect(db_path, timeout=5.0)
            c = conn.cursor()
            
            # Get most recent state from any session
            c.execute('''
                SELECT state_json, tick_count FROM field_states
                ORDER BY tick_count DESC
                LIMIT 1
            ''')
            
            row = c.fetchone()
            conn.close()
            
            if not row:
                self.logger.log("No saved field states found - starting fresh")
                return None
            
            state_json_str, tick_count = row
            state_data = json.loads(state_json_str)
            
            # Reconstruct particles from state
            particles = [self.from_state(p_data) for p_data in state_data]
            
            # Clean up ghost children (IDs that don't exist as particles)
            # This fixes corrupted data from the pre-fix bug
            valid_ids = {p.id for p in particles}
            ghosts_cleaned = 0
            for p in particles:
                original_count = len(p.child_ids)
                p.child_ids = [cid for cid in p.child_ids if cid in valid_ids]
                ghosts_cleaned += original_count - len(p.child_ids)
            
            if ghosts_cleaned > 0:
                self.logger.log({
                    "action": "cleanup_ghost_children",
                    "ghosts_removed": ghosts_cleaned,
                    "message": "Cleaned corrupted child_ids from previous bug"
                })
            
            self.logger.log({
                "action": "load_latest_session_state",
                "particles_loaded": len(particles),
                "tick_count": tick_count
            })
            
            return particles, tick_count
        
        except (sqlite3.DatabaseError, json.JSONDecodeError, ValueError) as e:
            self.logger.warning({
                "action": "load_latest_session_state",
                "error": str(e),
                "message": "Failed to load latest state, will start fresh"
            })
            return None

    def spawn_particles(self, path: Optional[str] = None, count: int = 30) -> List[Particle]:
        """
        Spawn particles, either loading from saved state or creating new.
        
        Args:
            path: Path to saved state file (if None, creates new particles).
            count: Number of particles to create if not loading from state.
            
        Returns:
            List of Particle objects ready for simulation.
            
        Raises:
            ValueError: If particle creation fails.
        """
        try:
            if path and os.path.exists(path):
                return self.memories.load_full_state(path)
            
            particles: List[Particle] = []
            for _ in range(count):
                p = Particle(field=self)
                particles.append(p)
            return particles
        except Exception as e:
            self.logger.error({
                "action": "spawn_particles",
                "error": str(e),
                "count": count,
                "path": path
            })
            raise ValueError(f"Failed to spawn particles: {e}")

    def long_range_force(
        self,
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        force_scale: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate long-range force between two positions.
        
        Uses inverse-square-style decay in 11D space.
        
        Args:
            pos_a: Source position (11D vector).
            pos_b: Target position (11D vector).
            force_scale: Strength multiplier for force calculation.
            
        Returns:
            Force vector in 11D space.
        """
        if force_scale is None:
            force_scale = self.config.long_range_force_scale
            
        dist = np.linalg.norm(np.array(pos_a, dtype=np.float32) - np.array(pos_b, dtype=np.float32))
        
        if dist < 1e-6:
            return np.zeros(11, dtype=np.float32)  # Avoid singularity

        direction = np.array(pos_b, dtype=np.float32) - np.array(pos_a, dtype=np.float32)
        norm_direction = direction / (np.linalg.norm(direction) + 1e-6)

        # Inverse-square-style decay
        magnitude = np.float32(force_scale) / dist
        return norm_direction * magnitude

    def batch_hyper_distance_matrix(
        self,
        positions: np.ndarray,
        weights: Optional[Dict[int, float]] = None
    ) -> np.ndarray:
        """
        Calculate weighted distance matrix between all particles.
        
        Uses dimension-specific weighting for 11D space.
        
        Args:
            positions: Array of shape (N, 11) with particle positions.
            weights: Dict mapping dimension index to weight (default: predefined).
            
        Returns:
            Distance matrix of shape (N, N).
        """
        if weights is None:
            weights = {
                0: 1, 1: 1, 2: 1,           # spatial x, y, z
                3: 0.5, 4: 0.25, 5: 0.25,  # temporal w, t, age
                6: 0.4, 7: 0.6, 8: 0.7,    # emotional rhythm, ?, valence
                9: 0.2, 10: 1.0             # legacy dimensions
            }
        
        w = np.array([weights.get(i, 1.0) for i in range(11)], dtype=np.float32)
        
        diffs = positions[:, None, :] - positions[None, :, :]  # Shape: (N, N, 11)
        dists = np.sqrt(np.sum((diffs * w) ** 2, axis=2))       # Shape: (N, N)
        return dists

    def get_neighbors(
        self,
        particle: Particle,
        max_neighbors: int = 10,
        radius: float = 0.6,
        particles: Optional[List[Particle]] = None,
        matrix: Optional[np.ndarray] = None
    ) -> List[Particle]:
        """
        Find nearby particles (neighbors) for a given particle.
        
        Uses pre-computed distance matrix for efficiency.
        
        Args:
            particle: Reference particle to find neighbors for.
            max_neighbors: Maximum neighbors to return.
            radius: Distance threshold for neighborhood.
            particles: List of all particles.
            matrix: Pre-computed distance matrix.
            
        Returns:
            List of nearby Particle objects.
        """
        if particles is None or matrix is None:
            return []
        
        try:
            idx = particles.index(particle)
            dists = matrix[idx]

            neighbors = [
                particles[i] for i in np.argsort(dists)[1:max_neighbors+1]
                if dists[i] <= radius and particles[i].alive
            ]
            return neighbors
        except (ValueError, IndexError) as e:
            self.logger.warning({
                "action": "get_neighbors",
                "error": str(e),
                "particle": particle.name
            })
            return []

    def compute_environmental_rhythm(self, tick_count: int) -> float:
        """
        Compute environmental rhythm as a sinusoidal oscillation.
        
        Uses configurable base, amplitude, and frequency to create
        a smooth environmental rhythm that particles synchronize with.
        
        Args:
            tick_count: Current simulation tick.
            
        Returns:
            Environmental rhythm value in [-1.0, 1.0].
        """
        return (self.config.env_rhythm_base + 
                self.config.env_rhythm_amplitude * 
                math.sin(tick_count * self.config.env_rhythm_frequency))

    def inspire_particles(self, particles):
        """Inspire random particles with energy boost and spontaneous reflection."""
        alive = [p for p in particles if p.alive]
        if len(alive) < 3:
            return
        for p in random.sample(alive, k=min(3, len(alive))):
            p.activation += random.uniform(0.05, 0.2) * p.energy
            # Generate a procedural reflection using the particle's reflect method
            neighbors = self.get_neighbors(p, 5, 0.5, particles)
            p.reflect(neighbors)
            p.clamp_state()

    def aggregate_reflections(self, particles):
        # pulls all active reflections into a list for diagnostics

        current_reflections = []
        for p in particles:
            for m in p.memory_bank:
                if m.get("reflection"):
                    current_reflections.append(m["reflection"])

        current_set = set(current_reflections)
        self.rolling_reflection.append(current_set)

        # flattening all previous sets in the win
        prior_reflections = set().union(*list(self.rolling_reflection)[:-1])
        new_reflections = current_set - prior_reflections

        # top-n reflections
        top = Counter(current_reflections).most_common(5)
        total_reflections = len(current_reflections)
        unique_count = len(set(current_reflections))
        reuse_ratio = round((top[0][1] / total_reflections), 2) if total_reflections else 0

        return {
            "top_reflections": top,
            "unique_count": unique_count,
            "total_reflections": total_reflections,
            "reuse_ratio": reuse_ratio,
            "new_reflections": len(new_reflections)
        }


    def tick(self, particles, tick_count, env_rhythm):
        # Filter dead particles IN-PLACE to preserve list reference
        # Remove dead particles from original list
        dead_indices = [i for i, p in enumerate(particles) if not p.alive]
        for i in reversed(dead_indices):  # Remove from end to preserve indices
            particles.pop(i)
        
        total_energy = sum(p.energy for p in particles)
        particle_context = {
            "all_particles": particles,
            "total_energy": total_energy,
        }
        
    

        # batch hyper distancing
        positions = np.stack([p.position for p in particles])
        distance_matrix = self.batch_hyper_distance_matrix(positions)
        
        # Pre-compute neighbors map for all particles (used for reproduction and reflection)
        neighbors_map = {
            p: self.get_neighbors(p, 10, 0.6, particles, distance_matrix)
            for p in particles
        }

        for p in particles:
            
            # Nearby detection
            neighbors = self.get_neighbors(p, 10, 0.6, particles, distance_matrix)
            p.adjust_behavior(neighbors, particle_context)

            adaptive_force = np.zeros(11)

            for other in neighbors:
                if p == other or not other.alive:
                    continue

                f = self.long_range_force(
                    p.position,
                    other.position,
                    force_scale=self.config.long_range_force_scale
                )
                adaptive_force += np.clip(f, -0.005, 0.005)

                adaptation = p.adaptive_component(other)
                adaptive_force += np.clip(adaptation * 0.001, -0.01, 0.01)

                p.velocity += adaptive_force
                other.velocity += adaptive_force * 0.1
                p.velocity = np.clip(p.velocity, -0.05, 0.05)
                other.velocity = np.clip(other.velocity, -0.05, 0.05)

            if p.energy > 2.0 or p.activation > 2.0:
                self.logger.log({
                    "tick": tick_count,
                    "warning": "Runaway particle",
                    "particle": p.name,
                    "energy": p.energy,
                    "activation": p.activation
                })
            
            p.update(env_rhythm, neighbors)
            p.clamp_state()
            
        # Check for reproduction opportunities based on config interval
        if tick_count % self.config.reproduction_check_interval == 0:
            new_offspring = []
            
            for p in particles:
                # Check population cap including offspring created this tick
                if len(particles) + len(new_offspring) >= self.config.max_population:
                    break  # Stop reproduction loop if cap reached
                    
                if p.energy < self.config.energy_threshold_reproduction:
                    continue
                
                # Find potential partners: nearby, high energy, synchronized rhythm
                potential_partners = [
                    n for n in neighbors_map.get(p, [])
                    if n.alive and 
                       n.energy >= self.config.energy_threshold_reproduction and
                       abs(p.position[6] - n.position[6]) < self.config.rhythm_sync_window and
                       n.id > p.id  # Prevent duplicate reproduction attempts
                ]
                
                if potential_partners:
                    partner = random.choice(potential_partners)
                    offspring = p.attempt_reproduction(partner, self.config, particles)
                    if offspring:
                        new_offspring.append(offspring)
            
            # Add all new offspring to particle list
            particles.extend(new_offspring)

        if tick_count % self.config.environmental_event_interval == 0 and self.config.simulated_input:
            self.inject_event(particles)

        if tick_count % self.config.rhythm_adjustment_interval == 0:
            self.inspire_particles(particles)
        
        if tick_count % self.config.random_reflection_interval == 0:
            for p in particles:
                if random.random() < self.config.random_reflection_chance:
                    p.reflect(neighbors_map.get(p, []))

        
        # Occasional random reflections with variable timing
        if tick_count > 0 and tick_count % random.randint(100, 400) == 0:
            for p in particles:
                if random.random() < self.config.random_reflection_chance:
                    p.random_reflection()

        if tick_count % self.config.memory_save_interval == 0:
            self.memories.save_memory_state(particles)

        if tick_count % self.config.full_state_backup_interval == 0:
            self.memories.backup_full_state(particles, tick_count)


    # user interaction

    def inject_event(self, particles):
        alive = [p for p in particles if p.alive]

        if alive:
            target = random.choice(alive)
            target.energy += 0.3
            target.activation += 0.2
            target.last_reflection = f"{target.generate_emotive_string()}"


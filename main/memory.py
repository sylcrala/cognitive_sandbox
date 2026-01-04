"""
Memory management system for the Cognitive Sandbox.
Handles persistent storage of particle memories and state snapshots.
"""
import os
import json
import datetime as dt
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from main.utils.logger import Logger

class MemorySystem:
    """
    SQLite-based memory system for efficient storage and querying.
    Replaces JSON system with transactional, indexed database backend.
    """
    
    def __init__(self, particle_field: Optional[Any] = None, config: Optional[Any] = None):
        """
        Initialize SQLite memory system.
        
        Args:
            particle_field: Reference to the ParticleField instance.
            config: AppConfig instance for paths and settings.
            
        Raises:
            RuntimeError: If database initialization fails.
        """
        self.particle_field = particle_field
        self.logger = Logger().get_logger("MemorySystem")
        self.config = config
        self.MEMORY_DIR = self.config.MEM_DIR
        self.DB_PATH = os.path.join(self.MEMORY_DIR, "memories.db")
        self.MAX_MEMORIES_PER_PARTICLE = 200  # Prevent unbounded growth
        
        try:
            self._initialize_database()
        except Exception as e:
            self.logger.error({
                "action": "database_init",
                "error": str(e)
            })
            raise RuntimeError(f"Failed to initialize memory database: {e}")

    def _initialize_database(self) -> None:
        """
        Create database tables if they don't exist.
        Sets up indexes for efficient querying.
        
        Raises:
            sqlite3.DatabaseError: If database operations fail.
        """
        os.makedirs(self.MEMORY_DIR, exist_ok=True)
        
        try:
            conn = sqlite3.connect(self.DB_PATH, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for concurrent access
            c = conn.cursor()
            
            # Particle memories table - enhanced with linguistic metadata
            c.execute('''
                CREATE TABLE IF NOT EXISTS particle_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    particle_id TEXT NOT NULL,
                    particle_name TEXT NOT NULL,
                    memory_json TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    reflection TEXT,
                    emotional_context TEXT,
                    words_used TEXT,
                    syntax_type TEXT,
                    particle_generation INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(particle_id, timestamp)
                )
            ''')
            
            # Particle state snapshots
            c.execute('''
                CREATE TABLE IF NOT EXISTS particles (
                    particle_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    energy REAL,
                    activation REAL,
                    generation INTEGER DEFAULT 0,
                    parent_ids TEXT,
                    child_ids TEXT,
                    position TEXT,
                    velocity TEXT,
                    last_reflection TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Session state backups
            c.execute('''
                CREATE TABLE IF NOT EXISTS field_states (
                    session_id TEXT NOT NULL,
                    tick_count INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, tick_count)
                )
            ''')
            
            # Create indexes for efficient querying
            c.execute('CREATE INDEX IF NOT EXISTS idx_particle_id ON particle_memories(particle_id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON particle_memories(timestamp)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_reflection ON particle_memories(reflection)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_session ON field_states(session_id)')
            
            conn.commit()
            conn.close()
        except sqlite3.DatabaseError as e:
            self.logger.error({
                "action": "database_init",
                "error": str(e),
                "db_path": self.DB_PATH
            })
            raise

    def save_particle_memory(self, particle_id: str, particle_name: str, memory: Dict[str, Any]) -> None:
        """
        Save a single memory entry for a particle with linguistic metadata.
        Enforces maximum memory limit per particle.
        
        Args:
            particle_id: Unique particle identifier.
            particle_name: Human-readable particle name.
            memory: Memory dict to store (may include linguistic metadata).
            
        Raises:
            sqlite3.DatabaseError: If database write fails.
        """
        try:
            conn = sqlite3.connect(self.DB_PATH, timeout=5.0)
            c = conn.cursor()
            
            memory_json = json.dumps(memory)
            timestamp = memory.get("timestamp", dt.datetime.now().timestamp())
            reflection = memory.get("reflection", None)
            emotional_context = memory.get("context", None)
            words_used = json.dumps(memory.get("words_used", []))  # JSON array
            syntax_type = memory.get("syntax_type", None)
            particle_generation = memory.get("generation", 0)
            
            c.execute('''
                INSERT OR IGNORE INTO particle_memories 
                (particle_id, particle_name, memory_json, timestamp, reflection, 
                 emotional_context, words_used, syntax_type, particle_generation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (particle_id, particle_name, memory_json, timestamp, reflection,
                  emotional_context, words_used, syntax_type, particle_generation))
            
            # Remove oldest memories if exceeded limit
            c.execute('''
                DELETE FROM particle_memories
                WHERE particle_id = ? AND id NOT IN (
                    SELECT id FROM particle_memories
                    WHERE particle_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
            ''', (particle_id, particle_id, self.MAX_MEMORIES_PER_PARTICLE))
            
            conn.commit()
            conn.close()
        except sqlite3.DatabaseError as e:
            self.logger.error({
                "action": "save_particle_memory",
                "error": str(e),
                "particle_id": particle_id
            })
            raise

    def load_particle_memories(self, particle_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load memories for a specific particle.
        
        Args:
            particle_id: Unique particle identifier.
            limit: Maximum number of memories to retrieve (defaults to MAX_MEMORIES_PER_PARTICLE).
            
        Returns:
            List of memory dicts for the particle.
            
        Raises:
            sqlite3.DatabaseError: If database read fails.
        """
        limit = limit or self.MAX_MEMORIES_PER_PARTICLE
        
        try:
            conn = sqlite3.connect(self.DB_PATH, timeout=5.0)
            c = conn.cursor()
            
            c.execute('''
                SELECT memory_json FROM particle_memories
                WHERE particle_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (particle_id, limit))
            
            rows = c.fetchall()
            conn.close()
            
            return [json.loads(row[0]) for row in rows]
        except (sqlite3.DatabaseError, json.JSONDecodeError) as e:
            self.logger.warning({
                "action": "load_particle_memories",
                "error": str(e),
                "particle_id": particle_id
            })
            return []

    def save_memory_state(self, particles: List[Any], base_dir: Optional[str] = None) -> None:
        """
        Save all unpersisted particle memories to the database.
        
        Args:
            particles: List of Particle objects.
            base_dir: Unused (kept for API compatibility with JSON system).
        """
        try:
            for p in particles:
                unpersisted = [m for m in p.memory_bank if not m.get("persisted", False)]
                
                for memory in unpersisted:
                    self.save_particle_memory(str(p.id), p.name, memory)
                    memory["persisted"] = True
        except Exception as e:
            self.logger.error({
                "action": "save_memory_state",
                "error": str(e)
            })

    def load_memory_state(self, particles: List[Any], base_dir: Optional[str] = None) -> None:
        """
        Load persisted memories from database into particles.
        
        Args:
            particles: List of Particle objects to load memories into.
            base_dir: Unused (kept for API compatibility with JSON system).
        """
        try:
            for p in particles:
                memories = self.load_particle_memories(str(p.id))
                p.memory_bank.extend(memories)
        except Exception as e:
            self.logger.warning({
                "action": "load_memory_state",
                "error": str(e)
            })

    def backup_full_state(self, particles: List[Any], tick_count: int, dir_path: str = "./backups/") -> None:
        """
        Store full field state in database.
        
        Args:
            particles: List of Particle objects.
            tick_count: Current simulation tick.
            dir_path: Unused (kept for API compatibility).
            
        Raises:
            sqlite3.DatabaseError: If database write fails.
        """
        try:
            state = [p.save_state() for p in particles]
            state_json = json.dumps(state)
            session_id = self.config.session_id
            timestamp = dt.datetime.now().timestamp()
            
            conn = sqlite3.connect(self.DB_PATH, timeout=5.0)
            c = conn.cursor()
            
            c.execute('''
                INSERT OR REPLACE INTO field_states
                (session_id, tick_count, state_json, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (session_id, tick_count, state_json, timestamp))
            
            conn.commit()
            conn.close()
        except (sqlite3.DatabaseError, IOError) as e:
            self.logger.error({
                "action": "backup_full_state",
                "error": str(e),
                "tick_count": tick_count
            })
            raise

    def load_full_state(self, path: str) -> List[Any]:
        """
        Load particle states from database backup (legacy compatibility).
        
        Args:
            path: Session ID to look up in database.
            
        Returns:
            List of Particle objects.
            
        Raises:
            ValueError: If state cannot be loaded.
        """
        try:
            conn = sqlite3.connect(self.DB_PATH, timeout=5.0)
            c = conn.cursor()
            
            # Try to load most recent state for this session
            c.execute('''
                SELECT state_json FROM field_states
                WHERE session_id = ?
                ORDER BY tick_count DESC
                LIMIT 1
            ''', (path,))
            
            row = c.fetchone()
            conn.close()
            
            if not row:
                raise ValueError(f"No state found for session {path}")
            
            data = json.loads(row[0])
            particles = [self.particle_field.from_state(p) for p in data]
            return particles
        except (sqlite3.DatabaseError, json.JSONDecodeError, ValueError) as e:
            self.logger.error({
                "action": "load_full_state",
                "error": str(e),
                "path": path
            })
            raise
"""
Entry point for the Cognitive Sandbox simulation.
Handles CLI argument parsing and orchestrates the main simulation loop.
"""
import argparse
import datetime as dt
import numpy as np
from typing import List, Optional
from main.utils.logger import Logger
from main.config import get_config
from main.visualizer import TextVisualizer
from main.engine.field import ParticleField
from main.engine.particles import Particle


class CognitiveSandbox:
    """
    Main orchestrator for the Cognitive Sandbox simulation environment.
    
    Manages particle spawning, memory loading, environment initialization,
    and delegates to the visualizer for the main simulation loop.
    
    Attributes:
        logger (Logger): Logger instance for this class.
        config (AppConfig): Global application configuration.
        particle_count (int): Number of particles to simulate.
        active_panel (str): Which panel to display on the right side (or None for clean mode).
        delay (float): Delay in seconds between simulation ticks.
        max_ticks (Optional[int]): Maximum number of ticks to run (None = infinite).
        norestore (bool): Whether to skip loading previous session state.
        particle_field (ParticleField): The particle physics engine.
        visualizer (TextVisualizer): The CLI visualization engine.
    """

    def __init__(
        self,
        particle_count: int = 20,
        delay: float = 0.1,
        max_ticks: Optional[int] = None,
        norestore: bool = False,
        active_panel: Optional[str] = None
    ) -> None:
        """
        Initialize the Cognitive Sandbox.
        
        Args:
            particle_count: Number of particles to spawn (default: 20).
            delay: Seconds to sleep between ticks (default: 0.1).
            max_ticks: Maximum simulation ticks (None = run indefinitely).
            norestore: Skip loading saved state from previous session (default: False).
            active_panel: Which panel to display (None = clean mode).
            
        Raises:
            RuntimeError: If configuration or initialization fails.
        """
        try:
            self.logger = Logger().get_logger("CognitiveSandbox")
            self.config = get_config()
            self.particle_count = particle_count
            self.delay = delay
            self.max_ticks = max_ticks
            self.norestore = norestore
            self.active_panel = active_panel
            
            self.particle_field = ParticleField(config=self.config)
            self.visualizer = TextVisualizer(
                config=self.config, 
                particle_field=self.particle_field,
                active_panel=self.active_panel
            )
            
            self.logger.log({
                "status": "CognitiveSandbox initialized",
                "particles": particle_count,
                "tick_delay": delay,
                "active_panel": active_panel or "clean mode"
            })
        except Exception as e:
            self.logger.error({
                "action": "initialization",
                "error": str(e)
            })
            raise RuntimeError(f"Failed to initialize CognitiveSandbox: {e}")

    def launch_sandbox(self) -> None:
        """
        Launch the cognitive sandbox simulation.
        
        Loads most recent saved state if available and not disabled,
        otherwise spawns new particles. Loads memories and initializes environment.
        
        Raises:
            RuntimeError: If sandbox startup fails.
        """
        try:
            self.logger.log("Cognitive Sandbox Initializing...")

            self.logger.log({
                "status": "System startup initializing...",
                "timestamp": dt.datetime.now().timestamp()
            })

            # Try to load most recent state from database (unless norestore is set)
            particles: Optional[List[Particle]] = None
            tick_count: int = 0
            
            if not self.norestore:
                result = self.particle_field.load_latest_session_state()
                if result:
                    particles, tick_count = result
                    self.logger.log({
                        "status": "Loaded saved state from database",
                        "particles": len(particles),
                        "tick_count": tick_count
                    })
            
            # If no saved state, spawn new particles
            if particles is None:
                particles = self.particle_field.spawn_particles(
                    path=None,
                    count=self.particle_count
                )
                tick_count = 0
                
                if not particles:
                    raise ValueError("Failed to spawn particles")
            
            camera_offset: np.ndarray = np.zeros(2)  # For X/Y positioning
            env_rhythm: float = self.particle_field.compute_environmental_rhythm(tick_count)

            # Load persisted memories from previous sessions
            self.particle_field.memories.load_memory_state(particles)

            self.logger.log({
                "status": "System startup successful.",
                "timestamp": dt.datetime.now().timestamp(),
                "particle_count": len(particles),
                "tick_count": tick_count,
                "env_rhythm": env_rhythm
            })

            # Start the main simulation loop
            self.visualizer.startup(
                particles=particles,
                tick_count=tick_count,
                camera_offset=camera_offset,
                env_rhythm=env_rhythm,
                delay=self.delay
            )
        except Exception as e:
            self.logger.error({
                "action": "launch_sandbox",
                "error": str(e),
                "timestamp": dt.datetime.now().timestamp()
            })
            raise RuntimeError(f"Sandbox launch failed: {e}")


def main() -> None:
    """
    Main entry point for the Cognitive Sandbox CLI application.
    
    Parses command-line arguments and launches the sandbox with specified parameters.
    """
    parser = argparse.ArgumentParser(
        description="Launch the Cognitive Sandbox simulation environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py                      # Clean mode (field only)
  python start.py --panel stats        # Show statistics panel
  python start.py --panel genealogy    # Show genealogy panel
  python start.py --particles 50 --panel legend
  python start.py --norestore --panel inspector

Available Panels:
  legend      - Particle type color legend
  stats       - Live population statistics
  diagnostics - Environment sync diagnostics
  reflections - Reflection analysis
  genealogy   - Generational & reproduction stats
  linguistics - Linguistic patterns & diversity
  inspector   - Most active particle inspector
        """
    )
    
    parser.add_argument(
        "--particles",
        type=int,
        default=30,
        help="Number of initial particles to spawn (default: 30)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between simulation ticks in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Maximum number of ticks to run (default: run indefinitely)"
    )
    parser.add_argument(
        "--norestore",
        action="store_true",
        help="Skip loading previous session state; start fresh"
    )
    parser.add_argument(
        "--panel",
        type=str,
        choices=["legend", "stats", "diagnostics", "reflections", "genealogy", "linguistics", "inspector"],
        default=None,
        help="Display a panel on the right side (default: clean mode, no panel)"
    )

    args = parser.parse_args()

    try:
        sandbox = CognitiveSandbox(
            particle_count=args.particles,
            delay=args.delay,
            max_ticks=args.ticks,
            norestore=args.norestore,
            active_panel=args.panel
        )
        sandbox.launch_sandbox()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        import traceback
        print(f"Fatal error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
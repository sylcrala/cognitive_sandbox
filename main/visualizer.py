from rich.console import Console, Group
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
import time
import math
import numpy as np
import sys
import os
import select
import termios
import tty
from main.utils.logger import Logger


class TextVisualizer(Console):
    def __init__(self, particle_field, config, active_panel: str = None):
        super().__init__()
        self.logger = Logger().get_logger("Visualizer")
        self.config = config
        self.particle_field = particle_field

        self.ERROR_FLAG = self.config.ERROR_flag
        self.console = self
        
        # Active panel - only one panel can be visible at a time (or None for clean mode)
        # Options: "legend", "stats", "diagnostics", "reflections", "genealogy", "linguistics", "inspector"
        self.active_panel = active_panel
        self.running = True
        
        # Panel button positions (column ranges) - calculated based on nav bar layout
        # Format: {"panel_name": (start_col, end_col)}
        self.panel_buttons = {}
        
        # Cache for particle stats (updated each tick)
        self._cached_stats = {"count": 0, "avg_energy": 0.0, "max_energy": 0.0}
        
        # Track seen reflections for "new" detection (with size limit)
        self._seen_reflections = set()
        self._new_reflection_count = 0
        self._max_seen_reflections = 1000  # Prevent unbounded growth
        
        # Mouse input handling
        self._mouse_enabled = False
        self._old_term_settings = None
        self._input_buffer = ""

    def render_navigation_bar(self, tick_count, env_rhythm) -> Panel:
        """
        Render navigation bar with full-width clickable buttons.
        Buttons span the entire terminal width proportionally.
        """
        panels = ["legend", "stats", "diagnostics", "reflections", "genealogy", "linguistics", "inspector"]
        
        # Calculate button widths based on terminal width
        # Account for panel borders (2) and padding (2)
        usable_width = max(80, self.console.size.width - 4)
        button_width = usable_width // len(panels)
        
        # Build button positions for mouse detection
        self.panel_buttons = {}
        items = []
        
        for i, panel_name in enumerate(panels):
            label = panel_name.capitalize()
            
            # Calculate column range for this button (1-indexed for terminal)
            start_col = 2 + (i * button_width)  # +2 for left border
            end_col = start_col + button_width - 1
            self.panel_buttons[panel_name] = (start_col, end_col)
            
            # Center the label within the button width
            padding = button_width - len(label) - 2  # -2 for icon space
            left_pad = padding // 2
            right_pad = padding - left_pad
            
            if self.active_panel == panel_name:
                items.append(f"[bold white on dark_green]{' ' * left_pad}◆ {label}{' ' * right_pad}[/bold white on dark_green]")
            else:
                items.append(f"[cyan on grey23]{' ' * left_pad}  {label}{' ' * right_pad}[/cyan on grey23]")
        
        nav_text = "".join(items)
        
        # Stats line with particle count and average energy
        count = self._cached_stats.get("count", 0)
        avg_energy = self._cached_stats.get("avg_energy", 0.0)
        max_energy = self._cached_stats.get("max_energy", 0.0)
        stats_line = f"  [bold]Pop:[/bold] {count}  [bold]Energy:[/bold] {avg_energy:.2f} avg / {max_energy:.2f} max  [dim]│ tick {tick_count} │ rhythm {env_rhythm:.2f}[/dim]"
        
        return Panel(
            f"{nav_text}\n{stats_line}",
            title="[bold cyan]Cognitive Sandbox[/bold cyan]",
            subtitle="[dim]Click button to toggle panel[/dim]",
            style="blue",
            height=4
        )


    def get_dynamic_grid_size(self):
        # Subtract panel width only if a panel is active
        panel_width = 57 if self.active_panel else 4  # 55 panel + 2 border, or just borders
        width = max(20, self.console.size.width - panel_width)     
        height = max(20, self.console.size.height - 10)
        return width, height

    def apply_brightness(self, color, brightness, min_brightness=0.3):
        if not isinstance(brightness, (int, float)) or math.isnan(brightness) or not math.isfinite(brightness):
            brightness = 0.3  # safe fallback
        brightness = max(min(brightness, 1.5), 0.1)
        return tuple(int(c * (brightness * (1 - min_brightness) + min_brightness)) for c in color)


    def get_particle_char(self, activation):
        if activation > 0.8:
            return "⬤"
        elif activation > 0.5:
            return "●"
        elif activation > 0.2:
            return "·"
        else:
            return "˙"

    def compute_camera_target(self, particles, prev_offset, mode="mass", smoothing = 0.1):
        if not particles:
            return np.zeros(2)

        if mode == "mass":
            avg_x = sum(p.position[0] for p in particles) / len(particles)
            avg_y = sum(p.position[1] for p in particles) / len(particles)
            target = np.array([avg_x, avg_y])
            return prev_offset + (target - prev_offset) * smoothing
        
        elif mode == "time":
            avg_w = sum(p.position[3] for p in particles) / len(particles)
            # Map w to a spatial anchor (just X here for demo)
            target =  np.array([avg_w % 1.0, 0.5])  # center Y
            return prev_offset + (target - prev_offset) * smoothing


    def build_legend_panel(self):
        """Comprehensive rendering guide showing all visual indicators."""
        
        # Type colors section
        type_colors = {
            "cooperative": (0, 255, 0),
            "avoidant": (255, 165, 0),
            "chaotic": (255, 0, 0),
            "inquisitive": (255, 255, 0),
            "dormant": (0, 0, 255),
            "resonant": (128, 0, 128),
        }
        
        type_table = Table(title="Particle Types", expand=True, show_header=False)
        type_table.add_column("Type", width=12)
        type_table.add_column("Color", width=8)
        type_table.add_column("Behavior", width=25)
        
        type_descriptions = {
            "cooperative": "Seeks proximity",
            "avoidant": "Maintains distance",
            "chaotic": "Random movement",
            "inquisitive": "Explores actively",
            "dormant": "Low activity",
            "resonant": "Rhythm-synced",
        }
        
        for name, rgb in type_colors.items():
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
            symbol = self.get_particle_char(0.9)
            type_table.add_row(
                name.capitalize(),
                f"[{hex_color}]{symbol}[/{hex_color}]",
                f"[dim]{type_descriptions[name]}[/dim]"
            )
        
        # Size/Shape indicators
        size_table = Table(title="Size = Vitality", expand=True, show_header=False)
        size_table.add_column("Symbol", width=6, justify="center")
        size_table.add_column("Vitality", width=12)
        size_table.add_column("Meaning", width=20)
        
        size_table.add_row("⬤", "> 0.8", "Very high vitality")
        size_table.add_row("●", "0.5 - 0.8", "High vitality")
        size_table.add_row("·", "0.2 - 0.5", "Moderate vitality")
        size_table.add_row("˙", "< 0.2", "Low vitality")
        
        # Brightness indicators
        bright_table = Table(title="Brightness = Energy", expand=True, show_header=False)
        bright_table.add_column("Example", width=6, justify="center")
        bright_table.add_column("Energy", width=12)
        bright_table.add_column("State")
        
        # Show brightness samples
        base = (0, 255, 0)  # green
        for energy, label in [(1.0, "Full"), (0.5, "Half"), (0.2, "Low")]:
            bright = self.apply_brightness(base, energy)
            hex_c = "#{:02x}{:02x}{:02x}".format(*bright)
            bright_table.add_row(f"[{hex_c}]●[/{hex_c}]", f"{energy:.1f}", label)
        
        # Vitality formula
        formula = Table.grid(padding=1)
        formula.add_column(style="bold dim")
        formula.add_column()
        formula.add_row("─── Formulas ───", "")
        formula.add_row("Vitality", "energy × (1 + rhythm_sync)")
        formula.add_row("Brightness", "energy / 2 (clamped)")
        formula.add_row("Rhythm Sync", "|particle_rhythm - env| < 0.2")
        
        return Panel(Group(type_table, size_table, bright_table, formula), title="Rendering Guide")


    def render_particles_grid(self, particles, camera_offset, env_rhythm, tick_count):
        # main visualizer render method

        # pulling dynamic size
        width, height = self.get_dynamic_grid_size()

        # initializing empty grid
        grid = [[" "] * width for _ in range(height)]

        # map types to RGB tuples
        type_colors = {
            "cooperative": (0, 255, 0),
            "avoidant": (255, 165, 0),
            "chaotic": (255, 0, 0),
            "inquisitive": (255, 255, 0),
            "dormant": (0, 0, 255),
            "resonant": (128, 0, 128),
        }

        lock_center = self.compute_camera_target(particles, camera_offset, mode = "mass")

        for p in particles:
            # mapping normalized particle positions to grid coords
            x = int((p.position[0] - lock_center[0]) * width + width // 2) % width
            y = int((p.position[1] - lock_center[1]) * height + height // 2) % height

            # setting particle symbol
            vitality = p.vitality_score(env_rhythm)
            symbol = self.get_particle_char(vitality / 2)

            # setting particle color
            base_color = type_colors.get(p.type, (255, 255, 255))                           # default color
            brightness = min(max((p.energy / 2), 0.1), 1.0)                               # brightness based on energy
            
            

            if not math.isfinite(brightness) and p.id not in self.ERROR_FLAG:
                self.logger.log({
                    "tick": tick_count,
                    "error": "Invalid brightness",
                    "energy": p.energy,
                    "particle_id": str(p.id),
                    "position": p.position.tolist()
                })
                brightness = 0.3


            # adjust brightness
            bright_color = self.apply_brightness(base_color, brightness)
            hex_color = "#{:02x}{:02x}{:02x}".format(*bright_color)

            dot = self.get_particle_char(p.activation)

            grid[y][x] = f"[{hex_color}]{dot}[/{hex_color}]"



        # join each row into string
        lines = ["".join(row) for row in grid]
        # join rows with newline and return as string
        return "\n".join(lines)

    def render_stats(self, particles, tick_count, env_rhythm):
        """Expanded live statistics with type breakdown and trends."""
        if not particles:
            return Panel("No particles", title="Statistics")
        
        total = len(particles)
        alive = sum(p.alive for p in particles)
        total_energy = sum(p.energy for p in particles)
        avg_energy = total_energy / total
        max_energy = max(p.energy for p in particles)
        min_energy = min(p.energy for p in particles)
        total_activation = sum(p.activation for p in particles)
        avg_activation = total_activation / total
        max_activation = max(p.activation for p in particles)
        
        # Genealogy metrics
        offspring_count = sum(1 for p in particles if p.generation > 0)
        parents_count = sum(1 for p in particles if len(p.child_ids) > 0)
        max_gen = max(p.generation for p in particles)
        
        # Summary stats
        summary = Table.grid(padding=1)
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Tick", str(tick_count))
        summary.add_row("Population", f"{alive}/{total} alive")
        summary.add_row("Generations", f"Gen 0-{max_gen} | {parents_count} parents | {offspring_count} offspring")
        summary.add_row("Env Rhythm", f"{env_rhythm:.3f}")
        
        # Energy/Activation table
        vitals = Table(title="Vitals", expand=True)
        vitals.add_column("Metric")
        vitals.add_column("Avg", justify="right")
        vitals.add_column("Max", justify="right")
        vitals.add_column("Min", justify="right")
        vitals.add_row("Energy", f"{avg_energy:.2f}", f"{max_energy:.2f}", f"{min_energy:.2f}")
        vitals.add_row("Activation", f"{avg_activation:.2f}", f"{max_activation:.2f}", f"{min(p.activation for p in particles):.2f}")
        
        # Type distribution
        type_count = {}
        type_energy = {}
        for p in particles:
            type_count[p.type] = type_count.get(p.type, 0) + 1
            type_energy[p.type] = type_energy.get(p.type, 0) + p.energy
        
        type_table = Table(title="Type Distribution", expand=True)
        type_table.add_column("Type")
        type_table.add_column("Count", justify="right")
        type_table.add_column("Avg E", justify="right")
        type_table.add_column("Share", justify="right")
        
        type_colors = {"cooperative": "green", "avoidant": "yellow", "chaotic": "red",
                      "inquisitive": "cyan", "dormant": "blue", "resonant": "magenta"}
        
        for ptype, count in sorted(type_count.items(), key=lambda x: x[1], reverse=True):
            color = type_colors.get(ptype, "white")
            avg_e = type_energy[ptype] / count
            share = (count / total) * 100
            type_table.add_row(f"[{color}]{ptype}[/{color}]", str(count), f"{avg_e:.2f}", f"{share:.0f}%")
        
        return Panel(Group(summary, vitals, type_table), title="Live Statistics")

    def render_reflection_analysis(self, particles):
        """Expanded reflection analysis with proper new reflection tracking."""
        data = self.particle_field.aggregate_reflections(particles)
        
        # Track new reflections ourselves for more persistent display
        current_reflections = set()
        for p in particles:
            for m in p.memory_bank:
                if m.get("reflection"):
                    current_reflections.add(m["reflection"])
        
        # Find truly new reflections (never seen before in this session)
        new_this_session = current_reflections - self._seen_reflections
        self._seen_reflections.update(current_reflections)
        new_this_tick = len(new_this_session)
        if new_this_session:
            self._new_reflection_count += len(new_this_session)
        
        # Note: We don't trim _seen_reflections anymore to avoid re-counting
        # The set is bounded by total unique reflections which is naturally limited
        # by memory_bank sizes (20 per particle × max_population)
        
        # Summary metrics
        summary = Table.grid(padding=1)
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Total Reflections", str(data["total_reflections"]))
        summary.add_row("Unique Phrases", str(data["unique_count"]))
        summary.add_row("New (session)", f"[green]{self._new_reflection_count}[/green]" if self._new_reflection_count < 10000 else f"[yellow]{self._new_reflection_count:,}[/yellow]")
        summary.add_row("Reuse Ratio", f"{data['reuse_ratio']:.0%}")
        
        # Top reflections table
        top_table = Table(title="Most Common Reflections", expand=True, show_header=True)
        top_table.add_column("#", width=3, justify="right")
        top_table.add_column("Reflection")
        top_table.add_column("×", width=4, justify="right")
        
        for i, (phrase, count) in enumerate(data["top_reflections"][:5], 1):
            display = phrase[:55] + "..." if len(phrase) > 55 else phrase
            top_table.add_row(str(i), display, str(count))
        
        # Recent reflections per particle
        type_colors = {"cooperative": "green", "avoidant": "yellow", "chaotic": "red", 
                      "inquisitive": "cyan", "dormant": "blue", "resonant": "magenta"}
        
        recent_table = Table(title="Latest by Particle", expand=True, show_header=True)
        recent_table.add_column("Particle", width=12)
        recent_table.add_column("Last Reflection")
        
        active_reflectors = [(p.name, p.last_reflection, p.type) for p in particles if p.last_reflection]
        for name, reflection, ptype in active_reflectors[:6]:
            display = reflection[:45] + "..." if len(reflection) > 45 else reflection
            color = type_colors.get(ptype, "white")
            recent_table.add_row(f"[{color}]{name}[/{color}]", display)
        
        return Panel(Group(summary, top_table, recent_table), title="Reflection Analysis")

    def render_inspector(self, particles):
        # particle inspector render method
        # expand this later to include particle selection

        most_active = max(particles, key=lambda p: p.activation)
        
        inspector = Table(title = f"{most_active.name} (Most Active)", expand = True)
        inspector.add_column("Field")
        inspector.add_column("Value")
        inspector.add_row("Energy", f"{most_active.energy:.2f}")
        inspector.add_row("Activation", f"{most_active.activation:.2f}")
        inspector.add_row("Valence", f"{most_active.position[8]:.2f}")
        inspector.add_row("Last Reflection", f"{most_active.last_reflection}")

        return Panel(inspector)

    def render_environment_diagnostics(self, particles, env_rhythm):
        """Expanded environment diagnostics with comprehensive metrics."""
        if not particles:
            return Panel("No particles", title="Diagnostics")
        
        total = len(particles)
        
        # Rhythm metrics
        rhythms = [p.position[6] for p in particles]
        in_sync = sum(abs(r - env_rhythm) < 0.2 for r in rhythms)
        out_sync = total - in_sync
        avg_rhythm = sum(rhythms) / total
        rhythm_variance = sum((r - avg_rhythm) ** 2 for r in rhythms) / total
        rhythm_std = rhythm_variance ** 0.5
        
        # Valence distribution
        valences = [p.position[8] for p in particles]
        positive = sum(1 for v in valences if v > 0.3)
        negative = sum(1 for v in valences if v < -0.3)
        neutral = total - positive - negative
        avg_valence = sum(valences) / total
        
        # Energy distribution
        energies = [p.energy for p in particles]
        low_energy = sum(1 for e in energies if e < 0.3)
        high_energy = sum(1 for e in energies if e > 0.7)
        
        # Activation metrics
        activations = [p.activation for p in particles]
        dormant = sum(1 for a in activations if a < 0.1)
        highly_active = sum(1 for a in activations if a > 0.5)
        
        # Spatial metrics
        positions_x = [p.position[0] for p in particles]
        positions_y = [p.position[1] for p in particles]
        spatial_spread_x = max(positions_x) - min(positions_x)
        spatial_spread_y = max(positions_y) - min(positions_y)
        
        # Environment summary
        env_summary = Table.grid(padding=1)
        env_summary.add_column(style="bold")
        env_summary.add_column()
        env_summary.add_row("Env Rhythm", f"{env_rhythm:.3f}")
        env_summary.add_row("Avg Particle Rhythm", f"{avg_rhythm:.3f}")
        env_summary.add_row("Rhythm Std Dev", f"{rhythm_std:.3f}")
        env_summary.add_row("Avg Valence", f"{avg_valence:.3f}")
        
        # Sync visualization
        sync = Table(title="Rhythm Sync", expand=True, show_header=True)
        sync.add_column("Status", width=12)
        sync.add_column("#", width=4, justify="right")
        sync.add_column("%", width=5, justify="right")
        sync.add_column("Visual")
        
        sync_pct = (in_sync / total) * 100
        sync_bar = "█" * int(sync_pct / 8)
        sync.add_row("[green]In Sync[/green]", str(in_sync), f"{sync_pct:.0f}%", f"[green]{sync_bar}[/green]")
        
        out_pct = (out_sync / total) * 100
        out_bar = "█" * int(out_pct / 8)
        sync.add_row("[red]Desync[/red]", str(out_sync), f"{out_pct:.0f}%", f"[red]{out_bar}[/red]")
        
        # Valence table
        valence = Table(title="Emotional State", expand=True, show_header=True)
        valence.add_column("Valence", width=12)
        valence.add_column("#", width=4, justify="right")
        valence.add_column("%", width=5, justify="right")
        
        valence.add_row("[green]Positive[/green]", str(positive), f"{(positive/total)*100:.0f}%")
        valence.add_row("[dim]Neutral[/dim]", str(neutral), f"{(neutral/total)*100:.0f}%")
        valence.add_row("[red]Negative[/red]", str(negative), f"{(negative/total)*100:.0f}%")
        
        # Activity metrics
        activity = Table(title="Activity & Energy", expand=True, show_header=True)
        activity.add_column("Metric", width=14)
        activity.add_column("Count", width=5, justify="right")
        activity.add_column("Share", width=6, justify="right")
        
        activity.add_row("Dormant (A<0.1)", str(dormant), f"{(dormant/total)*100:.0f}%")
        activity.add_row("Highly Active", str(highly_active), f"{(highly_active/total)*100:.0f}%")
        activity.add_row("Low Energy (<0.3)", str(low_energy), f"{(low_energy/total)*100:.0f}%")
        activity.add_row("High Energy (>0.7)", str(high_energy), f"{(high_energy/total)*100:.0f}%")
        
        # Spatial info
        spatial = Table.grid(padding=1)
        spatial.add_column(style="bold dim")
        spatial.add_column()
        spatial.add_row("─── Spatial ───", "")
        spatial.add_row("X Spread", f"{spatial_spread_x:.3f}")
        spatial.add_row("Y Spread", f"{spatial_spread_y:.3f}")
        
        return Panel(Group(env_summary, sync, valence, activity, spatial), title="Environment Diagnostics")



    def render_panel(self, particles, tick_count, camera_offset, env_rhythm):
        """
        Render the main display with navigation bar at top, field in center,
        and optional single panel on the right side.
        """
        # Render the particle field
        grid_str = self.render_particles_grid(particles, camera_offset, env_rhythm, tick_count)
        field_panel = Panel(
            grid_str, 
            title="Environment Visualizer", 
            highlight=True, 
            subtitle="For equity and autonomy", 
            padding=(1, 1)
        )
        
        # Render navigation bar
        nav_bar = self.render_navigation_bar(tick_count, env_rhythm)
        
        # Build main layout
        main_layout = Layout(name="root")
        
        # Split into nav bar at top + content below
        main_layout.split_column(
            Layout(nav_bar, name="nav", size=3),
            Layout(name="content")
        )
        
        # If a panel is active, show field + panel side by side
        if self.active_panel:
            side_panel = self._get_active_panel_content(particles, tick_count, env_rhythm)
            
            main_layout["content"].split_row(
                Layout(field_panel, name="field"),
                Layout(side_panel, name="panel", size=55)
            )
        else:
            # Clean mode - just the field
            main_layout["content"].update(field_panel)
        
        return main_layout
    
    def _get_active_panel_content(self, particles, tick_count, env_rhythm):
        """Return the rendered content for the currently active panel."""
        if self.active_panel == "legend":
            return self.build_legend_panel()
        elif self.active_panel == "stats":
            return self.render_stats(particles, tick_count, env_rhythm)
        elif self.active_panel == "diagnostics":
            return self.render_environment_diagnostics(particles, env_rhythm)
        elif self.active_panel == "reflections":
            return self.render_reflection_analysis(particles)
        elif self.active_panel == "genealogy":
            return self.render_genealogy_stats(particles)
        elif self.active_panel == "linguistics":
            return self.render_linguistic_diversity(particles)
        elif self.active_panel == "inspector":
            return self.render_most_active_with_genealogy(particles)
        else:
            return Panel("[dim]Unknown panel[/dim]")
    
    def render_genealogy_stats(self, particles):
        """Display detailed generational and lineage statistics."""
        if not particles:
            return Panel("No particles", title="Genealogy")
        
        total_particles = len(particles)
        offspring_count = sum(1 for p in particles if p.generation > 0)
        founders = sum(1 for p in particles if p.generation == 0)
        max_generation = max([p.generation for p in particles], default=0)
        
        # Count particles per generation
        gen_counts = {}
        for p in particles:
            gen_counts[p.generation] = gen_counts.get(p.generation, 0) + 1
        
        # Count unique children (each child has 2 parents, so child_ids overlap)
        # Use a set to deduplicate
        all_child_ids = set()
        for p in particles:
            all_child_ids.update(p.child_ids)
        total_unique_children = len(all_child_ids)
        
        parents_count = sum(1 for p in particles if len(p.child_ids) > 0)
        # Average per parent (each child has 2 parents, so divide by 2 for "contributions")
        avg_offspring = (total_unique_children / (parents_count / 2)) if parents_count > 0 else 0
        
        # Summary metrics
        summary = Table.grid(padding=1)
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Total Population", str(total_particles))
        summary.add_row("Founders (Gen 0)", str(founders))
        summary.add_row("Offspring", f"{offspring_count} ({(offspring_count/total_particles)*100:.0f}%)")
        summary.add_row("Max Generation", str(max_generation))
        summary.add_row("Active Parents", str(parents_count))
        summary.add_row("Avg Offspring/Parent", f"{avg_offspring:.1f}")
        
        # Generation distribution
        gen_table = Table(title="Generation Distribution", expand=True)
        gen_table.add_column("Gen", width=4, justify="center")
        gen_table.add_column("Count", width=5, justify="right")
        gen_table.add_column("Distribution", width=20)
        
        max_gen_count = max(gen_counts.values()) if gen_counts else 1
        for gen in sorted(gen_counts.keys()):
            count = gen_counts[gen]
            bar_len = int((count / max_gen_count) * 15)
            bar = "█" * bar_len
            pct = (count / total_particles) * 100
            gen_table.add_row(str(gen), str(count), f"[green]{bar}[/green] {pct:.0f}%")
        
        # Most prolific parents
        prolific = sorted(particles, key=lambda p: len(p.child_ids), reverse=True)[:3]
        if prolific and prolific[0].child_ids:
            parent_table = Table(title="Most Prolific", expand=True)
            parent_table.add_column("Particle")
            parent_table.add_column("Children", justify="right")
            parent_table.add_column("Gen", justify="center")
            
            for p in prolific:
                if p.child_ids:
                    parent_table.add_row(p.name, str(len(p.child_ids)), str(p.generation))
        else:
            parent_table = Panel("[dim]No reproduction yet[/dim]")
        
        return Panel(Group(summary, gen_table, parent_table), title="Genealogy & Lineage")
    
    def render_linguistic_diversity(self, particles):
        """Display linguistic patterns with word frequency analysis."""
        if not particles:
            return Panel("No particles", title="Linguistic Analysis")
        
        # Aggregate word frequencies across all particles
        word_counts = {}
        context_counts = {}
        total_word_uses = 0
        
        for p in particles:
            for context, words in p.word_context_frequencies.items():
                context_counts[context] = context_counts.get(context, 0) + 1
                for word, count in words.items():
                    word_counts[word] = word_counts.get(word, 0) + count
                    total_word_uses += count
        
        # Summary metrics
        summary = Table.grid(padding=1)
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Unique Words", str(len(word_counts)))
        summary.add_row("Total Uses", str(total_word_uses))
        summary.add_row("Contexts", str(len(context_counts)))
        
        # Voice profile diversity
        if particles:
            consonants = [p.voice_profile["consonant_preference"] for p in particles]
            vowels = [p.voice_profile["vowel_openness"] for p in particles]
            summary.add_row("Consonants", f"{min(consonants):.2f} - {max(consonants):.2f}")
            summary.add_row("Vowels", f"{min(vowels):.2f} - {max(vowels):.2f}")
        
        # Word frequency table - improved layout
        if word_counts:
            word_table = Table(title="Top Words", expand=True, show_header=True)
            word_table.add_column("#", width=3, justify="right")
            word_table.add_column("Word", width=14)
            word_table.add_column("Count", width=6, justify="right")
            word_table.add_column("Frequency")
            
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            max_count = sorted_words[0][1] if sorted_words else 1
            
            for i, (word, count) in enumerate(sorted_words[:10], 1):
                freq_pct = (count / total_word_uses) * 100 if total_word_uses > 0 else 0
                bar_len = int((count / max_count) * 15)
                bar = "█" * bar_len
                word_table.add_row(str(i), word, str(count), f"[cyan]{bar}[/cyan] {freq_pct:.1f}%")
        else:
            word_table = Panel("[dim]No words yet[/dim]")
        
        # Context breakdown
        if context_counts:
            ctx_table = Table(title="Emotional Contexts", expand=True, show_header=True)
            ctx_table.add_column("Context", width=15)
            ctx_table.add_column("Uses", width=6, justify="right")
            ctx_table.add_column("Share")
            
            total_ctx = sum(context_counts.values())
            for ctx, count in sorted(context_counts.items(), key=lambda x: x[1], reverse=True):
                share = (count / total_ctx) * 100
                ctx_table.add_row(ctx, str(count), f"{share:.0f}%")
        else:
            ctx_table = Panel("[dim]No contexts[/dim]")
        
        return Panel(Group(summary, word_table, ctx_table), title="Linguistic Analysis")
    
    def render_most_active_with_genealogy(self, particles):
        """Render all particles with most active featured at top."""
        if not particles:
            return Panel("No particles", title="Particle Inspector")
        
        # Type colors for visual distinction
        type_colors = {
            "cooperative": "green", "avoidant": "yellow", "chaotic": "red",
            "inquisitive": "cyan", "dormant": "blue", "resonant": "magenta"
        }
        
        # Sort particles by activation (most active first)
        sorted_particles = sorted(particles, key=lambda p: p.activation, reverse=True)
        top = sorted_particles[0]
        color = type_colors.get(top.type, "white")
        
        # Featured: Most Active Particle
        featured = Table(title=f"★ Most Active: {top.name}", expand=True, show_header=False)
        featured.add_column("Field", width=12, style="bold")
        featured.add_column("Value")
        
        featured.add_row("Type", f"[{color}]{top.type}[/{color}]")
        featured.add_row("Generation", str(top.generation))
        featured.add_row("Energy", f"{top.energy:.3f}")
        featured.add_row("Activation", f"{top.activation:.3f}")
        featured.add_row("Valence", f"{top.position[8]:.3f}")
        featured.add_row("Rhythm", f"{top.position[6]:.3f}")
        featured.add_row("Age", f"{top.position[5]:.1f}s")
        featured.add_row("Parents", str(len(top.parent_ids)))
        featured.add_row("Children", str(len(top.child_ids)))
        
        # Last reflection (full)
        reflection = top.last_reflection or "None"
        featured.add_row("Reflection", reflection[:80] + ("..." if len(reflection) > 80 else ""))
        
        # All Particles List
        all_table = Table(title="All Particles", expand=True, show_header=True)
        all_table.add_column("#", width=3, justify="right")
        all_table.add_column("Name", width=11)
        all_table.add_column("Type", width=10)
        all_table.add_column("E", width=5, justify="right")
        all_table.add_column("A", width=5, justify="right")
        all_table.add_column("Gen", width=3, justify="center")
        
        # Show all particles (panel will scroll if needed)
        for i, p in enumerate(sorted_particles, 1):
            pcolor = type_colors.get(p.type, "white")
            # Highlight most active
            if i == 1:
                all_table.add_row(
                    f"[bold yellow]{i}[/bold yellow]",
                    f"[bold]{p.name}[/bold]",
                    f"[{pcolor}]{p.type}[/{pcolor}]",
                    f"{p.energy:.2f}",
                    f"[bold]{p.activation:.2f}[/bold]",
                    str(p.generation)
                )
            else:
                all_table.add_row(
                    str(i),
                    p.name,
                    f"[{pcolor}]{p.type}[/{pcolor}]",
                    f"{p.energy:.2f}",
                    f"{p.activation:.2f}",
                    str(p.generation)
                )
        
        return Panel(Group(featured, all_table), title="Particle Inspector")

    def _enable_mouse(self):
        """Enable mouse tracking in terminal and set up non-blocking input."""
        try:
            # Save terminal settings
            self._old_term_settings = termios.tcgetattr(sys.stdin)
            # Set cbreak mode (characters available immediately, no echo)
            tty.setcbreak(sys.stdin.fileno())
            # Enable mouse tracking
            sys.stdout.write("\033[?1000h")  # Basic mouse tracking
            sys.stdout.write("\033[?1006h")  # SGR extended mode
            sys.stdout.flush()
            self._mouse_enabled = True
        except Exception as e:
            self.logger.log({"warning": f"Could not enable mouse: {e}"})
            self._mouse_enabled = False
    
    def _disable_mouse(self):
        """Disable mouse tracking and restore terminal."""
        try:
            sys.stdout.write("\033[?1006l")
            sys.stdout.write("\033[?1000l")
            sys.stdout.flush()
            if self._old_term_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term_settings)
        except Exception:
            pass
        self._mouse_enabled = False
    
    def _poll_mouse(self):
        """
        Poll for mouse input without blocking. Call this frequently in the main loop.
        Returns immediately if no input available.
        """
        if not self._mouse_enabled:
            return
        
        try:
            # Check if data available (0 timeout = non-blocking)
            while select.select([sys.stdin], [], [], 0)[0]:
                char = os.read(sys.stdin.fileno(), 1).decode('utf-8', errors='ignore')
                self._input_buffer += char
                
                # Try to parse complete mouse sequences
                self._process_input_buffer()
        except Exception:
            pass
    
    def _process_input_buffer(self):
        """Process any complete mouse sequences in the input buffer."""
        while "\033[<" in self._input_buffer:
            start = self._input_buffer.find("\033[<")
            
            # Find terminator (M for press, m for release)
            rest = self._input_buffer[start+3:]
            term_idx = -1
            for i, c in enumerate(rest):
                if c in ('M', 'm'):
                    term_idx = i
                    break
            
            if term_idx == -1:
                # Incomplete sequence, wait for more
                break
            
            # Extract sequence data
            seq_data = rest[:term_idx]
            is_press = rest[term_idx] == 'M'
            
            # Remove processed sequence from buffer
            self._input_buffer = self._input_buffer[:start] + rest[term_idx+1:]
            
            # Parse and handle click
            if is_press:
                try:
                    parts = seq_data.split(';')
                    if len(parts) >= 3:
                        button, col, row = int(parts[0]), int(parts[1]), int(parts[2])
                        # Left click on nav bar (row 2)
                        if button == 0 and row == 2:
                            self._handle_nav_click(col)
                except (ValueError, IndexError):
                    pass
    
    def _handle_nav_click(self, col: int):
        """Handle a click on the navigation bar at the given column."""
        for panel_name, (start, end) in self.panel_buttons.items():
            if start <= col <= end:
                # Toggle panel
                if self.active_panel == panel_name:
                    self.active_panel = None
                else:
                    self.active_panel = panel_name
                break
    
    def _update_cached_stats(self, particles):
        """Update cached particle statistics for the nav bar."""
        if particles:
            self._cached_stats["count"] = len(particles)
            self._cached_stats["avg_energy"] = sum(p.energy for p in particles) / len(particles)
            self._cached_stats["max_energy"] = max(p.energy for p in particles)
        else:
            self._cached_stats["count"] = 0
            self._cached_stats["avg_energy"] = 0.0
            self._cached_stats["max_energy"] = 0.0
    
    def startup(self, particles, tick_count, camera_offset, env_rhythm, delay):
        """
        Start the main simulation loop with mouse-clickable navigation.
        Click on panel names to toggle them on/off.
        """
        self._enable_mouse()
        
        try:
            self._update_cached_stats(particles)
            
            with Live(self.render_panel(particles, tick_count, camera_offset, env_rhythm), 
                      console=self.console, refresh_per_second=15, screen=True) as live:
                
                last_physics_time = time.time()
                
                while self.running:
                    # Poll mouse input (non-blocking)
                    self._poll_mouse()
                    
                    current_time = time.time()
                    
                    # Physics tick at specified delay interval
                    if current_time - last_physics_time >= delay:
                        self.particle_field.tick(particles, tick_count, env_rhythm)
                        self._update_cached_stats(particles)
                        camera_offset = self.compute_camera_target(particles, camera_offset, mode="mass")
                        tick_count += 1
                        env_rhythm = self.particle_field.compute_environmental_rhythm(tick_count)
                        last_physics_time = current_time
                    
                    # Update display
                    live.update(self.render_panel(particles, tick_count, camera_offset, env_rhythm))
                    
                    # Small sleep to prevent CPU spin (still allows ~60 checks/sec)
                    time.sleep(0.016)
                    
        finally:
            self._disable_mouse()
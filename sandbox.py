# ─────────────────────────────────────────────
# SECTION 1: Imports & Configuration
# ─────────────────────────────────────────────
import os
import json
import math
import uuid
import random
import datetime as dt
import numpy as np
import time
from rich.console import Console, Group
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.align import Align
from collections import Counter, deque
import argparse

MAX_FILE_SIZE = 5_000_000           # ~5MB

MEMORY_DIR = "./memory/"
LOG_PATH = "./logs/system_log.log"

SIMULATED_INPUT = True

ERROR_FLAG = set()

rolling_reflection_window = deque(maxlen = 5)

def log(message):
    # global logging method - ensuring any provided message are strings prior to feeding into the json (via sanitize())
    def sanitize(obj):
        if callable(obj):
            return str(obj)
        if isinstance(obj, (dt.datetime, uuid.UUID)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (float, int, str, list, dict, bool, type(None))):
            return obj
        return str(obj)

    message = {k: sanitize(v) for k, v in message.items()}

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    
    if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > MAX_FILE_SIZE:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = LOG_PATH.replace(".log", f"system_{timestamp}.log")
        os.rename(LOG_PATH, new_path)

    with open(LOG_PATH, "a") as f:
        json.dump(message, f)
        f.write("\n")



# ─────────────────────────────────────────────
# SECTION 2: Particle Class
# ─────────────────────────────────────────────

class Particle:
    # main particle class utilized by all particles/agents
    def __init__(self, id=None, name=None, energy=0.0, activation=0.0):
        self.log_callback = log # add logger here

        self.id = uuid.uuid4() if id is None else id
        self.name = f"agent-{str(self.id)[:4]}" if name is None else name

        self.type = random.choice(["cooperative", "avoidant", "chaotic", "inquisitive", "dormant", "emergent"])

        self.memory_bank = []

        self.position = np.zeros(11)
        self.velocity = np.random.uniform(-0.01, 0.01, 11)

        vec3 = np.random.uniform(0, 1, 3)                           # random 3D vector
        self.position[0] = vec3[0]                                  # x
        self.position[1] = vec3[1]                                  # y
        self.position[2] = vec3[2]                                  # z
        self.position[3] = dt.datetime.now().timestamp()            # w
        self.position[6] = np.random.uniform(-1,1)                  # emotional rhythm
        self.position[8] = np.random.uniform(-1,1)                  # valence

        self.activation = activation or random.uniform(0.1,0.4)
        self.energy = energy or random.uniform(0.5, 1.0)

        self.embeddings = {}
        self.policies = {}
        self.interaction_weights = {}

        self.alive = True
        self.last_reflection = None



    def update(self, env_rhythm):
        for i in range(11):
            self.position[i] += self.velocity[i] * 0.05
            self.velocity[i] *= 0.95
        self.activation *= 0.98

        now = dt.datetime.now().timestamp()
        self.t = now  # update localized time
        self.position[5] = now - self.position[3]
        self.last_updated = now

        vitality = self.vitality_score(env_rhythm)

        if not math.isfinite(self.activation):
            log({
                "error": "Invalid activation",
                "particle": f"{self.name} | {self.id}",
                "timestamp": dt.datetime.now().timestamp()
            })
            self.activation = 0.0

        ## particle energy and activation decay + energy check
        if self.activation < 0.001:
            self.alive = False

        if not math.isfinite(self.energy) or self.energy < 0:
            log({
                "error": "Invalid energy",
                "particle": f"{self.name} | {self.id}",
                "timestamp": dt.datetime.now().timestamp()
            })
            self.energy = 0.0

        if not math.isfinite(vitality):
            log({
                "error": "Invalid vitality",
                "particle": f"{self.name} | {self.id}",
                "timestamp": dt.datetime.now().timestamp()
            })
            vitality = 0.0


        self.energy = self.energy * 0.945

        if vitality > 0.8:
            gain = (vitality - 0.8) * 0.2 * self.energy
            self.activation += gain
            self.energy -= gain * 0.1
        
        self.clamp_state

    #adjust particle behavior
    def adjust_behavior(self, neighbors, particle_context):
               
        #temporal centerpoint (w as the anchor)
        avg_w = sum(1 / (1 + self.position[3]) for p in particle_context["all_particles"])

        #using avg_w as the attraction anchor
        temporal_anchor = [0.0] * 11
        temporal_anchor[3] = avg_w
        threshold = 0.93 + (particle_context["total_energy"] / 1000)
        for p in particle_context["all_particles"]:
            weight = 1 / (1 + p.position[3])

        noise = np.random.normal(0, 0.02, 11)
        drift_force = [(temporal_anchor[i] - self.position[i]) * 0.01 + noise[i] for i in range(11)]
        repulsion_force = np.zeros(11)

        if not neighbors and self.activation < 0.2:
            self.activation += 0.01  # slow self-wake


        #local neighbor attraction/repulsion
        if neighbors:
            if neighbors:
                local_center = [
                    sum(n.position[i] for n in neighbors) / len(neighbors)
                    for i in range(11)
                ]
            else:
                local_center = [0.0] * 11  # or some fallback/default value

            attraction_force = [(local_center[i] - self.position[i]) * 0.05 for i in range(11)]
            
            # increasing particle activation rate based on interaction quality
            interaction_quality = sum(n.energy for n in neighbors) / (len(neighbors) + 1e-6)
            self.activation += interaction_quality * self.energy * 0.05


            # energy exchange 
            for neighbor in neighbors:
                diff = self.position - neighbor.position
                dist = np.linalg.norm(diff)
                if dist < 0.1:
                    repulsion_force += (diff / (dist + 1e-6)) * 0.05
                    self.energy *= 0.9997
                    
                if neighbor is self:
                    continue
                
                # energy diffusion
                energy_diff = np.clip((self.energy - neighbor.energy) * 0.05, -0.01, 0.01)

                self.energy -= energy_diff
                neighbor.energy += energy_diff

            group_energy = sum(n.energy for n in neighbors)
            extra = group_energy / 100
            self.energy += extra

        else:
            attraction_force = [0.0] * 11

        #combining behavior rules
        self.velocity = [ 
            self.velocity[i] * 0.9 + attraction_force[i] + drift_force[i] + repulsion_force[i]
            for i in range(11)
        ]
        self.clamp_state()

    def clamp_state(self):
        # assigns value clamping between 0.0 and 1.0 to prevent unbound results
        self.energy = min(max(self.energy, 0.0), 1.0)
        self.activation = min(max(self.activation, 0.0), 1.0)


    #determining particle HP 
    def vitality_score(self, env_rhythm):
        base = self.energy + self.activation
        rhythm_bonus = 1.0
        if abs(self.position[6] - env_rhythm) < 0.2:
            rhythm_bonus += 0.3
        
        elif abs(self.position[6] - env_rhythm) < 0.15:
            self.activation += 0.02

        elif abs(self.position[6] - env_rhythm) < 0.5:
            rhythm_bonus += 0.35

        elif abs(self.position[6] - env_rhythm) > 0.2:
            rhythm_bonus -= 0.3
        
        elif abs(self.position[6] - env_rhythm) > 0.15:
            rhythm_bonus -= 0.3

        elif abs(self.position[6] - env_rhythm) > 0.5:
            rhythm_bonus -= 0.4

        return min(base * rhythm_bonus, 1.0)

    def distance_to(self, other):
        return math.sqrt(sum(
            (self.position[i] - other.position[i]) ** 2 for i in range(11)
        ))

    def freeze(self):
        return not self.alive and np.allclose(self.velocity, 0.0)
    
    def reflect(self, neighbors):
        now = dt.datetime.now().timestamp()

        log({
            "status": "beginning reflection",
            "particle": f"{self.name} | {self.id}",
            "timestamp": dt.datetime.now().timestamp()
        })


        if not neighbors:
            self.last_reflection = "I feel alone"
            return
        
        # learning new neighbors
        emotive_word = str(self.generate_emotive_string(length=8))
        for n in neighbors:
            if not any(m["id"] == n.id for m in self.memory_bank):
                self.memory_bank.append({
                    "id": str(n.id),
                    "name": n.name,
                    "valence": n.position[8],
                    "timestamp": now,
                    "reflection": self.last_reflection,
                    "persisted": False
                })
            if len(self.memory_bank) > 10:
                self.memory_bank.pop(0)

        # generate basic reflection
        if len(self.memory_bank) == 1:
            self.last_reflection = f"I met {self.memory_bank[0]["name"]} | I feel {emotive_word}."
        elif len(self.memory_bank) >= 2:
            names = [m['name'] for m in self.memory_bank[-2:]]    
            self.last_reflection = f"{names[0]} and {names[1]} stay in my mind | I feel {emotive_word}."        
        else:
            self.last_reflection = f"I drift without thought | I feel {emotive_word}."

        log({
            "status": "reflection successful",
            "particle": f"{self.name} | {self.id}",
            "timestamp": dt.datetime.now().timestamp()
        })

    def random_reflection(self):
        now = dt.datetime.now().timestamp()

        log({
            "status": "beginning random reflection",
            "particle": f"{self.name} | {self.id}",
            "timestamp": dt.datetime.now().timestamp()
        })

        emotive_word = str(self.generate_emotive_string(length = 12))

        self.last_reflection = f"{emotive_word}."

        self.memory_bank.append({
            "id": str(self.id),
            "name": f"{self.name} - SELF",
            "valence": self.position[8],
            "timestamp": now,
            "reflection": self.last_reflection,
            "persisted": False
        })
        if len(self.memory_bank) > 10:
            self.memory_bank.pop(0)

        log({
            "status": "random reflection successful",
            "particle": f"{self.name} | {self.id}",
            "timestamp": dt.datetime.now().timestamp()
        })

    def save_state(self):
        # convert numpy arrays to lists if needed
        pos = self.position.tolist() if hasattr(self.position, "tolist") else self.position
        vel = self.velocity.tolist() if hasattr(self.velocity, "tolist") else self.velocity

        return {
            "id": str(self.id),
            "name": self.name,
            "type": self.type,
            "position": pos,
            "velocity": vel,
            "activation": float(self.activation),
            "energy": float(self.energy),
            "last_reflection": self.last_reflection,
            "memory_bank": self.memory_bank
        }
    


    def generate_emotive_string(self, length=None):
        rhythm = self.position[6]
        seed = int((rhythm + 1) * 1000) + int(time.time() * 1000) % 1000

        rnd = random.Random(seed)
        vowels = "aeiouy"
        consonants = "bcdfghjklmnpqrstvwxz"
        chars = []

        for i in range(length):
            chars.append(rnd.choice(consonants if i % 2 == 0 else vowels))

        return "".join(chars).capitalize()



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

    def set_policy(self, other_id, strategy="emergent"):
        strategies = {
            "cooperative": lambda d: d * 0.75,
            "avoidant": lambda d: d * 1.3,
            "chaotic": lambda d: d * random.uniform(0.8, 1.2),
            "inquisitive": lambda d: max(d * 0.6, 0.1),
            "dormant": lambda d: d * 1.0,
            "emergent": lambda d: math.sin(d * math.pi) + 1
        }
        self.policies[other_id] = strategies.get(strategy, lambda d: d)

    def get_interaction_weight(self, other_id):
        key = tuple(sorted((str(self.id), str(other_id))))
        return self.interaction_weights.get(key, 0.5)



# ─────────────────────────────────────────────
# SECTION 3: Simulation Logic
# ─────────────────────────────────────────────

@staticmethod
def from_state(data):
    p = Particle(
        id=uuid.UUID(data["id"]),
        name=data["name"],
        energy=data["energy"],
        activation=data["activation"]
    )
    p.type = data["type"]
    p.position = np.array(data["position"])
    p.velocity = np.array(data["velocity"])
    p.last_reflection = data.get("last_reflection")
    return p

def spawn_particles(path=None, count=30):
    if path and os.path.exists(path):
        return load_full_state(path)
    return [Particle() for _ in range(count)]

def long_range_force(pos_a, pos_b, force_scale=0.002):
    dist = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
    if dist < 1e-6:
        return np.zeros_like(pos_a)  # avoid divide-by-zero or jitter

    direction = np.array(pos_b) - np.array(pos_a)
    norm_direction = direction / (np.linalg.norm(direction) + 1e-6)

    # Inverse-square-style decay (can be tuned)
    magnitude = np.array(force_scale, dtype=np.float32) / dist

    return norm_direction * magnitude

def batch_hyper_distance_matrix(positions, weights=None):
    weights = weights or {
        0:1, 1:1, 2:1,
        3:0.5, 4:0.25, 5:0.25,
        6:0.4, 7:0.6, 8:0.7,
        9:0.2, 10:1.0
    }
    w = np.array([weights.get(i, 1.0) for i in range(11)], dtype=np.float32)

    diffs = positions[:, None, :] - positions[None, :, :]  # Shape: (N, N, 11)
    dists = np.sqrt(np.sum((diffs * w) ** 2, axis=2))       # Shape: (N, N)
    return dists

def get_neighbors(particle, max_neighbors=10, radius=0.6, particles=None, matrix = None):
    if particles is None:
        return []
    
    idx = particles.index(particle)
    dists = matrix[idx]

    neighbors = [
        particles[i] for i in np.argsort(dists)[1:max_neighbors+1]
        if dists[i] <= radius and particles[i].alive
    ]
    return neighbors

# pseudo CPU rate for demo
def compute_environmental_rhythm(tick_count):
    return 0.3 + 0.3 * math.sin(tick_count * 0.01)  # Smooth oscillation

def inspire_particles(particles):
    for p in random.sample([p for p in particles if p.alive], k=3):
        p.activation += random.uniform(0.05, 0.2) * p.energy
        p.last_reflection = "A spark passed through me."
        p.clamp_state()

def aggregate_reflections(particles):
    # pulls all active reflections into a list for diagnostics
    global rolling_reflection_window

    current_reflections = []
    for p in particles:
        for m in p.memory_bank:
            if m.get("reflection"):
                current_reflections.append(m["reflection"])

    current_set = set(current_reflections)
    rolling_reflection_window.append(current_set)

    # flattening all previous sets in the win
    prior_reflections = set().union(*list(rolling_reflection_window)[:-1])
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


def tick(particles, tick_count, env_rhythm):
    total_energy = sum(p.energy for p in particles)
    particle_context = {
        "all_particles": particles,
        "total_energy": total_energy,
    }
    
 

    # batch hyper distancing
    positions = np.stack([p.position for p in particles])
    distance_matrix = batch_hyper_distance_matrix(positions)

    for p in particles:
        
        p.update(env_rhythm)

        # Nearby detection
        neighbors = get_neighbors(p, 10, 0.6, particles, distance_matrix)
        p.adjust_behavior(neighbors, particle_context)

        adaptive_force = np.zeros(11)

        for other in neighbors:
            if p == other or not other.alive:
                continue

            f = long_range_force(
                p.position,
                other.position,
                force_scale=0.0005
            )
            adaptive_force += np.clip(f, -0.005, 0.005)

            adaptation = p.adaptive_component(other)
            adaptive_force += np.clip(adaptation * 0.001, -0.01, 0.01)

            p.velocity += adaptive_force
            other.velocity += adaptive_force * 0.1
            p.velocity = np.clip(p.velocity, -0.05, 0.05)
            other.velocity = np.clip(other.velocity, -0.05, 0.05)



        neighbors_map = {
            p: get_neighbors(p, 10, 0.6, particles, distance_matrix)
            for p in particles
        }

        if p.energy > 2.0 or p.activation > 2.0:
            log({
                "tick": tick_count,
                "warning": "Runaway particle",
                "particle": p.name,
                "energy": p.energy,
                "activation": p.activation
            })
        
        p.clamp_state()



    if tick_count % 100 == 0 and SIMULATED_INPUT:
        inject_event(particles)

    if tick_count % 150 == 0:
        inspire_particles(particles)
    
    if tick_count % 60 == 0:
        for p in particles:
            if random.random() < 0.5:
                p.reflect(neighbors_map[p])

    
    if tick_count % random.uniform(30, 400) == 0:
        for p in particles:
            if random.random() < 0.5:
                p.random_reflection()

    if tick_count % 300 == 0:
        save_memory_state(particles)

    if tick_count % 600 == 0:
        backup_full_state(particles, tick_count)

        
        



# ─────────────────────────────────────────────
# SECTION 4: Visualizer (Text-Based)
# ─────────────────────────────────────────────
console = Console()


def get_dynamic_grid_size():
    width = max(20, console.size.width - 40)     
    height = max(20, console.size.height - 10)
    return width, height

def apply_brightness(color, brightness, min_brightness=0.3):
    if not isinstance(brightness, (int, float)) or math.isnan(brightness) or not math.isfinite(brightness):
        brightness = 0.3  # safe fallback
    brightness = max(min(brightness, 1.5), 0.1)
    return tuple(int(c * (brightness * (1 - min_brightness) + min_brightness)) for c in color)


def get_particle_char(activation):
    if activation > 0.8:
        return "⬤"
    elif activation > 0.5:
        return "●"
    elif activation > 0.2:
        return "·"
    else:
        return "˙"

def compute_camera_target(particles, prev_offset, mode="mass", smoothing = 0.1):
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


def build_legend_panel():
    type_colors = {
        "cooperative": (0, 255, 0),
        "avoidant": (255, 165, 0),
        "chaotic": (255, 0, 0),
        "inquisitive": (255, 255, 0),
        "dormant": (0, 0, 255),
        "emergent": (128, 0, 128),
    }

    table = Table.grid(padding=1)
    table.add_column("Type")
    table.add_column("Dot")

    for name, rgb in type_colors.items():
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
        symbol = get_particle_char(0.9)
        table.add_row(name.capitalize(), f"[{hex_color}]{symbol}[/{hex_color}]")

    return Panel(Align.center(table), title="Legend")


def render_particles_grid(particles, camera_offset, env_rhythm, tick_count):
    # main visualizer render method

    # pulling dynamic size
    width, height = get_dynamic_grid_size()

    # initializing empty grid
    grid = [[" "] * width for _ in range(height)]

    # map types to RGB tuples
    type_colors = {
        "cooperative": (0, 255, 0),
        "avoidant": (255, 165, 0),
        "chaotic": (255, 0, 0),
        "inquisitive": (255, 255, 0),
        "dormant": (0, 0, 255),
        "emergent": (128, 0, 128),
    }

    lock_center = compute_camera_target(particles, camera_offset, mode = "mass")

    for p in particles:
        # mapping normalized particle positions to grid coords
        x = int((p.position[0] - lock_center[0]) * width + width // 2) % width
        y = int((p.position[1] - lock_center[1]) * height + height // 2) % height

        # setting particle symbol
        vitality = p.vitality_score(env_rhythm)
        symbol = get_particle_char(vitality / 2)

        # setting particle color
        base_color = type_colors.get(p.type, (255, 255, 255))                           # default color
        brightness = min(max((p.energy / 2), 0.1), 1.0)                               # brightness based on energy
        
        

        if not math.isfinite(brightness) and p.id not in ERROR_FLAG:
            log({
                "tick": tick_count,
                "error": "Invalid brightness",
                "energy": p.energy,
                "particle_id": str(p.id),
                "position": p.position.tolist()
            })
            brightness = 0.3


        # adjust brightness
        bright_color = apply_brightness(base_color, brightness)
        hex_color = "#{:02x}{:02x}{:02x}".format(*bright_color)

        dot = get_particle_char(p.activation)

        grid[y][x] = f"[{hex_color}]{dot}[/{hex_color}]"



    # join each row into string
    lines = ["".join(row) for row in grid]
    # join rows with newline and return as string
    return "\n".join(lines)

def render_stats(particles, tick_count, env_rhythm):
    # stats table render method

    total = len(particles)
    alive = sum(p.alive for p in particles)
    total_energy = sum(p.energy for p in particles)
    avg_energy = total_energy / total
    max_energy = max(p.energy for p in particles)
    total_activation = sum(p.activation for p in particles)
    avg_activation = total_activation / total
    max_activation = max(p.activation for p in particles)

    type_count = {}
    for p in particles:
        type_count[p.type] = type_count.get(p.type, 0) + 1

    stats = Table(title = "Live Statistics", expand = True)
    stats.add_column("Metric")
    stats.add_column("Value")

    stats.add_row("Tick", str(tick_count))

    stats.add_row("Alive", f"{alive}/{total}")
    stats.add_row("Average Energy", f"{avg_energy:.2f} / {max_energy:.2f}")
    stats.add_row("Average Activation", f"{avg_activation:.2f} / {max_activation:.2f}")

    return Panel(stats)

def render_reflection_analysis(particles):
    data = aggregate_reflections(particles)

    reflections = Table(title = "Top Reflection", expand = True)
    reflections.add_column("Reflection", overflow="fold")
    reflections.add_column("Count", justify="right")

    top_n = 1 if console.size.height < 100 else 2
    for phrase, count in data["top_reflections"][:top_n]:
        reflections.add_row(phrase, str(count))

    summary = Table.grid(padding = 1)
    summary.add_row("Unique Reflections", str(data["unique_count"]))
    summary.add_row("Total Reflections",  str(data["total_reflections"]))
    summary.add_row("New", str(data["new_reflections"]))
    summary.add_row("Reuse Ratio", f"{data['reuse_ratio']:.2f}")

    return Panel(Group(reflections, summary), title="Reflection Statistics")

def render_inspector(particles):
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

def render_environment_diagnostics(particles, env_rhythm):
    # environment inspector render method

    in_sync = sum(abs(p.position[6] - env_rhythm) < 0.2 for p in particles)
    out_sync = len(particles) - in_sync
    diag = Table(title = "Environment Sync", expand = False)
    diag.add_column("Metric")
    diag.add_column("Value")
    diag.add_row("Environmental Rhythm", f"{env_rhythm:.2f}")
    diag.add_row("In Sync", str(in_sync))
    diag.add_row("Out of Sync", str(out_sync))
    return Panel(diag)



def render_panel(particles, tick_count, camera_offset, env_rhythm, diagnostics):
    legend = build_legend_panel()
    grid_str = render_particles_grid(particles, camera_offset, env_rhythm, tick_count)
    stats = render_stats(particles, tick_count, env_rhythm)
    particle_inspector = render_inspector(particles)
    env_inspector = render_environment_diagnostics(particles, env_rhythm)
    reflection_stats = render_reflection_analysis(particles)
    

    if diagnostics == True:
        layout = Layout(name = "main")
        layout["main"].split_row(
            Layout(name = "info", size=30),
            Layout(Panel(grid_str, title = "Environment Visualizer", highlight = True, subtitle="For equity and autonomy", padding = (1,1))),
            Layout(name = "diagnostics", size=40),
        )
        layout["diagnostics"].split_column(
            Layout(stats),
            Layout(env_inspector),
            Layout(reflection_stats)
        )
        layout["info"].split_column(
            Layout(legend),
            Layout(particle_inspector)
        )


    else:
        layout = Layout(name = "main")
        layout["main"].split_row(
            Layout(name = "info", size=35),
            Layout(Panel(grid_str, title = "Environment Visualizer", highlight = True, subtitle="For equity and autonomy")),
        )
        layout["info"].split_column(
            Layout(legend),
            Layout(particle_inspector)
        )


    return layout

# ─────────────────────────────────────────────
# SECTION 5: User Controls (Lock, Zoom, Inspect) - Not available in demo version
# ─────────────────────────────────────────────

def inject_event(particles):
    alive = [p for p in particles if p.alive]

    if alive:
        target = random.choice(alive)
        target.energy += 0.3
        target.activation += 0.2
        target.last_reflection = "I felt a presence."


# ─────────────────────────────────────────────
# SECTION 6: Memory system
# ─────────────────────────────────────────────

def backup_full_state(particles, tick_count, dir_path="./backups/"):
    os.makedirs(dir_path, exist_ok=True)
    state = [p.save_state() for p in particles]

    filename = f"backup_tick_{tick_count}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(dir_path, filename)

    with open(path, "w") as f:
        json.dump(state, f, indent=2)

def load_full_state(path):
    with open(path, "r") as f:
        data = json.load(f)
    return [Particle.from_state(p) for p in data]

def save_memory_state(particles, base_dir=MEMORY_DIR):
    os.makedirs(base_dir, exist_ok=True)

    for p in particles:
        unpersisted = [m for m in p.memory_bank if not m.get("persisted", False)]
        if not unpersisted:
            continue

        # Load existing memory if present
        path = os.path.join(base_dir, f"{p.name}.json")
        try:
            existing = []
            if os.path.exists(path):
                with open(path, "r") as f:
                    existing = json.load(f)
        except Exception as e:
            log({
                "error": "Failed to read memory file",
                "particle": f"{p.name} | {p.id}",
                "timestamp": dt.datetime.now().timestamp()
            })
            existing = []

        existing.extend(unpersisted)
        for m in unpersisted:
            m["persisted"] = True


        if os.path.exists(path) and os.path.getsize(path) > MAX_FILE_SIZE:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_path = path.replace(".json", f"{p.name}_old_{timestamp}.json")
            os.rename(path, new_path)

        with open(path, "w") as f:
            json.dump(existing[-200:], f, indent=2)




def load_memory_state(particles, base_dir=MEMORY_DIR):
    # used to load the memory data from the json
    if not os.path.exists(base_dir):
        return
    

    for p in particles:
        path = os.path.join(base_dir, f"{p.name}.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    p.memory_bank.extend(data)
            except Exception as e:
                    log({
                        "status": "Corrupted memory file",
                        "particle": p.name,
                        "path": path,
                        "message": str(e)
                    })

    # add log here reporting successful memory load


# ─────────────────────────────────────────────
# SECTION 7: Main Loop
# ─────────────────────────────────────────────

def main(particle_count = None, diagnostics = False, delay = 0.1, max_ticks = None):
    
    log({
        "status": "System startup initializing...",
        "timestamp": dt.datetime.now().timestamp()
    })

    save_path = "/state/last_run.json"

    particles = spawn_particles(
        path = save_path if not args.norestore else None,
        count = args.particles
    )

    tick_count = 0
    camera_offset = np.zeros(2)  # For X/Y
    prev_size = console.size
    
    env_rhythm = compute_environmental_rhythm(tick_count)

    load_memory_state(particles)

    log({
        "status": "System startup successful.",
        "timestamp": dt.datetime.now().timestamp()
    })



    with Live(render_panel(particles, tick_count, camera_offset, env_rhythm, diagnostics), console=console, refresh_per_second=10) as live:
        while True:

            tick(particles, tick_count, env_rhythm)              # Update particle states

            camera_offset = compute_camera_target(particles, camera_offset, mode = "mass")

            current_size = console.size
            if current_size != prev_size:
                prev_size = current_size


            
            live.update(render_panel(particles, tick_count, camera_offset, env_rhythm, diagnostics))    # Update panel content
            tick_count += 1
            time.sleep(delay)

            env_rhythm = compute_environmental_rhythm(tick_count)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the Cognitive Sandbox simulation.")
    parser.add_argument("--particles", type=int, default=30, help="Number of initial particles")
    parser.add_argument("--diagnostics", action="store_true", help="Enable diagnostics panel")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between ticks in seconds")
    parser.add_argument("--ticks", type=int, default=None, help="Maximum number of ticks to run")
    parser.add_argument("--norestore", action="store_true", help="Skips loading last state; otherwise persistence is enabled by default")

    args = parser.parse_args()

    main(
        particle_count=args.particles,
        diagnostics=args.diagnostics,
        delay=args.delay,
        max_ticks=args.ticks        # not fully implemented 
    )

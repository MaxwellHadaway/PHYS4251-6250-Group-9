# PHYS4251/6250
# Group 9
# Maxwell Hadaway, Kenechukwu Aniedobe, Michael Olatunji
# Batch simulation version (no GUI) for parameter sweeps

import random
import math

# --------------------
# CONFIG
# --------------------
WIDTH, HEIGHT = 1000, 1000

TIME_SCALE = 0.2

DNA_RADIUS = 10
CAS9_RADIUS = 20

DIRECTIONS = [
    (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, -1), (1, -1), (-1, 1), (1, 1)
]

UPDATE_INTERVAL_MS = 20       # used as fixed dt = 0.02 s
BIND_TIME_JUNK = 1.0          # fallback, junk uses its own sampled time
BIND_TIME_VIRUS = 2.0         # fallback, virus uses its own sampled time

COOLDOWN_JUNK = 2.5
COOLDOWN_VIRUS_FAIL = 2.5     # virus ignore time after failed check

EXPERIMENT_DURATION = 10.0    # <<< changed from 20.0 to 10.0 s

COLLISION_BUFFER = 0.0

# success probability: P(immediate cut when Cas9 hits a virus)
SUCCESS_PROB = 0.8

# Scale virus dwell times so they don't exceed ~12 s
VIRUS_TIME_SCALE = 0.27   # 0.27 * 43.7 ≈ 11.8 s max for strongest bind

# fixed Cas9 speed (same as GUI default)
CAS9_SPEED = 20

# output file for MATLAB
OUTPUT_FILE = "batch_results.txt"

# --------------------
# JUNK DNA MISMATCH DISTRIBUTION
# --------------------
JUNK_DISTANCES = list(range(1, 11))   # 1..10

_raw_p = [0.75 * (0.25 ** (n - 1)) for n in JUNK_DISTANCES]
_raw_sum = sum(_raw_p)
JUNK_PROBS = [v / _raw_sum for v in _raw_p]


def sample_junk_bind_time():
    """
    Sample a mismatch 'distance' n from {1..10} with P(n),
    then compute t_bound = 0.0026 * exp(0.9729 * n).
    (Your first MATLAB model for junk DNA.)
    """
    n = random.choices(JUNK_DISTANCES, weights=JUNK_PROBS, k=1)[0]
    t_bound = 0.0026 * math.exp(0.9729 * n)
    return t_bound


# --------------------
# VIRUS MISMATCH / BIND-TIME MODEL (from your MATLAB code)
# --------------------
def generate_virus_dwell_times(n):
    """
    MATLAB logic:

      x = 1:10;
      p = 0.75 * 0.25^(x-1); p = p / sum(p);
      distance = randsample(x, ..., p);
      distance' = -distance + 11;
      t_bound = 0.0026 * exp(0.9729 * distance');

    Then scaled by VIRUS_TIME_SCALE so max is ~12 s.
    """
    x_vals = list(range(1, 11))

    p = [0.75 * (0.25 ** (xi - 1)) for xi in x_vals]
    s = sum(p)
    p = [pi / s for pi in p]

    dwell_times = []

    for _ in range(n):
        distance = random.choices(x_vals, weights=p, k=1)[0]
        distance_prime = -distance + 11
        t_bound = 0.0026 * math.exp(0.9729 * distance_prime)
        t_bound *= VIRUS_TIME_SCALE
        dwell_times.append(t_bound)

    return dwell_times


# --------------------
# DUMMY CANVAS (no-op, replaces Tk canvas)
# --------------------
class DummyCanvas:
    def create_oval(self, *args, **kwargs):
        return 0

    def coords(self, *args, **kwargs):
        pass

    def delete(self, *args, **kwargs):
        pass


# --------------------
# CLASSES
# --------------------
class DNA:
    def __init__(self, canvas, x, y, radius, kind):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.r = radius
        self.kind = kind        # 'junk' or 'virus'
        self.alive = True
        self.dir = random.choice(DIRECTIONS)
        self.cooldown_until = 0.0
        self.bound = False      # prevents multi-Cas9 binding

        if self.kind == "junk":
            self.junk_bind_time = sample_junk_bind_time()
        else:
            self.junk_bind_time = None

        # virus-specific bind time (set later in create_dna)
        self.virus_bind_time = None

        # graphical stuff is ignored in batch mode, but we keep calls
        self.id = canvas.create_oval(
            self.x - self.r, self.y - self.r,
            self.x + self.r, self.y + self.r
        )

    def move_step(self, speed):
        if not self.alive:
            return

        dx, dy = self.dir
        new_x = self.x + dx * speed * 0.3
        new_y = self.y + dy * speed * 0.3

        # walls
        if new_x - self.r < 0:
            new_x = self.r
            dx = -dx
        elif new_x + self.r > WIDTH:
            new_x = WIDTH - self.r
            dx = -dx

        if new_y - self.r < 0:
            new_y = self.r
            dy = -dy
        elif new_y + self.r > HEIGHT:
            new_y = HEIGHT - self.r
            dy = -dy

        self.x = new_x
        self.y = new_y
        self.dir = (dx, dy)

        if random.random() < 0.1:
            self.dir = random.choice(DIRECTIONS)

        self.canvas.coords(
            self.id,
            self.x - self.r, self.y - self.r,
            self.x + self.r, self.y + self.r
        )


class Cas9:
    def __init__(self, canvas, x, y, radius):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.r = radius

        self.dir = random.choice(DIRECTIONS)
        self.state = "free"       # 'free', 'bound_junk', 'bound_virus'
        self.bound_to = None
        self.bound_until = 0.0
        self.alive = True

        self.id = canvas.create_oval(
            self.x - self.r, self.y - self.r,
            self.x + self.r, self.y + self.r
        )

    def bind_to(self, dna, now, bind_time, state_name):
        self.state = state_name
        self.bound_to = dna
        self.bound_until = now + bind_time
        self.x, self.y = dna.x, dna.y
        self.update_canvas_pos()

    def update_canvas_pos(self):
        self.canvas.coords(
            self.id,
            self.x - self.r, self.y - self.r,
            self.x + self.r, self.y + self.r
        )

    def move_step(self, speed, now):
        if not self.alive:
            return

        # --------------
        # BOUND STATES
        # --------------
        if self.state in ("bound_junk", "bound_virus"):
            if self.bound_to and self.bound_to.alive:
                self.x, self.y = self.bound_to.x, self.bound_to.y
                self.update_canvas_pos()

            if now >= self.bound_until:
                # junk: detach and put junk on cooldown
                if self.state == "bound_junk":
                    if self.bound_to and self.bound_to.alive:
                        self.bound_to.cooldown_until = now + COOLDOWN_JUNK
                    self.state = "free"
                    self.bound_to = None

                # virus: this state now means FAILED recognition dwell
                elif self.state == "bound_virus":
                    if self.bound_to and self.bound_to.alive:
                        self.bound_to.bound = False
                        # virus cooldown was set at collision time
                    self.state = "free"
                    self.bound_to = None

            return

        # --------------
        # FREE STATE (Brownian)
        # --------------
        dx, dy = self.dir
        new_x = self.x + dx * speed
        new_y = self.y + dy * speed

        if new_x - self.r < 0:
            new_x = self.r
            dx = -dx
        elif new_x + self.r > WIDTH:
            new_x = WIDTH - self.r
            dx = -dx

        if new_y - self.r < 0:
            new_y = self.r
            dy = -dy
        elif new_y + self.r > HEIGHT:
            new_y = HEIGHT - self.r
            dy = -dy

        self.x = new_x
        self.y = new_y
        self.dir = (dx, dy)

        if random.random() < 0.2:
            self.dir = random.choice(DIRECTIONS)

        self.update_canvas_pos()


# --------------------
# UTILS
# --------------------
def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def create_dna(canvas, num_junk, num_virus):
    dna_list = []

    virus_dwell_times = generate_virus_dwell_times(num_virus)
    virus_idx = 0

    # junk DNA
    for _ in range(num_junk):
        x = random.uniform(DNA_RADIUS, WIDTH - DNA_RADIUS)
        y = random.uniform(DNA_RADIUS, HEIGHT - DNA_RADIUS)
        dna_list.append(DNA(canvas, x, y, DNA_RADIUS, "junk"))

    # virus DNA
    for _ in range(num_virus):
        x = random.uniform(DNA_RADIUS, WIDTH - DNA_RADIUS)
        y = random.uniform(DNA_RADIUS, HEIGHT - DNA_RADIUS)
        d = DNA(canvas, x, y, DNA_RADIUS, "virus")
        d.virus_bind_time = virus_dwell_times[virus_idx]
        virus_idx += 1
        dna_list.append(d)

    return dna_list


def create_cas9(canvas, num_cas9):
    lst = []
    for _ in range(num_cas9):
        x = random.uniform(CAS9_RADIUS, WIDTH - CAS9_RADIUS)
        y = random.uniform(CAS9_RADIUS, HEIGHT - CAS9_RADIUS)
        lst.append(Cas9(canvas, x, y, CAS9_RADIUS))
    return lst


# --------------------
# SINGLE SIMULATION
# --------------------
def run_single_sim(num_junk, num_virus, num_cas9):
    """
    Run one 10 s simulation and return:
      - virus_kill_density = kills / initial_virus
    """
    canvas = DummyCanvas()

    dna_list = create_dna(canvas, num_junk, num_virus)
    cas9_list = create_cas9(canvas, num_cas9)

    capture_count = 0  # number of virus kills

    sim_time = 0.0
    dt = UPDATE_INTERVAL_MS / 1000.0  # 0.02 s

    while sim_time < EXPERIMENT_DURATION:
        # movement
        for d in dna_list:
            d.move_step(CAS9_SPEED)
        for c in cas9_list:
            c.move_step(CAS9_SPEED, sim_time)

        # collisions
        for c in cas9_list:
            if not c.alive or c.state != "free":
                continue

            for d in dna_list:
                if not d.alive:
                    continue

                # global cooldown (junk or virus)
                if sim_time < d.cooldown_until:
                    continue

                # virus already occupied by a failed-check Cas9
                if d.kind == "virus" and d.bound:
                    continue

                if distance(c, d) <= (c.r + d.r + COLLISION_BUFFER):
                    if d.kind == "junk":
                        bind_time = d.junk_bind_time if d.junk_bind_time is not None else BIND_TIME_JUNK
                        c.bind_to(d, sim_time, bind_time, "bound_junk")

                    else:  # virus
                        p = random.random()
                        if p <= SUCCESS_PROB:
                            # IMMEDIATE SUCCESS: no dwell, both disappear
                            capture_count += 1
                            d.alive = False
                            canvas.delete(d.id)
                            c.alive = False
                            canvas.delete(c.id)
                        else:
                            # FAILED MATCH: dwell for virus_bind_time, then detach
                            d.bound = True
                            d.cooldown_until = sim_time + COOLDOWN_VIRUS_FAIL
                            bind_time = (d.virus_bind_time * TIME_SCALE) if d.virus_bind_time is not None else BIND_TIME_VIRUS
                            c.bind_to(d, sim_time, bind_time, "bound_virus")
                        # either way, this Cas9 is done with collisions this step
                    break

        # prune dead
        dna_list = [d for d in dna_list if d.alive]
        cas9_list = [c for c in cas9_list if c.alive]

        sim_time += dt

    if num_virus > 0:
        kill_density = capture_count / num_virus
    else:
        kill_density = 0.0

    return kill_density


# --------------------
# BATCH LOOP
# --------------------
def main():
    with open(OUTPUT_FILE, "w") as f:
        f.write("init_cas9_virus\tinit_junk\tvirus_kill_density\n")

        # junk DNA: 0 to 300, step 10
        for junk in range(0, 301, 10):
            # cas9 = virus: 1 to <90, step 3 => 1,4,7,...,88
            for cv in range(1, 90, 3):
                kill_density = run_single_sim(junk, cv, cv)
                f.write(f"{cv}\t{junk}\t{kill_density:.6f}\n")
                # optional: progress print
                print(f"Junk={junk:3d}, Cas9/Virus={cv:2d}, kill_density={kill_density:.3f}")

    print(f"\nAll simulations complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

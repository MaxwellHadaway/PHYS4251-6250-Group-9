# PHYS4251/6250
# Group 9
# Maxwell Hadaway, Kenechukwu Aniedobe, Michael Olatunji
# LAST UPDATED: 12/4/2025
import tkinter as tk
import random
import math
import time
import matplotlib.pyplot as plt   # for histograms

# --------------------
# CONFIG
# --------------------
WIDTH, HEIGHT = 1000, 1000

# Default starting values (used as slider defaults)
NUM_JUNK_DNA = 100
NUM_VIRUS_DNA = 15
NUM_CAS9 = 15

TIME_SCALE = 0.2

DNA_RADIUS = 10
CAS9_RADIUS = 20

BG_COLOR = "black"

COLOR_DNA_JUNK = "gray70"
COLOR_DNA_VIRUS = "red"
COLOR_CAS9_FREE = "lime green"
COLOR_CAS9_BOUND_JUNK = "yellow"
COLOR_CAS9_BOUND_VIRUS = "red3"   # big red ball: Cas9 stuck to virus (failed)

DIRECTIONS = [
    (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, -1), (1, -1), (-1, 1), (1, 1)
]

UPDATE_INTERVAL_MS = 20

BIND_TIME_JUNK = 1.0            # fallback, junk uses its own sampled time
BIND_TIME_VIRUS = 2.0           # fallback, virus uses its own sampled time

COOLDOWN_JUNK = 2.5
COOLDOWN_VIRUS_FAIL = 2.5       # virus ignore time after failed check

EXPERIMENT_DURATION = 20.0
BIN_WIDTH = 5.0

COLLISION_BUFFER = 0.0

# success probability: P(immediate cut when Cas9 hits a virus)
SUCCESS_PROB = 0.8

# Scale virus dwell times so they don't exceed ~12 s
VIRUS_TIME_SCALE = 0.27   # 0.27 * 43.7 ≈ 11.8 s max for strongest bind

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

        color = COLOR_DNA_JUNK if kind == "junk" else COLOR_DNA_VIRUS
        self.id = canvas.create_oval(
            self.x - self.r, self.y - self.r,
            self.x + self.r, self.y + self.r,
            fill=color, outline=color
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
            self.x + self.r, self.y + self.r,
            fill=COLOR_CAS9_FREE, outline=COLOR_CAS9_FREE
        )

    def set_color(self):
        if self.state == "free":
            color = COLOR_CAS9_FREE
        elif self.state == "bound_junk":
            color = COLOR_CAS9_BOUND_JUNK
        elif self.state == "bound_virus":
            color = COLOR_CAS9_BOUND_VIRUS   # big red ball: failed virus check
        else:
            color = COLOR_CAS9_FREE
        self.canvas.itemconfig(self.id, fill=color, outline=color)

    def bind_to(self, dna, now, bind_time, state_name):
        self.state = state_name
        self.bound_to = dna
        self.bound_until = now + bind_time
        self.x, self.y = dna.x, dna.y
        self.update_canvas_pos()
        self.set_color()

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
                    self.set_color()

                # virus: this state now means FAILED recognition dwell
                elif self.state == "bound_virus":
                    if self.bound_to and self.bound_to.alive:
                        self.bound_to.bound = False
                        # virus cooldown was set at collision time
                    self.state = "free"
                    self.bound_to = None
                    self.set_color()

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
# MAIN
# --------------------
def main():
    root = tk.Tk()
    root.title("Cas9 Search Experiment - 20s Cutoff")

    # Main layout: canvas on the LEFT, controls on the RIGHT
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    # Canvas on the left
    canvas = tk.Canvas(main_frame, width=WIDTH, height=HEIGHT, bg=BG_COLOR)
    canvas.pack(side="left")

    # Right-hand control panel
    right_panel = tk.Frame(main_frame, bg="gray15")
    right_panel.pack(side="right", fill="y")

    # Timer at top of canvas
    timer_id = canvas.create_text(
        WIDTH // 2, 20,
        text="Time: 00:00",
        fill="white",
        font=("Helvetica", 16, "bold")
    )

    # --- Controls (sliders + button) ---
    control_frame = tk.LabelFrame(right_panel, text="Initial Populations", fg="white", bg="gray15")
    control_frame.pack(fill="x", padx=5, pady=5)

    junk_var = tk.IntVar(value=NUM_JUNK_DNA)
    virus_var = tk.IntVar(value=NUM_VIRUS_DNA)
    cas9_var = tk.IntVar(value=NUM_CAS9)

    tk.Label(control_frame, text="Junk DNA", fg="white", bg="gray15").grid(row=0, column=0, sticky="w")
    tk.Scale(
        control_frame, from_=0, to=300, orient="horizontal",
        variable=junk_var, length=200
    ).grid(row=0, column=1, padx=5, pady=2)

    tk.Label(control_frame, text="Virus DNA", fg="white", bg="gray15").grid(row=1, column=0, sticky="w")
    tk.Scale(
        control_frame, from_=0, to=100, orient="horizontal",
        variable=virus_var, length=200
    ).grid(row=1, column=1, padx=5, pady=2)

    tk.Label(control_frame, text="Cas9", fg="white", bg="gray15").grid(row=2, column=0, sticky="w")
    tk.Scale(
        control_frame, from_=0, to=100, orient="horizontal",
        variable=cas9_var, length=200
    ).grid(row=2, column=1, padx=5, pady=2)

    start_button = tk.Button(control_frame, text="Start / Restart Experiment")
    start_button.grid(row=3, column=0, columnspan=2, pady=5)

    # Speed slider
    speed_frame = tk.LabelFrame(right_panel, text="Cas9 Speed", fg="white", bg="gray15")
    speed_frame.pack(fill="x", padx=5, pady=5)

    speed_var = tk.IntVar(value=20)
    tk.Scale(
        speed_frame, from_=1, to=20, orient="horizontal",
        variable=speed_var, length=200
    ).pack(padx=5, pady=5)

    # Info label
    info_label = tk.Label(right_panel, text="", fg="white", bg="gray15", anchor="w", justify="left")
    info_label.pack(fill="x", padx=5, pady=5)

    # Legend / basic info
    legend_frame = tk.LabelFrame(right_panel, text="Legend", fg="white", bg="gray15")
    legend_frame.pack(fill="x", padx=5, pady=5)

    def legend_row(parent, color, text, row):
        swatch = tk.Canvas(parent, width=20, height=20, bg="gray15", highlightthickness=0)
        swatch.grid(row=row, column=0, padx=3, pady=2)
        swatch.create_oval(2, 2, 18, 18, fill=color, outline=color)
        tk.Label(parent, text=text, fg="white", bg="gray15", anchor="w", justify="left")\
            .grid(row=row, column=1, sticky="w")

    legend_row(legend_frame, COLOR_CAS9_FREE,
               "Green small: free Cas9", 0)
    legend_row(legend_frame, COLOR_CAS9_BOUND_JUNK,
               "Yellow small: Cas9 checking junk DNA", 1)
    legend_row(legend_frame, COLOR_CAS9_BOUND_VIRUS,
               "Red big: Cas9 bound to virus (failed)", 2)
    legend_row(legend_frame, COLOR_DNA_VIRUS,
               "Red small: virus DNA", 3)
    legend_row(legend_frame, COLOR_DNA_JUNK,
               "Gray small: junk DNA", 4)

    # Simulation state
    dna_list = []
    cas9_list = []
    capture_times = []       # virus kills
    junk_check_times = []    # junk DNA checked by Cas9
    start_time = None
    experiment_running = False

    def end_experiment():
        # histograms of virus kills and junk checks per time bin
        n_bins = int(EXPERIMENT_DURATION // BIN_WIDTH)
        counts_virus = [0] * n_bins
        counts_junk = [0] * n_bins

        for t in capture_times:
            if 0 <= t < EXPERIMENT_DURATION:
                idx = int(t // BIN_WIDTH)
                if idx < n_bins:
                    counts_virus[idx] += 1

        for t in junk_check_times:
            if 0 <= t < EXPERIMENT_DURATION:
                idx = int(t // BIN_WIDTH)
                if idx < n_bins:
                    counts_junk[idx] += 1

        labels = [f"{int(i * BIN_WIDTH)}-{int((i + 1) * BIN_WIDTH)}" for i in range(n_bins)]
        x_vals = range(n_bins)

        fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

        axs[0].bar(x_vals, counts_virus)
        axs[0].set_ylabel("Virus kills")
        axs[0].set_title("Cas9 virus kills per 5 s")

        axs[1].bar(x_vals, counts_junk)
        axs[1].set_xticks(list(x_vals))
        axs[1].set_xticklabels(labels)
        axs[1].set_xlabel("Time window (s)")
        axs[1].set_ylabel("Junk DNA checks")
        axs[1].set_title("Cas9 junk-DNA checks per 5 s")

        plt.tight_layout()
        plt.show()

    def start_experiment():
        nonlocal dna_list, cas9_list, capture_times, junk_check_times, start_time, experiment_running

        # Close old plots
        plt.close('all')

        # Clear old objects from canvas (except timer text)
        for d in dna_list:
            canvas.delete(d.id)
        for c in cas9_list:
            canvas.delete(c.id)

        # Create new population using slider values
        num_junk = junk_var.get()
        num_virus = virus_var.get()
        num_cas9 = cas9_var.get()

        dna_list = create_dna(canvas, num_junk, num_virus)
        cas9_list = create_cas9(canvas, num_cas9)

        capture_times = []
        junk_check_times = []
        start_time = time.perf_counter()
        experiment_running = True

        info_label.config(text="Experiment Running...")
        canvas.itemconfig(timer_id, text="Time: 00:00")

    # bind button to start_experiment
    start_button.config(command=start_experiment)

    def update():
        nonlocal dna_list, cas9_list, start_time, experiment_running, capture_times, junk_check_times

        # Always reschedule update loop
        root.after(UPDATE_INTERVAL_MS, update)

        if not experiment_running or start_time is None:
            return

        now = time.perf_counter()
        sim_time = now - start_time

        if sim_time >= EXPERIMENT_DURATION:
            canvas.itemconfig(timer_id, text=f"Time: {int(sim_time):02d}")
            info_label.config(text="Experiment Complete")
            experiment_running = False
            if capture_times or junk_check_times:
                end_experiment()
            return

        minutes = int(sim_time // 60)
        seconds = int(sim_time % 60)
        canvas.itemconfig(timer_id, text=f"Time: {minutes:02d}:{seconds:02d}")

        speed = speed_var.get()

        for d in dna_list:
            d.move_step(speed)
        for c in cas9_list:
            c.move_step(speed, sim_time)

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
                        # log time when junk DNA is checked
                        junk_check_times.append(sim_time)

                        bind_time = d.junk_bind_time if d.junk_bind_time is not None else BIND_TIME_JUNK
                        c.bind_to(d, sim_time, bind_time, "bound_junk")

                    else:  # virus
                        p = random.random()
                        if p <= SUCCESS_PROB:
                            # IMMEDIATE SUCCESS: no dwell, both disappear
                            capture_times.append(sim_time)
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

        dna_list = [d for d in dna_list if d.alive]
        cas9_list = [c for c in cas9_list if c.alive]

        info_label.config(
            text=f"Junk: {sum(d.kind == 'junk' for d in dna_list)}  |  "
                 f"Virus: {sum(d.kind == 'virus' for d in dna_list)}  |  "
                 f"Free Cas9: {sum(c.state == 'free' for c in cas9_list)}"
        )

    # kick off the update loop (simulation starts only when button pressed)
    update()
    root.mainloop()


if __name__ == "__main__":
    main()

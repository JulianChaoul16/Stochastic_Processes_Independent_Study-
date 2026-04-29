"""
Stochastic SIR Model – COVID-19 Spread Simulation
===================================================
Simulates epidemic spread with randomness in the infection rate β.
β is sampled from Normal(β_mean, β_std) at each time step.
γ (recovery rate) is kept constant.

Adjustable parameters are grouped at the top of the script.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#  PARAMETERS 
N          = 20000      # Total population
I0         = 10          # Initially infected individuals
R0_init    = 0           # Initially recovered

# Stochastic infection rate
BETA_MEAN  = 0.30        # Mean β  (transmissions per day per infected)
BETA_STD   = 0.05        # Std of β – controls spread / variability

# Recovery rate (constant)
GAMMA      = 0.10        # γ  (1/γ = average infectious period in days)

# Simulation settings
N_SIMS     = 200         # Number of independent Monte-Carlo runs
T_DAYS     = 160         # Simulation length (days)
DT         = 1.0         # Time step (days)

COMPARE_DETERMINISTIC = True   # Overlay deterministic SIR on the plot?


#  CORE MODEL
def run_stochastic_sir(N, I0, R0_init, beta_mean, beta_std, gamma,
                       t_days, dt, rng):
    """Run one stochastic SIR trajectory."""
    steps = int(t_days / dt)
    S = np.empty(steps); I = np.empty(steps); R = np.empty(steps)

    S[0] = N - I0 - R0_init
    I[0] = I0
    R[0] = R0_init

    for t in range(1, steps):
        # Sample β ~ N(mean, std), clip to [0, 1] to stay physical
        beta = max(0.0, rng.normal(beta_mean, beta_std))

        s, i, r = S[t-1], I[t-1], R[t-1]
        dS = -beta * s * i / N * dt
        dI = (beta * s * i / N - gamma * i) * dt
        dR = gamma * i * dt

        S[t] = max(0.0, s + dS)
        I[t] = max(0.0, i + dI)
        R[t] = max(0.0, r + dR)

    return S, I, R


def run_deterministic_sir(N, I0, R0_init, beta, gamma, t_days, dt):
    """Run the classic deterministic SIR model."""
    steps = int(t_days / dt)
    S = np.empty(steps); I = np.empty(steps); R = np.empty(steps)

    S[0] = N - I0 - R0_init
    I[0] = I0
    R[0] = R0_init

    for t in range(1, steps):
        s, i, r = S[t-1], I[t-1], R[t-1]
        dS = -beta * s * i / N * dt
        dI = (beta * s * i / N - gamma * i) * dt
        dR = gamma * i * dt
        S[t] = max(0.0, s + dS)
        I[t] = max(0.0, i + dI)
        R[t] = max(0.0, r + dR)

    return S, I, R


#  RUN SIMULATIONS


steps  = int(T_DAYS / DT)
time   = np.linspace(0, T_DAYS, steps)

rng    = np.random.default_rng(seed=42)

all_S  = np.empty((N_SIMS, steps))
all_I  = np.empty((N_SIMS, steps))
all_R  = np.empty((N_SIMS, steps))

for k in range(N_SIMS):
    S, I, R = run_stochastic_sir(N, I0, R0_init,
                                  BETA_MEAN, BETA_STD, GAMMA,
                                  T_DAYS, DT, rng)
    all_S[k] = S
    all_I[k] = I
    all_R[k] = R

# Summary statistics
mean_I = all_I.mean(axis=0)
std_I  = all_I.std(axis=0)
p5_I   = np.percentile(all_I, 5,  axis=0)
p95_I  = np.percentile(all_I, 95, axis=0)

mean_S = all_S.mean(axis=0)
mean_R = all_R.mean(axis=0)

# Deterministic reference
if COMPARE_DETERMINISTIC:
    det_S, det_I, det_R = run_deterministic_sir(
        N, I0, R0_init, BETA_MEAN, GAMMA, T_DAYS, DT)



#  FIGURE 1 – Infection curves (main plot)


plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))

# Individual trajectories (thin, transparent)
for k in range(N_SIMS):
    ax.plot(time, all_I[k] / N * 100,
            color="#e05c5c", alpha=0.06, linewidth=0.7)

# ±1 SD band
ax.fill_between(time,
                (mean_I - std_I) / N * 100,
                (mean_I + std_I) / N * 100,
                color="#e05c5c", alpha=0.25,
                label="±1 SD (stochastic)")

# 5th–95th percentile band
ax.fill_between(time, p5_I / N * 100, p95_I / N * 100,
                color="#e05c5c", alpha=0.10,
                label="5th–95th percentile")

# Mean stochastic curve
ax.plot(time, mean_I / N * 100,
        color="#c0392b", linewidth=2.5, label="Mean (stochastic)")

# Deterministic overlay
if COMPARE_DETERMINISTIC:
    ax.plot(time, det_I / N * 100,
            color="#2c3e50", linewidth=2.2, linestyle="--",
            label=f"Deterministic (β={BETA_MEAN})")

# Peak annotation
peak_day = time[np.argmax(mean_I)]
peak_pct = mean_I.max() / N * 100
ax.annotate(f"Mean peak: {peak_pct:.1f}% infected\n(day {peak_day:.0f})",
            xy=(peak_day, peak_pct),
            xytext=(peak_day + 10, peak_pct + 1.5),
            arrowprops=dict(arrowstyle="->", color="#2c3e50"),
            fontsize=9, color="#2c3e50")

ax.set_xlabel("Days", fontsize=12)
ax.set_ylabel("Infected (% of population)", fontsize=12)
ax.set_title(
    f"Stochastic SIR Model – COVID-19 Spread Simulation\n"
    f"N={N:,}  |  I₀={I0}  |  β ~ N({BETA_MEAN}, {BETA_STD})  |  "
    f"γ={GAMMA}  |  R₀≈{BETA_MEAN/GAMMA:.1f}  |  {N_SIMS} simulations",
    fontsize=12
)
ax.legend(fontsize=10, loc="upper right")
ax.set_xlim(0, T_DAYS)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("C:\\Users\\JuJuC\\OneDrive\\Desktop\\IS\\programs\\sir_infection_curves.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: sir_infection_curves.png")



#  FIGURE 2 – Full compartment overview (S, I, R)

fig2, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

compartments = [
    (all_S, mean_S, "#3498db", "Susceptible (S)"),
    (all_I, mean_I, "#e74c3c", "Infected (I)"),
    (all_R, mean_R, "#2ecc71", "Recovered (R)"),
]

det_curves = [det_S, det_I, det_R] if COMPARE_DETERMINISTIC else [None]*3

for ax2, (data, mean_curve, color, label), det in zip(axes, compartments, det_curves):
    std_curve = data.std(axis=0)

    for k in range(N_SIMS):
        ax2.plot(time, data[k] / N * 100,
                 color=color, alpha=0.04, linewidth=0.6)

    ax2.fill_between(time,
                     (mean_curve - std_curve) / N * 100,
                     (mean_curve + std_curve) / N * 100,
                     color=color, alpha=0.30)

    ax2.plot(time, mean_curve / N * 100,
             color=color, linewidth=2.4, label="Mean (stochastic)")

    if det is not None:
        ax2.plot(time, det / N * 100,
                 color="#2c3e50", linewidth=1.8, linestyle="--",
                 label="Deterministic")

    ax2.set_ylabel(f"{label}\n(% population)", fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=9, loc="upper right")

axes[-1].set_xlabel("Days", fontsize=12)
fig2.suptitle(
    f"Stochastic SIR – All Compartments  "
    f"(β ~ N({BETA_MEAN}, {BETA_STD}), γ={GAMMA}, {N_SIMS} runs)",
    fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig("C:\\Users\\JuJuC\\OneDrive\\Desktop\\IS\\programs\\sir_all_compartments.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: sir_all_compartments.png")


#  FIGURE 3 – Distribution of peak infections


peak_fractions = all_I.max(axis=1) / N * 100   # peak % per simulation

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.hist(peak_fractions, bins=30, color="#e05c5c", edgecolor="white",
         linewidth=0.6, alpha=0.85)
ax3.axvline(peak_fractions.mean(), color="#c0392b", linewidth=2.2,
            linestyle="--", label=f"Mean = {peak_fractions.mean():.1f}%")
ax3.axvline(np.median(peak_fractions), color="#8e44ad", linewidth=2.2,
            linestyle=":", label=f"Median = {np.median(peak_fractions):.1f}%")
if COMPARE_DETERMINISTIC:
    ax3.axvline(det_I.max() / N * 100, color="#2c3e50",
                linewidth=2.0, linestyle="-",
                label=f"Deterministic = {det_I.max()/N*100:.1f}%")

ax3.set_xlabel("Peak Infected (% of population)", fontsize=12)
ax3.set_ylabel("Number of simulations", fontsize=12)
ax3.set_title(
    f"Distribution of Peak Infections Across {N_SIMS} Simulations\n"
    f"β ~ N({BETA_MEAN}, {BETA_STD}),  γ={GAMMA}",
    fontsize=12
)
ax3.legend(fontsize=10)
plt.tight_layout()
plt.savefig("C:\\Users\\JuJuC\\OneDrive\\Desktop\\IS\\programs\\sir_peak_distribution.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: sir_peak_distribution.png")


#  SUMMARY STATS (printed to console)

print("\n── Simulation Summary ──────────────────────────────")
print(f"  Population N        : {N:,}")
print(f"  Initial infected I₀ : {I0}")
print(f"  β ~ N(mean={BETA_MEAN}, std={BETA_STD})")
print(f"  γ (recovery rate)   : {GAMMA}  →  avg infectious period = {1/GAMMA:.1f} days")
print(f"  Basic reprod. number R₀ ≈ β_mean/γ = {BETA_MEAN/GAMMA:.2f}")
print(f"  Simulations run     : {N_SIMS}  over {T_DAYS} days")
print(f"\n  Peak infected (mean) : {peak_fractions.mean():.2f}% ± {peak_fractions.std():.2f}%")
print(f"  Peak infected (range): {peak_fractions.min():.2f}% – {peak_fractions.max():.2f}%")
if COMPARE_DETERMINISTIC:
    print(f"  Deterministic peak  : {det_I.max()/N*100:.2f}%")
print("────────────────────────────────────────────────────")
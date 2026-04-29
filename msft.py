"""
MSFT Stock Price Modeling with Stochastic Processes
====================================================
Models implemented (progressively advanced):
  1. Random Walk
  2. Random Walk with Drift
  3. Brownian Motion (arithmetic)
  4. Geometric Brownian Motion
  5. Regime-Switching (2-state Markov chain, plain)
  6. GBM + Regime-Switching  ← Itô correction per regime
  7. GBM + Regime-Switching + LSTM  ← adaptive ML regime classifier

Data: Microsoft (MSFT) 2015-2018
Train: 2015-2017  |  Test: 2018
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. DATA ACQUISITION
# ─────────────────────────────────────────────

def load_data():
    """Try yfinance; fall back to synthetic MSFT-like data."""
    try:
        import yfinance as yf
        import pandas as pd
        df = yf.download("MSFT", start="2015-01-01", end="2019-01-01",
                         progress=False, auto_adjust=True)
        # yfinance ≥0.2 may return a multi-level column index
        if isinstance(df.columns, pd.MultiIndex):
            prices = df["Close"]["MSFT"].dropna()
        else:
            prices = df["Close"].dropna()
        print("✓ Downloaded real MSFT data via yfinance.")
        return prices
    except Exception:
        print("⚠  yfinance unavailable — using synthetic MSFT-like data.")
        return _synthetic_msft()


def _synthetic_msft():
    """Realistic MSFT-like synthetic prices (2015-01-02 → 2018-12-31)."""
    import pandas as pd

    rng = np.random.default_rng(42)
    # ~756 training days (2015-2017) + ~251 test days (2018)
    n_train = 756
    n_test  = 251
    n_total = n_train + n_test

    # Regime parameters (loosely calibrated to MSFT history)
    # Regime 0: bull  μ=0.0008, σ=0.010
    # Regime 1: bear  μ=-0.0003, σ=0.018
    regimes  = np.zeros(n_total, dtype=int)
    P = np.array([[0.97, 0.03],   # transition matrix
                  [0.05, 0.95]])
    state = 0
    for t in range(1, n_total):
        state = rng.choice(2, p=P[state])
        regimes[t] = state

    mu_r  = [0.0008, -0.0003]
    sig_r = [0.010,   0.018]

    log_rets = np.array([rng.normal(mu_r[s], sig_r[s]) for s in regimes])
    S0 = 46.0          # ~MSFT price Jan 2015
    prices_arr = S0 * np.exp(np.cumsum(log_rets))

    dates = pd.bdate_range("2015-01-02", periods=n_total)
    return pd.Series(prices_arr, index=dates, name="Close")


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def prepare_data(prices):
    import pandas as pd

    train = prices[(prices.index >= "2015-01-01") & (prices.index < "2018-01-01")]
    test  = prices[(prices.index >= "2018-01-01") & (prices.index < "2019-01-01")]

    log_ret_train = np.log(train / train.shift(1)).dropna().values
    log_ret_test  = np.log(test  / test.shift(1)).dropna().values

    return train, test, log_ret_train, log_ret_test


# ─────────────────────────────────────────────
# 3. SIMULATION HELPERS
# ─────────────────────────────────────────────

N_PATHS = 100

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def simulate_price_paths(log_return_generator, S0, n_steps, n_paths):
    """
    log_return_generator(n_steps) → array of log returns for one path.
    Returns price matrix shape (n_paths, n_steps+1).
    """
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for i in range(n_paths):
        lr = log_return_generator(n_steps)
        paths[i, 1:] = S0 * np.exp(np.cumsum(lr))
    return paths


# ─────────────────────────────────────────────
# 4. MODELS
# ─────────────────────────────────────────────

# --- 4.1 Random Walk ---
def model_random_walk(log_ret_train, S0, n_steps, rng):
    """Randomly resample past log returns (no drift assumption)."""
    sigma = np.std(log_ret_train)

    def gen(n):
        return rng.choice(log_ret_train, size=n, replace=True)

    paths = simulate_price_paths(gen, S0, n_steps, N_PATHS)
    params = {"σ": sigma}
    return paths, params


# --- 4.2 Random Walk with Drift ---
def model_rw_drift(log_ret_train, S0, n_steps, rng):
    """Resampled log returns centred on the historical mean."""
    mu    = np.mean(log_ret_train)
    sigma = np.std(log_ret_train)

    def gen(n):
        residuals = log_ret_train - mu          # de-meaned
        return mu + rng.choice(residuals, size=n, replace=True)

    paths = simulate_price_paths(gen, S0, n_steps, N_PATHS)
    params = {"μ": mu, "σ": sigma}
    return paths, params


# --- 4.3 Brownian Motion (arithmetic) ---
def model_brownian(log_ret_train, S0, n_steps, rng):
    """
    Arithmetic Brownian Motion on log-price:
        ln S(t+dt) = ln S(t) + μ·dt + σ·√dt·Z
    Here dt = 1 day (returns already daily).
    """
    mu    = np.mean(log_ret_train)
    sigma = np.std(log_ret_train)

    def gen(n):
        return rng.normal(mu, sigma, size=n)

    paths = simulate_price_paths(gen, S0, n_steps, N_PATHS)
    params = {"μ": mu, "σ": sigma}
    return paths, params


# --- 4.4 Geometric Brownian Motion ---
def model_gbm(log_ret_train, S0, n_steps, rng):
    """
    GBM exact solution:
        S(t+dt) = S(t) · exp((μ - σ²/2)·dt + σ·√dt·Z)
    """
    mu    = np.mean(log_ret_train)
    sigma = np.std(log_ret_train)
    drift = mu - 0.5 * sigma ** 2          # Itô correction

    def gen(n):
        return drift + sigma * rng.standard_normal(n)

    paths = simulate_price_paths(gen, S0, n_steps, N_PATHS)
    params = {"μ (raw)": mu, "σ": sigma, "drift (Itô)": drift}
    return paths, params


# --- 4.5 Regime-Switching ---

def _fit_hmm_em(log_rets, n_states=2, n_iter=200, rng=None, seed=0):
    """
    Simple 2-state Gaussian HMM via EM (Baum-Welch), hand-coded.
    Returns: mu_k, sig_k, transition matrix P, initial probs pi.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    T = len(log_rets)
    K = n_states

    # Initialise: split into K roughly equal chunks
    idx = np.argsort(log_rets)
    mu  = np.array([log_rets[idx[:T//2]].mean(), log_rets[idx[T//2:]].mean()])
    sig = np.array([log_rets[idx[:T//2]].std() + 1e-6,
                    log_rets[idx[T//2:]].std()  + 1e-6])
    P   = np.full((K, K), 1/K)
    pi  = np.ones(K) / K

    def gauss(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    log_lik_prev = -np.inf
    for _ in range(n_iter):
        # E-step: forward-backward
        B = np.column_stack([gauss(log_rets, mu[k], sig[k]) for k in range(K)])
        B = np.clip(B, 1e-300, None)

        # Forward
        alpha = np.zeros((T, K))
        alpha[0] = pi * B[0]
        scale    = np.zeros(T)
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ P) * B[t]
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t]

        # Backward
        beta = np.ones((T, K))
        for t in range(T-2, -1, -1):
            beta[t] = (P * B[t+1] * beta[t+1]).sum(axis=1)
            beta[t] /= beta[t].sum()

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            xi[t] = alpha[t:t+1].T * P * B[t+1] * beta[t+1]
            xi[t] /= xi[t].sum()

        log_lik = np.sum(np.log(scale))
        if abs(log_lik - log_lik_prev) < 1e-6:
            break
        log_lik_prev = log_lik

        # M-step
        pi  = gamma[0]
        P   = xi.sum(axis=0) / xi.sum(axis=0).sum(axis=1, keepdims=True)
        for k in range(K):
            w    = gamma[:, k]
            mu[k]  = np.dot(w, log_rets) / w.sum()
            sig[k] = np.sqrt(np.dot(w, (log_rets - mu[k])**2) / w.sum()) + 1e-6

    # Viterbi decode for state sequence
    log_B   = np.log(np.clip(B, 1e-300, None))
    log_P   = np.log(np.clip(P, 1e-300, None))
    log_pi  = np.log(np.clip(pi, 1e-300, None))
    viterbi = np.zeros((T, K))
    psi     = np.zeros((T, K), dtype=int)
    viterbi[0] = log_pi + log_B[0]
    for t in range(1, T):
        trans = viterbi[t-1:t].T + log_P
        psi[t] = trans.argmax(axis=0)
        viterbi[t] = trans.max(axis=0) + log_B[t]
    states = np.zeros(T, dtype=int)
    states[-1] = viterbi[-1].argmax()
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]

    # Ensure state 0 = low-vol, state 1 = high-vol
    if sig[0] > sig[1]:
        mu  = mu[[1, 0]]
        sig = sig[[1, 0]]
        P   = P[[1, 0]][:, [1, 0]]
        pi  = pi[[1, 0]]
        states = 1 - states

    return mu, sig, P, pi, states, gamma


def model_regime_switching(log_ret_train, S0, n_steps, rng):
    """2-state Markov-switching Gaussian model."""
    mu, sig, P, pi, train_states, _ = _fit_hmm_em(log_ret_train, rng=rng)

    stat_dist    = np.array([P[1,0] / (P[0,1] + P[1,0]),
                              P[0,1] / (P[0,1] + P[1,0])])
    momentum_20  = np.sum(log_ret_train[-20:])
    last_state   = int(train_states[-1])
    one_hot      = np.zeros(2); one_hot[last_state] = 1.0
    momentum_bias = np.array([-0.2, 0.2]) if momentum_20 < 0 else np.array([0.2, -0.2])
    init_dist    = np.clip(0.5*one_hot + 0.3*stat_dist + 0.2*momentum_bias, 0.01, 0.99)
    init_dist   /= init_dist.sum()

    def gen(n):
        state = int(rng.choice(2, p=init_dist))
        rets  = np.zeros(n)
        for t in range(n):
            rets[t] = rng.normal(mu[state], sig[state])
            state   = int(rng.choice(2, p=P[state]))
        return rets

    paths = simulate_price_paths(gen, S0, n_steps, N_PATHS)
    params = {
        "μ₀ (bull)": mu[0], "σ₀": sig[0],
        "μ₁ (bear)": mu[1], "σ₁": sig[1],
        "P(stay bull)": P[0, 0], "P(stay bear)": P[1, 1],
    }
    return paths, params, train_states, mu, sig, P


# ─────────────────────────────────────────────
# 4.6  GBM + Regime Switching  (Model 6)
# ─────────────────────────────────────────────
#
# Combines Geometric Brownian Motion with a 2-state Markov regime model:
#   • HMM (Baum-Welch EM) identifies bull/bear regimes from training data
#   • Each regime gets its own (μ_k, σ_k) pair
#   • At each simulation step the Markov chain transitions between regimes
#   • The GBM price equation applies the Itô correction per-regime:
#       r_t = (μ_{s_t} - σ_{s_t}²/2)·Δt  +  σ_{s_t}·√Δt·Z_t
#   • This improves on plain GBM (single global σ) and plain Regime-Switching
#     (no Itô correction) by unifying both properly
# ─────────────────────────────────────────────

def model_gbm_regime(log_ret_train, S0, n_steps, rng):
    """
    GBM + Regime-Switching:
      1. Fit 2-state HMM → per-regime (μ_k, σ_k) and transition matrix P
      2. Simulate 100 INDEPENDENT paths:
           - Each path samples its own starting regime from the stationary
             distribution π, blended with the final Viterbi state
           - GBM step with Itô correction, regime-specific params:
               r_t = (μ_{s_t} - σ_{s_t}²/2) + σ_{s_t} · Z_t
               S_t = S_{t-1} · exp(r_t)

    Key fixes vs naive implementation:
      • Each path uses its own local RNG state — paths are truly independent
      • Starting regime drawn per-path from blend of last Viterbi state and π
      • Momentum filter: recent 20-day return biases initial regime toward bear
        if trailing return is negative (carries real signal into 2018)
    """
    mu_k, sig_k, P, pi, train_states, gamma = _fit_hmm_em(log_ret_train, rng=rng)

    # Stationary distribution of the Markov chain
    stat_dist = np.array([P[1,0] / (P[0,1] + P[1,0]),
                          P[0,1] / (P[0,1] + P[1,0])])

    # Momentum signal: 20-day trailing return at end of training
    momentum_20 = np.sum(log_ret_train[-20:])
    # Blend: 50% last Viterbi state, 30% stationary dist, 20% momentum
    last_state = int(train_states[-1])
    one_hot    = np.zeros(2); one_hot[last_state] = 1.0
    momentum_bias = np.array([0.0, 0.0])
    if momentum_20 < 0:
        momentum_bias = np.array([-0.2, 0.2])   # push toward bear
    else:
        momentum_bias = np.array([0.2, -0.2])   # push toward bull
    init_dist = np.clip(0.5*one_hot + 0.3*stat_dist + 0.2*momentum_bias,
                        0.01, 0.99)
    init_dist /= init_dist.sum()

    def gen(n):
        # Each path gets an independent starting regime
        state = int(rng.choice(2, p=init_dist))
        rets  = np.zeros(n)
        for t in range(n):
            drift   = mu_k[state] - 0.5 * sig_k[state]**2
            rets[t] = drift + sig_k[state] * rng.standard_normal()
            state   = int(rng.choice(2, p=P[state]))
        return rets

    paths = simulate_price_paths(gen, S0, n_steps, N_PATHS)
    params = {
        "μ₀ bull":        mu_k[0],
        "σ₀ bull":        sig_k[0],
        "drift₀ (Itô)":   mu_k[0] - 0.5 * sig_k[0]**2,
        "μ₁ bear":        mu_k[1],
        "σ₁ bear":        sig_k[1],
        "drift₁ (Itô)":   mu_k[1] - 0.5 * sig_k[1]**2,
        "P(bull→bull)":   P[0, 0],
        "P(bear→bear)":   P[1, 1],
        "init P(bull)":   init_dist[0],
        "init P(bear)":   init_dist[1],
        "20d momentum":   momentum_20,
    }
    return paths, params, train_states, mu_k, sig_k, P


def plot_gbm_regime(model_name, train_prices, test_prices,
                    paths, train_states, mu_k, sig_k, P, out_path):
    """
    5-panel layout:
      Row 0: Training price with regime bands  |  Per-regime return distributions
      Row 1: Simulated price paths (full width)
      Row 2: Prediction error                  |  Final price distribution
    """
    from scipy import stats as sp_stats

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#f8f9fa")
    gs = GridSpec(3, 2, figure=fig,
                  left=0.07, right=0.97,
                  bottom=0.06, top=0.88,
                  hspace=0.48, wspace=0.33)

    ax_regime  = fig.add_subplot(gs[0, 0])
    ax_retdist = fig.add_subplot(gs[0, 1])
    ax_paths   = fig.add_subplot(gs[1, :])
    ax_err     = fig.add_subplot(gs[2, 0])
    ax_dist    = fig.add_subplot(gs[2, 1])

    x_tr = np.arange(len(train_prices))

    # ── Panel 1: Training price + regime background ───────────────────
    ax_regime.plot(x_tr, train_prices.values,
                   color=COLORS["actual"], linewidth=1.3, zorder=5, label="MSFT price")
    for t in range(len(train_states) - 1):
        c = COLORS["bull"] if train_states[t] == 0 else COLORS["bear"]
        ax_regime.axvspan(t, t + 1, alpha=0.22, color=c, linewidth=0)
    bull_p = mpatches.Patch(color=COLORS["bull"], alpha=0.6,
                            label=f"Bull  μ={mu_k[0]:.4f}  σ={sig_k[0]:.4f}")
    bear_p = mpatches.Patch(color=COLORS["bear"], alpha=0.6,
                            label=f"Bear  μ={mu_k[1]:.4f}  σ={sig_k[1]:.4f}")
    ax_regime.legend(handles=[bull_p, bear_p, ax_regime.lines[0]], fontsize=8)
    ax_regime.set_title("Training 2015–2017: HMM Viterbi Regimes\n"
                        "(GBM Itô drift applied per regime)",
                        fontsize=10, fontweight="bold")
    ax_regime.set_xlabel("Trading Day (2015–2017)", fontsize=9)
    ax_regime.set_ylabel("Stock Price (USD $)", fontsize=9)
    ax_regime.grid(alpha=0.3)
    ax_regime.tick_params(labelsize=8)

    # ── Panel 2: Per-regime return distributions + GBM Gaussian fit ───
    log_ret = np.log(train_prices / train_prices.shift(1)).dropna().values
    bull_rets = log_ret[train_states[:len(log_ret)] == 0]
    bear_rets = log_ret[train_states[:len(log_ret)] == 1]
    bins = np.linspace(log_ret.min(), log_ret.max(), 45)
    ax_retdist.hist(bull_rets, bins=bins, density=True, alpha=0.5,
                    color=COLORS["bull"], edgecolor="white", lw=0.3,
                    label=f"Bull returns (n={len(bull_rets)})")
    ax_retdist.hist(bear_rets, bins=bins, density=True, alpha=0.5,
                    color=COLORS["bear"], edgecolor="white", lw=0.3,
                    label=f"Bear returns (n={len(bear_rets)})")
    xr = np.linspace(log_ret.min(), log_ret.max(), 200)
    ax_retdist.plot(xr, sp_stats.norm.pdf(xr, mu_k[0] - 0.5*sig_k[0]**2, sig_k[0]),
                    color="#27ae60", lw=2.2,
                    label=f"Bull GBM drift={mu_k[0]-0.5*sig_k[0]**2:.4f}")
    ax_retdist.plot(xr, sp_stats.norm.pdf(xr, mu_k[1] - 0.5*sig_k[1]**2, sig_k[1]),
                    color="#c0392b", lw=2.2, linestyle="--",
                    label=f"Bear GBM drift={mu_k[1]-0.5*sig_k[1]**2:.4f}")
    ax_retdist.set_title("Per-Regime Return Distributions\n"
                         "with Itô-Corrected GBM Gaussian Fits",
                         fontsize=10, fontweight="bold")
    ax_retdist.set_xlabel("Daily Log-Return", fontsize=9)
    ax_retdist.set_ylabel("Probability Density", fontsize=9)
    ax_retdist.legend(fontsize=7.5)
    ax_retdist.grid(alpha=0.3)
    ax_retdist.tick_params(labelsize=8)

    # ── Panels 3-5: Standard panels ───────────────────────────────────
    rmse_val, fdiff = _plot_panels(ax_paths, ax_err, ax_dist,
                                   model_name, test_prices, paths)

    eq = MODEL_EQUATIONS.get(model_name, "")
    _add_equation_banner(fig, eq)
    fig.suptitle(
        f"MSFT 2018 Backtest — {model_name}\n"
        f"RMSE: ${rmse_val:.2f}   |   Final Δ: ${fdiff:.2f}   |   "
        f"P(bull→bull)={P[0,0]:.3f}   P(bear→bear)={P[1,1]:.3f}",
        fontsize=11, fontweight="bold", y=0.995
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return rmse_val, fdiff


# ─────────────────────────────────────────────
# 4.7  GBM + Regime Switching + LSTM  (Model 7)
# ─────────────────────────────────────────────
#
# Architecture — three-way fusion:
#
#  TRAINING:
#    Log-returns ──► HMM (Baum-Welch EM)
#                       │
#                       ▼
#               Viterbi states  s_t ∈ {0=bull, 1=bear}
#               Per-regime params: μ_k, σ_k,  transition P
#                       │
#    Features per SEQ_LEN window:       ──► PyTorch LSTM classifier
#      • log-return                             2 layers, hidden=48
#      • 5-day rolling vol                      output: P(bear)
#      • 21-day rolling vol                     loss: weighted BCE
#      • 5-day momentum
#      • HMM posterior P(bear)  ← soft label from EM
#
#  SIMULATION (2018):
#    At each step t:
#      1. LSTM reads last SEQ_LEN feature rows → P_lstm(bear)
#      2. Markov chain reads current state    → P_markov(bear)
#      3. Blend: P(bear) = 0.6·P_lstm + 0.4·P_markov
#      4. Sample regime s_t ~ Bernoulli(P_bear)
#      5. GBM step with Itô correction, regime-specific (μ_s, σ_s):
#           r_t = (μ_{s_t} - σ_{s_t}²/2) + σ_{s_t}·Z_t
#           S_t = S_{t-1}·exp(r_t)
#
# Why this beats Model 6:
#  • Pure Markov assumes constant P — LSTM adapts regime transitions
#    to current market conditions (momentum, vol level, trend)
#  • 5-channel features give the LSTM richer signal than just returns
#  • LSTM hidden state acts as a rolling "market memory" across 20 days
# ─────────────────────────────────────────────

SEQ_LEN   = 20    # LSTM lookback window (trading days)
LSTM_BLEND = 0.6  # weight on LSTM regime prob vs Markov (0=pure Markov, 1=pure LSTM)


def _build_regime_features(log_rets, gamma, seq_len=SEQ_LEN):
    """
    Build 5-channel feature matrix for the LSTM regime classifier.

    Channels per timestep:
      0: log_return                — raw daily return
      1: 5-day rolling vol         — short-term volatility level
      2: 21-day rolling vol        — medium-term volatility level
      3: 5-day momentum            — cumulative 5-day return (trend)
      4: HMM posterior P(bear)     — soft regime label from EM fit

    Returns:
      X      — shape (N, seq_len, 5)  sliding window feature tensors
      y      — shape (N,)             binary regime labels (0=bull, 1=bear)
      feats  — shape (T, 5)           full feature matrix for sim seeding
    """
    T = len(log_rets)
    feats = np.zeros((T, 5), dtype=np.float32)
    feats[:, 0] = log_rets
    for t in range(T):
        w5       = log_rets[max(0, t - 4): t + 1]
        w21      = log_rets[max(0, t - 20): t + 1]
        feats[t, 1] = w5.std()  + 1e-8
        feats[t, 2] = w21.std() + 1e-8
        feats[t, 3] = w5.sum()
        feats[t, 4] = float(gamma[t, 1])   # P(bear)

    X, y = [], []
    for i in range(T - seq_len):
        X.append(feats[i: i + seq_len])
        y.append(1.0 if gamma[i + seq_len, 1] > 0.5 else 0.0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), feats


def _train_lstm_classifier(X, y, hidden=48, layers=2, epochs=150, lr=3e-3):
    """
    Train a 2-layer LSTM binary classifier: bull (0) vs bear (1).
    Input:  X shape (N, seq_len, 5)
    Output: P(bear) per window via Sigmoid head
    Loss:   class-balanced BCE with cosine LR decay
    Falls back gracefully if PyTorch is unavailable.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None, []

    class LSTMRegimeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=5, hidden_size=hidden,
                                num_layers=layers, batch_first=True,
                                dropout=0.15)
            self.head = nn.Sequential(
                nn.Linear(hidden, 24),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(24, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    Xt = torch.tensor(X)
    yt = torch.tensor(y)

    X_mean = Xt.mean(dim=(0, 1), keepdim=True)
    X_std  = Xt.std(dim=(0, 1),  keepdim=True) + 1e-8
    Xt_n   = (Xt - X_mean) / X_std

    model   = LSTMRegimeClassifier()
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    pos_frac   = float(yt.mean().clamp(0.05, 0.95))
    pos_weight = torch.tensor([(1 - pos_frac) / pos_frac])

    history = []
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(Xt_n)
        loss = -(pos_weight * yt * torch.log(pred + 1e-8)
                 + (1 - yt) * torch.log(1 - pred + 1e-8)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        history.append(float(loss))
        if (ep + 1) % 30 == 0:
            acc = ((pred > 0.5).float() == yt).float().mean()
            print(f"    Epoch {ep+1:>3}/{epochs}  "
                  f"loss={loss.item():.4f}  acc={acc.item():.3f}")

    model.eval()
    model.X_mean = X_mean
    model.X_std  = X_std
    return model, history


def _lstm_regime_prob(model, recent_feats):
    """One LSTM forward pass → P(bear). Returns float in [0, 1]."""
    try:
        import torch
    except ImportError:
        return None
    feat = np.array(recent_feats[-SEQ_LEN:], dtype=np.float32)[np.newaxis]
    Xt   = torch.tensor(feat)
    Xt_n = (Xt - model.X_mean) / model.X_std
    with torch.no_grad():
        return float(model(Xt_n).item())


def model_gbm_regime_lstm(log_ret_train, S0, n_steps, rng):
    """
    GBM + Regime-Switching + LSTM  (Model 7)
    ------------------------------------------
    1. Fit HMM → per-regime (μ_k, σ_k, P)  and  Viterbi states + posteriors
    2. Build 5-channel sliding-window features
    3. Train LSTM classifier: P(bear | last SEQ_LEN feature rows)
    4. Simulate:
         Each step → LSTM + Markov blended regime → regime-specific GBM step
    """
    print("  Fitting 2-state HMM (Baum-Welch EM)...")
    mu_k, sig_k, P, pi, train_states, gamma = _fit_hmm_em(log_ret_train, rng=rng)

    print("  Building 5-channel feature windows...")
    X, y, all_feats = _build_regime_features(log_ret_train, gamma)

    print("  Training LSTM regime classifier (PyTorch)...")
    lstm_clf, loss_history = _train_lstm_classifier(X, y)

    pytorch_available = lstm_clf is not None
    if pytorch_available:
        print(f"  LSTM trained  ({len(loss_history)} epochs, "
              f"final loss={loss_history[-1]:.4f})")
    else:
        print("  ⚠  PyTorch not found — using pure Markov transitions.")
        print("     Install PyTorch:  pip install torch")

    # Same momentum-aware init distribution as Model 6
    stat_dist    = np.array([P[1,0] / (P[0,1] + P[1,0]),
                              P[0,1] / (P[0,1] + P[1,0])])
    momentum_20  = np.sum(log_ret_train[-20:])
    last_state   = int(train_states[-1])
    one_hot      = np.zeros(2); one_hot[last_state] = 1.0
    momentum_bias = np.array([-0.2, 0.2]) if momentum_20 < 0 else np.array([0.2, -0.2])
    init_dist    = np.clip(0.5*one_hot + 0.3*stat_dist + 0.2*momentum_bias, 0.01, 0.99)
    init_dist   /= init_dist.sum()

    seed_feats = list(all_feats[-SEQ_LEN:])

    def gen(n):
        buf_feats = list(seed_feats)
        # Each path samples its own starting regime from momentum-aware init dist
        state     = int(rng.choice(2, p=init_dist))
        rets      = np.zeros(n)

        for t in range(n):
            # ── 1. LSTM regime probability ──────────────────────────
            if pytorch_available and len(buf_feats) >= SEQ_LEN:
                p_bear_lstm = _lstm_regime_prob(lstm_clf, buf_feats)
            else:
                p_bear_lstm = float(P[state, 1])   # Markov fallback

            # ── 2. Markov transition probability ────────────────────
            p_bear_markov = float(P[state, 1])

            # ── 3. Blend LSTM + Markov ──────────────────────────────
            p_bear = float(np.clip(
                LSTM_BLEND * p_bear_lstm + (1 - LSTM_BLEND) * p_bear_markov,
                0.01, 0.99
            ))

            # ── 4. Sample regime ────────────────────────────────────
            state = int(rng.random() < p_bear)   # 1=bear, 0=bull

            # ── 5. GBM step with Itô correction (regime-specific) ───
            drift = mu_k[state] - 0.5 * sig_k[state]**2
            r     = drift + sig_k[state] * rng.standard_normal()
            rets[t] = r

            # Update feature buffer
            last_rets = np.array([f[0] for f in buf_feats[-20:]] + [r])
            buf_feats.append(np.array([
                r,
                last_rets[-5:].std()  + 1e-8,
                last_rets[-21:].std() + 1e-8,
                last_rets[-5:].sum(),
                p_bear
            ], dtype=np.float32))

        return rets

    paths = simulate_price_paths(gen, S0, n_steps, N_PATHS)
    params = {
        "μ₀ bull":        mu_k[0],
        "σ₀ bull":        sig_k[0],
        "μ₁ bear":        mu_k[1],
        "σ₁ bear":        sig_k[1],
        "P(bull→bull)":   P[0, 0],
        "P(bear→bear)":   P[1, 1],
        "LSTM blend":     LSTM_BLEND,
        "LSTM hidden":    48,
        "LSTM layers":    2,
        "LSTM epochs":    150,
        "LSTM features":  5,
        "PyTorch active": pytorch_available,
    }
    return paths, params, train_states, gamma, mu_k, sig_k, P, loss_history, pytorch_available


def plot_gbm_regime_lstm(model_name, train_prices, test_prices, paths,
                         train_states, gamma, mu_k, sig_k, P,
                         log_ret_train, params, loss_history,
                         pytorch_active, out_path):
    """
    7-panel layout:
      Row 0: Training price + regimes  |  LSTM training loss
      Row 1: HMM posteriors P(bear)    |  Per-regime return distributions
      Row 2: Simulated paths (full width)
      Row 3: Prediction error          |  Final price distribution
    """
    from scipy import stats as sp_stats

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor("#f8f9fa")
    gs = GridSpec(4, 2, figure=fig,
                  left=0.07, right=0.97,
                  bottom=0.05, top=0.89,
                  hspace=0.52, wspace=0.33)

    ax_regime  = fig.add_subplot(gs[0, 0])
    ax_loss    = fig.add_subplot(gs[0, 1])
    ax_post    = fig.add_subplot(gs[1, 0])
    ax_retdist = fig.add_subplot(gs[1, 1])
    ax_paths   = fig.add_subplot(gs[2, :])
    ax_err     = fig.add_subplot(gs[3, 0])
    ax_dist    = fig.add_subplot(gs[3, 1])

    x_tr = np.arange(len(train_prices))

    # ── Panel 1: Training price coloured by regime ────────────────────
    ax_regime.plot(x_tr, train_prices.values,
                   color=COLORS["actual"], linewidth=1.3, zorder=5, label="MSFT price")
    for t in range(len(train_states) - 1):
        c = COLORS["bull"] if train_states[t] == 0 else COLORS["bear"]
        ax_regime.axvspan(t, t + 1, alpha=0.22, color=c, linewidth=0)
    bull_p = mpatches.Patch(color=COLORS["bull"], alpha=0.6,
                            label=f"Bull  μ={mu_k[0]:.4f}  σ={sig_k[0]:.4f}")
    bear_p = mpatches.Patch(color=COLORS["bear"], alpha=0.6,
                            label=f"Bear  μ={mu_k[1]:.4f}  σ={sig_k[1]:.4f}")
    ax_regime.legend(handles=[bull_p, bear_p, ax_regime.lines[0]], fontsize=8)
    ax_regime.set_title("Training 2015–2017: HMM Viterbi Regimes\n"
                        "(LSTM learns to predict these transitions)",
                        fontsize=10, fontweight="bold")
    ax_regime.set_xlabel("Trading Day (2015–2017)", fontsize=9)
    ax_regime.set_ylabel("Stock Price (USD $)", fontsize=9)
    ax_regime.grid(alpha=0.3); ax_regime.tick_params(labelsize=8)

    # ── Panel 2: LSTM training loss ───────────────────────────────────
    if pytorch_active and loss_history:
        ax_loss.plot(loss_history, color="#3498db", lw=1.2, alpha=0.55,
                     label="BCE loss (raw)")
        smooth = pd.Series(loss_history).rolling(15, min_periods=1).mean().values
        ax_loss.plot(smooth, color="#e74c3c", lw=2.2, linestyle="--",
                     label="15-epoch MA")
        ax_loss.set_title("LSTM Classifier Training Loss\n"
                          "(Binary Cross-Entropy, class-balanced, cosine LR)",
                          fontsize=10, fontweight="bold")
        ax_loss.set_xlabel("Epoch", fontsize=9)
        ax_loss.set_ylabel("Weighted BCE Loss", fontsize=9)
        ax_loss.legend(fontsize=8)
        ax_loss.annotate(
            f"Final: {loss_history[-1]:.4f}",
            xy=(len(loss_history) - 1, loss_history[-1]),
            xytext=(len(loss_history) * 0.55, max(loss_history) * 0.82),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
            fontsize=9, color="#e74c3c", fontweight="bold"
        )
    else:
        ax_loss.set_facecolor("#ffeaa7")
        ax_loss.text(0.5, 0.5,
                     "PyTorch not installed.\n\n"
                     "LSTM regime classifier will activate\n"
                     "automatically once installed:\n\n"
                     "  pip install torch\n\n"
                     "Fallback: pure Markov transitions.",
                     ha="center", va="center", fontsize=11,
                     transform=ax_loss.transAxes,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax_loss.set_title("LSTM Training Loss\n(PyTorch required)", fontsize=10, fontweight="bold")
        ax_loss.set_xlabel("Epoch", fontsize=9)
        ax_loss.set_ylabel("BCE Loss", fontsize=9)
    ax_loss.grid(alpha=0.3); ax_loss.tick_params(labelsize=8)

    # ── Panel 3: HMM posterior P(bear) ────────────────────────────────
    x_post = np.arange(len(gamma))
    ax_post.fill_between(x_post, gamma[:, 1], alpha=0.45,
                         color=COLORS["bear"], label="P(bear | data)")
    ax_post.fill_between(x_post, gamma[:, 0], alpha=0.25,
                         color=COLORS["bull"], label="P(bull | data)")
    ax_post.axhline(0.5, color="k", lw=1.0, linestyle="--", alpha=0.4,
                    label="Decision boundary")
    ax_post.set_title("HMM Posterior Probabilities\n"
                      "(LSTM training targets — channel 4 of feature vector)",
                      fontsize=10, fontweight="bold")
    ax_post.set_xlabel("Trading Day (2015–2017)", fontsize=9)
    ax_post.set_ylabel("Posterior Probability", fontsize=9)
    ax_post.set_ylim(0, 1)
    ax_post.legend(fontsize=8); ax_post.grid(alpha=0.3); ax_post.tick_params(labelsize=8)

    # ── Panel 4: Per-regime return distributions ──────────────────────
    log_ret = np.log(train_prices / train_prices.shift(1)).dropna().values
    n_states = len(log_ret)
    bull_rets = log_ret[train_states[:n_states] == 0]
    bear_rets = log_ret[train_states[:n_states] == 1]
    bins = np.linspace(log_ret.min(), log_ret.max(), 45)
    ax_retdist.hist(bull_rets, bins=bins, density=True, alpha=0.5,
                    color=COLORS["bull"], edgecolor="white", lw=0.3,
                    label=f"Bull (n={len(bull_rets)})")
    ax_retdist.hist(bear_rets, bins=bins, density=True, alpha=0.5,
                    color=COLORS["bear"], edgecolor="white", lw=0.3,
                    label=f"Bear (n={len(bear_rets)})")
    xr = np.linspace(log_ret.min(), log_ret.max(), 200)
    d0 = mu_k[0] - 0.5 * sig_k[0]**2
    d1 = mu_k[1] - 0.5 * sig_k[1]**2
    ax_retdist.plot(xr, sp_stats.norm.pdf(xr, d0, sig_k[0]),
                    color="#27ae60", lw=2.2,
                    label=f"Bull GBM N(drift={d0:.4f}, σ={sig_k[0]:.4f})")
    ax_retdist.plot(xr, sp_stats.norm.pdf(xr, d1, sig_k[1]),
                    color="#c0392b", lw=2.2, linestyle="--",
                    label=f"Bear GBM N(drift={d1:.4f}, σ={sig_k[1]:.4f})")
    ax_retdist.set_title("Per-Regime Return Distributions\n"
                         "with Itô-Corrected GBM Gaussian Fits",
                         fontsize=10, fontweight="bold")
    ax_retdist.set_xlabel("Daily Log-Return", fontsize=9)
    ax_retdist.set_ylabel("Probability Density", fontsize=9)
    ax_retdist.legend(fontsize=7.5); ax_retdist.grid(alpha=0.3); ax_retdist.tick_params(labelsize=8)

    # ── Panels 5-7: Standard price / error / distribution ─────────────
    rmse_val, fdiff = _plot_panels(ax_paths, ax_err, ax_dist,
                                   model_name, test_prices, paths)

    eq = MODEL_EQUATIONS.get(model_name, "")
    _add_equation_banner(fig, eq)

    pt_str = "✓ PyTorch LSTM active" if pytorch_active else "⚠ Markov fallback (install PyTorch)"
    fig.suptitle(
        f"MSFT 2018 Backtest — {model_name}\n"
        f"RMSE: ${rmse_val:.2f}   |   Final Δ: ${fdiff:.2f}   |   "
        f"Bull: μ={mu_k[0]:.4f} σ={sig_k[0]:.4f}   |   "
        f"Bear: μ={mu_k[1]:.4f} σ={sig_k[1]:.4f}   |   {pt_str}",
        fontsize=11, fontweight="bold", y=0.995
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return rmse_val, fdiff

# ─────────────────────────────────────────────
# 5. PLOTTING
# ─────────────────────────────────────────────

COLORS = {
    "actual":  "#1a1a2e",
    "paths":   "#a8d8ea",
    "avg":     "#e94560",
    "drift":   "#f39c12",
    "bull":    "#2ecc71",
    "bear":    "#e74c3c",
    "eq_bg":   "#2c3e50",
    "eq_fg":   "#ecf0f1",
}

# Equations displayed at the top of each model figure
MODEL_EQUATIONS = {
    "Random Walk":
        r"$S_t = S_{t-1} \cdot e^{\epsilon_t}$"
        "          "
        r"$\epsilon_t \sim \mathrm{Resample}(\log\,\mathrm{returns}_{train})$",

    "Random Walk with Drift":
        r"$S_t = S_{t-1} \cdot e^{\mu + \tilde{\epsilon}_t}$"
        "     "
        r"$\mu = \overline{\Delta \log S},\quad \tilde{\epsilon}_t = \epsilon_t - \mu$",

    "Brownian Motion":
        r"$\ln S_t = \ln S_{t-1} + \mu \Delta t + \sigma \sqrt{\Delta t} \cdot Z_t$"
        "     "
        r"$Z_t \sim \mathcal{N}(0,1)$",

    "Geometric Brownian Motion":
        r"$S_t = S_{t-1} \cdot \exp[(\mu - \sigma^2/2)\Delta t + \sigma \sqrt{\Delta t} \cdot Z_t]$"
        "     "
        r"$Z_t \sim \mathcal{N}(0,1)$   (Ito correction: $-\sigma^2/2$)",

    "Regime-Switching (HMM)":
        r"$r_t \sim \mathcal{N}(\mu_{s_t},\, \sigma_{s_t}^2)$"
        "     "
        r"$P(s_t=j \mid s_{t-1}=i) = p_{ij},\quad s_t \in \{0_{bull},\, 1_{bear}\}$",

    "GBM + Regime-Switching":
        r"$r_t = (\mu_{s_t} - \sigma_{s_t}^2/2) + \sigma_{s_t} Z_t$"
        "     "
        r"$s_t \sim \mathrm{Markov}(P),\quad Z_t \sim \mathcal{N}(0,1)$   (Ito per regime)",

    "GBM + Regime + LSTM":
        r"$P(bear_t) = 0.6 \cdot \mathrm{LSTM}(x_{t-L:t}) + 0.4 \cdot P_{markov}$"
        "     "
        r"$r_t = (\mu_{s_t} - \sigma_{s_t}^2/2) + \sigma_{s_t} Z_t,\quad s_t \sim \mathrm{Bernoulli}(P_{bear_t})$",
}


def _add_equation_banner(fig, equation_text):
    """Add a shaded equation box just below the figure title."""
    ax_eq = fig.add_axes([0.05, 0.93, 0.90, 0.055])
    ax_eq.set_facecolor(COLORS["eq_bg"])
    ax_eq.set_xlim(0, 1); ax_eq.set_ylim(0, 1)
    ax_eq.axis("off")
    ax_eq.text(0.5, 0.5, equation_text,
               ha="center", va="center", fontsize=11,
               color=COLORS["eq_fg"], fontfamily="monospace",
               transform=ax_eq.transAxes)


def _plot_panels(ax_paths, ax_err, ax_dist,
                 model_name, test_prices, paths):
    """
    Draw the three standard panels (price paths, error, final distribution).
    Returns (rmse_val, final_diff).
    """
    n_pts = len(test_prices)
    x     = np.arange(n_pts)
    avg   = paths[:, :n_pts].mean(axis=0)

    # ── (a) Price paths ──────────────────────────────────────────────
    for i in range(N_PATHS):
        ax_paths.plot(x, paths[i, :n_pts],
                      color=COLORS["paths"], alpha=0.15, linewidth=0.6)
    ax_paths.plot(x, avg,                color=COLORS["avg"],    linewidth=2.2, label="Avg simulation", zorder=4)
    ax_paths.plot(x, test_prices.values, color=COLORS["actual"], linewidth=2.2, label="Actual MSFT",    zorder=5)
    ax_paths.set_title(f"{model_name}: 100 Simulated Price Paths vs Actual (2018)",
                       fontsize=11, fontweight="bold", pad=6)
    ax_paths.set_xlabel("Trading Day (2018)", fontsize=9)
    ax_paths.set_ylabel("Stock Price (USD $)", fontsize=9)
    ax_paths.legend(fontsize=9, loc="upper left")
    ax_paths.grid(alpha=0.3)
    ax_paths.tick_params(labelsize=8)

    # ── (b) Prediction error over time ───────────────────────────────
    err = avg - test_prices.values
    ax_err.fill_between(x, err, 0, where=(err > 0),
                        color=COLORS["bear"], alpha=0.55, label="Over-prediction")
    ax_err.fill_between(x, err, 0, where=(err <= 0),
                        color=COLORS["bull"], alpha=0.55, label="Under-prediction")
    ax_err.plot(x, err, color="#555", linewidth=0.8, alpha=0.7)
    ax_err.axhline(0, color="k", linewidth=1.0, linestyle="--")
    ax_err.set_title("Prediction Error Over Time\n(Avg Simulation − Actual)", fontsize=10, fontweight="bold")
    ax_err.set_xlabel("Trading Day (2018)", fontsize=9)
    ax_err.set_ylabel("Error (USD $)", fontsize=9)
    ax_err.legend(fontsize=8)
    ax_err.grid(alpha=0.3)
    ax_err.tick_params(labelsize=8)

    # ── (c) Distribution of final simulated prices ────────────────────
    final_sim = paths[:, n_pts - 1]
    ax_dist.hist(final_sim, bins=25, color=COLORS["paths"],
                 edgecolor="white", linewidth=0.5, label="Simulated final prices")
    ax_dist.axvline(float(test_prices.iloc[-1]), color=COLORS["actual"],
                    linewidth=2.2, label=f"Actual end: ${float(test_prices.iloc[-1]):.2f}")
    ax_dist.axvline(final_sim.mean(), color=COLORS["avg"],
                    linewidth=2.2, linestyle="--",
                    label=f"Avg simulated: ${final_sim.mean():.2f}")
    ax_dist.set_title("Distribution of Final Simulated Prices\n(End of 2018)", fontsize=10, fontweight="bold")
    ax_dist.set_xlabel("Stock Price (USD $)", fontsize=9)
    ax_dist.set_ylabel("Number of Paths", fontsize=9)
    ax_dist.legend(fontsize=8)
    ax_dist.grid(alpha=0.3)
    ax_dist.tick_params(labelsize=8)

    rmse_val   = rmse(test_prices.values, avg)
    final_diff = abs(final_sim.mean() - float(test_prices.iloc[-1]))
    return rmse_val, final_diff


def plot_standard_model(model_name, test_prices, paths, out_path, params=None):
    """One figure: equation banner + 3 panels."""
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor("#f8f9fa")

    # Leave room at top for equation banner (0–8 % of fig height) + title
    gs = GridSpec(2, 2, figure=fig,
                  left=0.07, right=0.97,
                  bottom=0.07, top=0.88,
                  hspace=0.42, wspace=0.33)

    ax_paths = fig.add_subplot(gs[0, :])
    ax_err   = fig.add_subplot(gs[1, 0])
    ax_dist  = fig.add_subplot(gs[1, 1])

    rmse_val, fdiff = _plot_panels(ax_paths, ax_err, ax_dist,
                                   model_name, test_prices, paths)

    eq = MODEL_EQUATIONS.get(model_name, "")
    _add_equation_banner(fig, eq)

    fig.suptitle(
        f"MSFT 2018 Backtest — {model_name}"
        f"   |   RMSE: ${rmse_val:.2f}   |   Final price Δ: ${fdiff:.2f}",
        fontsize=12, fontweight="bold", y=0.995
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return rmse_val, fdiff


def plot_rw_drift(model_name, test_prices, paths, params, out_path):
    """
    Model 2 — special version that adds a 4th panel:
    the cumulative drift line vs actual price, highlighting
    how the drift component alone shapes the forecast.
    """
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#f8f9fa")

    gs = GridSpec(2, 3, figure=fig,
                  left=0.06, right=0.97,
                  bottom=0.07, top=0.88,
                  hspace=0.42, wspace=0.35)

    ax_paths = fig.add_subplot(gs[0, :])          # full-width top row
    ax_err   = fig.add_subplot(gs[1, 0])
    ax_dist  = fig.add_subplot(gs[1, 1])
    ax_drift = fig.add_subplot(gs[1, 2])           # new drift panel

    rmse_val, fdiff = _plot_panels(ax_paths, ax_err, ax_dist,
                                   model_name, test_prices, paths)

    # ── Drift panel ──────────────────────────────────────────────────
    mu    = params["μ"]
    S0    = float(test_prices.iloc[0])
    n_pts = len(test_prices)
    x     = np.arange(n_pts)

    # Pure deterministic drift path  S0 * exp(μ * t)
    drift_path = S0 * np.exp(mu * x)

    # Average simulated path
    avg = paths[:, :n_pts].mean(axis=0)

    ax_drift.plot(x, test_prices.values, color=COLORS["actual"],
                  linewidth=2.2, label="Actual MSFT", zorder=5)
    ax_drift.plot(x, avg,               color=COLORS["avg"],
                  linewidth=2.0, linestyle="-", label="Avg simulation", zorder=4)
    ax_drift.plot(x, drift_path,        color=COLORS["drift"],
                  linewidth=2.2, linestyle="--", label=f"Pure drift  μ={mu:.5f}/day", zorder=3)

    # Shade the gap between drift and actual to show drift direction
    ax_drift.fill_between(x, drift_path, test_prices.values,
                          where=(drift_path > test_prices.values),
                          color=COLORS["bear"], alpha=0.25, label="Drift above actual")
    ax_drift.fill_between(x, drift_path, test_prices.values,
                          where=(drift_path <= test_prices.values),
                          color=COLORS["bull"], alpha=0.25, label="Drift below actual")

    # Arrow annotation showing drift direction on the plot
    mid = n_pts // 2
    ax_drift.annotate(
        f"Drift direction\n(μ = {mu:.5f})",
        xy=(mid, drift_path[mid]),
        xytext=(mid + 15, drift_path[mid] + (1 if mu >= 0 else -1) * S0 * 0.06),
        arrowprops=dict(arrowstyle="->", color=COLORS["drift"], lw=1.8),
        color=COLORS["drift"], fontsize=8, fontweight="bold"
    )

    ax_drift.set_title("Drift Component Highlighted\n"
                       r"$S_0\,e^{\,\mu t}$  vs Actual", fontsize=10, fontweight="bold")
    ax_drift.set_xlabel("Trading Day (2018)", fontsize=9)
    ax_drift.set_ylabel("Stock Price (USD $)", fontsize=9)
    ax_drift.legend(fontsize=7.5, loc="best")
    ax_drift.grid(alpha=0.3)
    ax_drift.tick_params(labelsize=8)

    # ── Equation banner & title ───────────────────────────────────────
    eq = MODEL_EQUATIONS.get(model_name, "")
    _add_equation_banner(fig, eq)

    fig.suptitle(
        f"MSFT 2018 Backtest — {model_name}"
        f"   |   RMSE: ${rmse_val:.2f}   |   Final price Δ: ${fdiff:.2f}",
        fontsize=12, fontweight="bold", y=0.995
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return rmse_val, fdiff


def plot_regime_switching(model_name, train_prices, test_prices,
                          paths, train_states, mu, sig, P, out_path):
    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor("#f8f9fa")

    gs = GridSpec(3, 2, figure=fig,
                  left=0.07, right=0.97,
                  bottom=0.06, top=0.88,
                  hspace=0.48, wspace=0.33)

    # ── (1) Training regime coloring ─────────────────────────────────
    ax_regime = fig.add_subplot(gs[0, :])
    x_tr = np.arange(len(train_prices))
    ax_regime.plot(x_tr, train_prices.values,
                   color=COLORS["actual"], linewidth=1.3, zorder=5, label="MSFT price")
    for t in range(len(train_states) - 1):
        color = COLORS["bull"] if train_states[t] == 0 else COLORS["bear"]
        ax_regime.axvspan(t, t + 1, alpha=0.22, color=color, linewidth=0)
    bull_p = mpatches.Patch(color=COLORS["bull"], alpha=0.6, label="Regime 0 — Bull (low vol)")
    bear_p = mpatches.Patch(color=COLORS["bear"], alpha=0.6, label="Regime 1 — Bear (high vol)")
    ax_regime.legend(handles=[bull_p, bear_p, ax_regime.lines[0]], fontsize=9)
    ax_regime.set_title("Training Period (2015–2017): Viterbi-Decoded Regimes",
                        fontsize=11, fontweight="bold")
    ax_regime.set_xlabel("Trading Day (2015–2017)", fontsize=9)
    ax_regime.set_ylabel("Stock Price (USD $)", fontsize=9)
    ax_regime.grid(alpha=0.3)
    ax_regime.tick_params(labelsize=8)

    # ── (2-4) Standard panels for 2018 test ──────────────────────────
    ax_paths = fig.add_subplot(gs[1, :])
    ax_err   = fig.add_subplot(gs[2, 0])
    ax_dist  = fig.add_subplot(gs[2, 1])

    rmse_val, fdiff = _plot_panels(ax_paths, ax_err, ax_dist,
                                   model_name, test_prices, paths)

    # ── Equation banner & title ───────────────────────────────────────
    eq = MODEL_EQUATIONS.get(model_name, "")
    _add_equation_banner(fig, eq)

    regime_str = (
        f"Bull: μ={mu[0]:.4f}, σ={sig[0]:.4f}  |  "
        f"Bear: μ={mu[1]:.4f}, σ={sig[1]:.4f}  |  "
        f"P(bull→bull)={P[0,0]:.3f}, P(bear→bear)={P[1,1]:.3f}"
    )
    fig.suptitle(
        f"MSFT 2018 Backtest — {model_name}\n"
        f"RMSE: ${rmse_val:.2f}   |   Final price Δ: ${fdiff:.2f}   |   {regime_str}",
        fontsize=11, fontweight="bold", y=0.995
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return rmse_val, fdiff


def plot_comparison(results, out_path):
    """Bar chart comparing RMSE and final-price error across all models."""
    models    = list(results.keys())
    rmse_vals  = [results[m]["rmse"]       for m in models]
    fdiff_vals = [results[m]["final_diff"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#f8f9fa")

    palette = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]

    # RMSE
    bars1 = ax1.bar(models, rmse_vals, color=palette, edgecolor="white", linewidth=1.5, width=0.55)
    ax1.set_title("RMSE by Model  (lower = better)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Model", fontsize=10)
    ax1.set_ylabel("RMSE (USD $)", fontsize=10)
    ax1.set_ylim(0, max(rmse_vals) * 1.30)
    for bar, val in zip(bars1, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"${val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.tick_params(axis="x", rotation=15, labelsize=9)
    ax1.grid(axis="y", alpha=0.35)

    # Final price diff
    bars2 = ax2.bar(models, fdiff_vals, color=palette, edgecolor="white", linewidth=1.5, width=0.55)
    ax2.set_title("|Final Price Difference| by Model  (lower = better)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Model", fontsize=10)
    ax2.set_ylabel("Absolute Final Price Difference (USD $)", fontsize=10)
    ax2.set_ylim(0, max(fdiff_vals) * 1.30)
    for bar, val in zip(bars2, fdiff_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"${val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.tick_params(axis="x", rotation=15, labelsize=9)
    ax2.grid(axis="y", alpha=0.35)

    fig.suptitle("Model Performance Comparison — MSFT 2018 Backtesting",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    import os
    out_dir = r"C:\Users\JuJuC\Onedrive\Desktop\IS\programs\final"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(99)

    # -- Load & split --
    print("\n━━━ Loading data ━━━")
    prices = load_data()
    train, test, log_ret_train, log_ret_test = prepare_data(prices)
    S0 = float(test.iloc[0])
    n_steps = len(test) - 1

    print(f"Training samples : {len(train)} days  (first price: ${float(train.iloc[0]):.2f})")
    print(f"Test samples     : {len(test)} days  (first price: ${float(test.iloc[0]):.2f})")

    results = {}

    # ─── Model 1: Random Walk ───
    print("\n━━━ Model 1: Random Walk ━━━")
    paths_rw, params_rw = model_random_walk(log_ret_train, S0, n_steps, rng)
    rmse_rw, fdiff_rw = plot_standard_model(
        "Random Walk", test, paths_rw,
        f"{out_dir}/model1_random_walk.png", params=params_rw
    )
    results["Random Walk"] = {"rmse": rmse_rw, "final_diff": fdiff_rw, "params": params_rw}
    print(f"  Params : {params_rw}")
    print(f"  RMSE   : ${rmse_rw:.2f}  |  Final diff: ${fdiff_rw:.2f}")

    # ─── Model 2: RW + Drift ───
    print("\n━━━ Model 2: Random Walk with Drift ━━━")
    paths_rwd, params_rwd = model_rw_drift(log_ret_train, S0, n_steps, rng)
    rmse_rwd, fdiff_rwd = plot_rw_drift(
        "Random Walk with Drift", test, paths_rwd, params_rwd,
        f"{out_dir}/model2_rw_drift.png"
    )
    results["RW + Drift"] = {"rmse": rmse_rwd, "final_diff": fdiff_rwd, "params": params_rwd}
    print(f"  Params : {params_rwd}")
    print(f"  RMSE   : ${rmse_rwd:.2f}  |  Final diff: ${fdiff_rwd:.2f}")

    # ─── Model 3: Brownian Motion ───
    print("\n━━━ Model 3: Brownian Motion ━━━")
    paths_bm, params_bm = model_brownian(log_ret_train, S0, n_steps, rng)
    rmse_bm, fdiff_bm = plot_standard_model(
        "Brownian Motion", test, paths_bm,
        f"{out_dir}/model3_brownian.png", params=params_bm
    )
    results["Brownian Motion"] = {"rmse": rmse_bm, "final_diff": fdiff_bm, "params": params_bm}
    print(f"  Params : {params_bm}")
    print(f"  RMSE   : ${rmse_bm:.2f}  |  Final diff: ${fdiff_bm:.2f}")

    # ─── Model 4: GBM ───
    print("\n━━━ Model 4: Geometric Brownian Motion ━━━")
    paths_gbm, params_gbm = model_gbm(log_ret_train, S0, n_steps, rng)
    rmse_gbm, fdiff_gbm = plot_standard_model(
        "Geometric Brownian Motion", test, paths_gbm,
        f"{out_dir}/model4_gbm.png", params=params_gbm
    )
    results["GBM"] = {"rmse": rmse_gbm, "final_diff": fdiff_gbm, "params": params_gbm}
    print(f"  Params : {params_gbm}")
    print(f"  RMSE   : ${rmse_gbm:.2f}  |  Final diff: ${fdiff_gbm:.2f}")

    # ─── Model 5: Regime-Switching ───
    print("\n━━━ Model 5: Regime-Switching HMM ━━━")
    paths_rs, params_rs, train_states, mu_rs, sig_rs, P_rs = \
        model_regime_switching(log_ret_train, S0, n_steps, rng)
    rmse_rs, fdiff_rs = plot_regime_switching(
        "Regime-Switching (HMM)", train, test, paths_rs,
        train_states, mu_rs, sig_rs, P_rs,
        f"{out_dir}/model5_regime_switching.png"
    )
    results["Regime-Switch"] = {"rmse": rmse_rs, "final_diff": fdiff_rs, "params": params_rs}
    print(f"  Params : {params_rs}")
    print(f"  RMSE   : ${rmse_rs:.2f}  |  Final diff: ${fdiff_rs:.2f}")

    # ─── Model 6: GBM + Regime-Switching ───
    print("\n━━━ Model 6: GBM + Regime-Switching ━━━")
    paths_gbmr, params_gbmr, train_states_6, mu_6, sig_6, P_6 = \
        model_gbm_regime(log_ret_train, S0, n_steps, rng)
    rmse_gbmr, fdiff_gbmr = plot_gbm_regime(
        "GBM + Regime-Switching", train, test,
        paths_gbmr, train_states_6, mu_6, sig_6, P_6,
        f"{out_dir}/model6_gbm_regime.png"
    )
    results["GBM + Regime"] = {"rmse": rmse_gbmr, "final_diff": fdiff_gbmr, "params": params_gbmr}
    print(f"  Params : { {k: round(float(v), 6) for k, v in params_gbmr.items()} }")
    print(f"  RMSE   : ${rmse_gbmr:.2f}  |  Final diff: ${fdiff_gbmr:.2f}")

    # ─── Model 7: GBM + Regime-Switching + LSTM ───
    print("\n━━━ Model 7: GBM + Regime-Switching + LSTM (PyTorch) ━━━")
    (paths_grl, params_grl, train_states_7, gamma_7,
     mu_7, sig_7, P_7, loss_hist_7, pt_active_7) = \
        model_gbm_regime_lstm(log_ret_train, S0, n_steps, rng)
    rmse_grl, fdiff_grl = plot_gbm_regime_lstm(
        "GBM + Regime + LSTM", train, test,
        paths_grl, train_states_7, gamma_7,
        mu_7, sig_7, P_7, log_ret_train,
        params_grl, loss_hist_7, pt_active_7,
        f"{out_dir}/model7_gbm_regime_lstm.png"
    )
    results["GBM + Regime + LSTM"] = {"rmse": rmse_grl, "final_diff": fdiff_grl, "params": params_grl}
    print(f"  Params : { {k: (round(float(v), 6) if isinstance(v, (float, int)) and not isinstance(v, bool) else v) for k, v in params_grl.items()} }")
    print(f"  RMSE   : ${rmse_grl:.2f}  |  Final diff: ${fdiff_grl:.2f}")

    # ─── Comparison ───
    print("\n━━━ Generating comparison chart ━━━")
    plot_comparison(results, f"{out_dir}/model_comparison.png")

    # ─── Summary table ───
    print("\n" + "═"*60)
    print("  MODEL PERFORMANCE SUMMARY (MSFT 2018 Backtest)")
    print("═"*60)
    print(f"  {'Model':<24}  {'RMSE ($)':>10}  {'|Final Δ| ($)':>14}")
    print("─"*60)
    sorted_models = sorted(results.items(), key=lambda x: x[1]["rmse"])
    for name, res in sorted_models:
        print(f"  {name:<24}  {res['rmse']:>10.2f}  {res['final_diff']:>14.2f}")
    best = sorted_models[0][0]
    print("─"*60)
    print(f"  Best by RMSE: {best}")
    print("═"*60)

    print(f"\n✓ All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
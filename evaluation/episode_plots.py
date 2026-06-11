
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tanh model & fitting
# ---------------------------------------------------------------------------

def tanh_model(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """y(x) = d + b * tanh(a*x - c)"""
    return d + b * np.tanh(a * x - c)


def fit_tanh_curve(
    x: np.ndarray,
    y: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Fit y(x) = d + b * tanh(a*x - c) using nonlinear least squares.

    Returns (a, b, c, d) or None if fitting fails.
    Requires at least 4 data points.
    """
    if not SCIPY_AVAILABLE:
        return None
    if len(x) < 4:
        return None

    # Derive initial guesses from data
    y_range = float(np.ptp(y)) if np.ptp(y) > 1e-10 else 1e-4
    b0 = y_range / 2.0
    d0 = float(np.median(y))
    a0 = 1.0
    c0 = float(np.median(x))

    try:
        popt, _ = curve_fit(
            tanh_model,
            x,
            y,
            p0=[a0, b0, c0, d0],
            maxfev=5000,
        )
        return tuple(float(v) for v in popt)   # (a, b, c, d)
    except Exception:
        return None


def compute_burstiness(a: float, b: float, all_a: List[float], all_b: List[float]) -> float:
    """
    tau = |a_scaled| * b_scaled
    where scaling is min-max normalisation over all fitted negotiations.
    """
    if len(all_a) < 2 or len(all_b) < 2:
        return 0.0

    a_min, a_max = min(all_a), max(all_a)
    b_min, b_max = min(all_b), max(all_b)

    def _scale(v, lo, hi):
        denom = max(1e-8, hi - lo)
        return (abs(v) - lo) / denom

    a_scaled = _scale(a, a_min, a_max)
    b_scaled = _scale(b, b_min, b_max)
    return float(a_scaled * b_scaled)


def compute_cri(a: float, T: int) -> float:
    """
    CRI = 1 - 1.32 / (|a| * T)
    Clamped to [0, 1].
    """
    denom = abs(a) * max(T, 1)
    if denom < 1e-10:
        return 1.0
    return float(np.clip(1.0 - 1.32 / denom, 0.0, 1.0))


def compute_cri_star(
    a: float,
    b: float,
    T: int,
    theta: float = 0.1,
) -> float:
    """
    Data-driven CRI*.

    1. Instantaneous speed s(x) = |a*b| * (1 - tanh²(a*x - c))
       (c cancels in the normalised-speed profile because peak speed is |a*b|)
    2. Normalise: s_hat(x) = s(x) / max s(x) in [0, T]
    3. Active window W = {x : s_hat(x) >= theta}
    4. CRI* = 1 - len(W) / T

    For a discrete approximation we sample at T integer points.
    """
    if T < 2:
        return 1.0

    xs = np.linspace(0, T, T)
    # Speed proportional: |ab| * sech²(ax - c)  -> max at x = c/a
    # sech² at max = 1, so normalised speed = sech²(ax - c)
    speed = 1.0 / np.cosh(a * xs) ** 2   # normalised (c=0 for profile shape)
    max_s = float(speed.max())
    if max_s < 1e-10:
        return 1.0
    speed_hat = speed / max_s

    active = int((speed_hat >= theta).sum())
    return float(np.clip(1.0 - active / T, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Extract offer sequences from episode payload
# ---------------------------------------------------------------------------

def _extract_offer_sequence(episode_payload: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """
    Returns (propose_utils, respond_utils) in episode order,
    using the 'utility' field recorded in each step.
    """
    propose_utils, respond_utils = [], []
    for step in episode_payload.get("steps", []):
        u = step.get("utility")
        if u is None:
            continue
        if step.get("kind") == "propose":
            propose_utils.append(float(u))
        elif step.get("kind") == "respond":
            respond_utils.append(float(u))
    return propose_utils, respond_utils


def _extract_all_utils_in_order(episode_payload: Dict[str, Any]) -> List[float]:
    """Return all utility values in step order (propose + respond interleaved)."""
    utils = []
    for step in episode_payload.get("steps", []):
        u = step.get("utility")
        if u is not None:
            utils.append(float(u))
    return utils


# ---------------------------------------------------------------------------
# Per-episode tanh metrics
# ---------------------------------------------------------------------------

def compute_per_episode_tanh_metrics(
    episode_payload: Dict[str, Any],
    all_params: Optional[List[Tuple[float, float, float, float]]] = None,
) -> Dict[str, Any]:
    """
    Compute tanh-based concession metrics for a SINGLE episode.

    Parameters
    ----------
    episode_payload : dict
        Episode payload from RLNegotiator.build_episode_payload()
    all_params : optional list of (a, b, c, d) from all episodes
        Needed for proper burstiness scaling. If None, scaling uses this
        episode only (burstiness will always equal 0).

    Returns
    -------
    dict with keys:
        tanh_params: (a, b, c, d) or None
        burstiness: float
        cri: float
        cri_star: float
        anchor_distance: float  (first offer utility)
        propose_utils: list[float]
        fitted_curve: list[float] or None
        n_steps: int
    """
    propose_utils, _ = _extract_offer_sequence(episode_payload)

    result: Dict[str, Any] = {
        "tanh_params": None,
        "burstiness": 0.0,
        "cri": 0.0,
        "cri_star": 0.0,
        "anchor_distance": propose_utils[0] if propose_utils else 0.0,
        "propose_utils": propose_utils,
        "fitted_curve": None,
        "n_steps": len(episode_payload.get("steps", [])),
    }

    if len(propose_utils) < 4:
        return result

    x = np.arange(len(propose_utils), dtype=float)
    y = np.array(propose_utils, dtype=float)

    params = fit_tanh_curve(x, y)
    if params is None:
        return result

    a, b, c, d = params
    T = len(propose_utils)

    result["tanh_params"] = params
    result["cri"] = compute_cri(a, T)
    result["cri_star"] = compute_cri_star(a, b, T)

    # Fitted curve for plotting
    x_dense = np.linspace(0, T - 1, max(T * 4, 50))
    result["fitted_curve"] = tanh_model(x_dense, a, b, c, d).tolist()
    result["_x_dense"] = x_dense.tolist()

    # Burstiness needs population-level normalisation
    if all_params is not None:
        all_a = [p[0] for p in all_params if p is not None]
        all_b = [p[1] for p in all_params if p is not None]
        if all_a and all_b:
            result["burstiness"] = compute_burstiness(a, b, all_a, all_b)

    return result


# ---------------------------------------------------------------------------
# Per-episode figure
# ---------------------------------------------------------------------------

def plot_episode(
    episode_idx: int,
    episode_payload: Dict[str, Any],
    tanh_metrics: Dict[str, Any],
    agreement: Any,
    save_dir: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot a single episode with 4 subplots:
      1. Offer utility trajectory (propose steps) + tanh fit
      2. All step utilities (propose + respond)
      3. Cumulative reward proxy (step utility)
      4. Concession speed profile derived from tanh fit
    """
    steps = episode_payload.get("steps", [])
    propose_utils = tanh_metrics["propose_utils"]
    all_utils = _extract_all_utils_in_order(episode_payload)
    tanh_params = tanh_metrics["tanh_params"]
    fitted_curve = tanh_metrics.get("fitted_curve")
    x_dense = tanh_metrics.get("_x_dense")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Episode {episode_idx}  |  "
        f"Agreement: {'YES' if agreement is not None else 'NO'}  |  "
        f"Steps: {len(steps)}",
        fontsize=12,
        fontweight="bold",
    )

    # ---- Subplot 1: Learner proposal utility over turns ----
    ax1 = axes[0, 0]
    if propose_utils:
        ax1.plot(propose_utils, "bo-", label="Proposal utility", linewidth=2)
    if fitted_curve is not None and x_dense is not None:
        ax1.plot(x_dense, fitted_curve, "r--", label="Tanh fit", linewidth=1.5)

    params_str = ""
    if tanh_params is not None:
        a, b, c, d = tanh_params
        params_str = f"a={a:.3f}, b={b:.3f}\nCRI={tanh_metrics['cri']:.3f}, τ={tanh_metrics['burstiness']:.3f}"
    ax1.set_title(f"Proposal Utility Trajectory\n{params_str}", fontsize=9)
    ax1.set_xlabel("Proposal step")
    ax1.set_ylabel("Utility")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- Subplot 2: All utilities in step order ----
    ax2 = axes[0, 1]
    if all_utils:
        propose_xs = [i for i, s in enumerate(steps) if s.get("kind") == "propose" and s.get("utility") is not None]
        respond_xs = [i for i, s in enumerate(steps) if s.get("kind") == "respond" and s.get("utility") is not None]
        p_utils = [s["utility"] for s in steps if s.get("kind") == "propose" and s.get("utility") is not None]
        r_utils = [s["utility"] for s in steps if s.get("kind") == "respond" and s.get("utility") is not None]

        if propose_xs and p_utils:
            ax2.plot(propose_xs, p_utils, "b^-", label="Propose", markersize=6)
        if respond_xs and r_utils:
            ax2.plot(respond_xs, r_utils, "rs-", label="Respond (received)", markersize=6)

    ax2.set_title("All Steps: Utility per Turn", fontsize=9)
    ax2.set_xlabel("Step index")
    ax2.set_ylabel("Utility")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- Subplot 3: Cumulative utility (proxy for reward) ----
    ax3 = axes[1, 0]
    if all_utils:
        cumsum = np.cumsum(all_utils)
        ax3.plot(cumsum, "g-o", label="Cumulative utility", linewidth=2)
    ax3.set_title("Cumulative Utility (Reward Proxy)", fontsize=9)
    ax3.set_xlabel("Step index")
    ax3.set_ylabel("Cumulative utility")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- Subplot 4: Concession speed profile from tanh ----
    ax4 = axes[1, 1]
    if tanh_params is not None and x_dense is not None:
        a, b, c, d = tanh_params
        xs = np.array(x_dense)
        # Instantaneous speed = |a*b| * sech²(ax - c)
        speed = abs(a * b) / np.cosh(a * xs - c) ** 2
        max_sp = speed.max() if speed.max() > 0 else 1.0
        speed_norm = speed / max_sp
        ax4.fill_between(xs, speed_norm, alpha=0.4, color="purple")
        ax4.plot(xs, speed_norm, "purple", linewidth=1.5)
        ax4.axhline(0.1, color="gray", linestyle="--", linewidth=1, label="θ=0.1 (CRI* threshold)")
        cri_str = f"CRI={tanh_metrics['cri']:.3f}  CRI*={tanh_metrics['cri_star']:.3f}"
        ax4.set_title(f"Concession Speed Profile\n{cri_str}", fontsize=9)
    else:
        ax4.set_title("Concession Speed Profile\n(insufficient data)", fontsize=9)

    ax4.set_xlabel("Step")
    ax4.set_ylabel("Normalised speed")
    handles, labels = ax4.get_legend_handles_labels()
    if handles:
        ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"episode_{episode_idx:04d}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plots across all episodes
# ---------------------------------------------------------------------------

def plot_per_episode_summary(
    all_tanh_metrics: List[Dict[str, Any]],
    episode_indices: List[int],
    save_dir: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    After evaluating N episodes, plot summary time-series of tanh metrics:
      - CRI over episodes
      - CRI* over episodes
      - Burstiness over episodes
      - Anchor distance over episodes
    """
    cris = [m["cri"] for m in all_tanh_metrics]
    cri_stars = [m["cri_star"] for m in all_tanh_metrics]
    burstiness = [m["burstiness"] for m in all_tanh_metrics]
    anchors = [m["anchor_distance"] for m in all_tanh_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle("Per-Episode Tanh Concession Metrics", fontsize=12, fontweight="bold")

    def _plot(ax, values, title, ylabel, color="steelblue"):
        ax.plot(episode_indices, values, "o-", color=color, markersize=4, linewidth=1.5)
        if values:
            ax.axhline(np.mean(values), color="red", linestyle="--", linewidth=1, label=f"Mean={np.mean(values):.3f}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], cris, "CRI (Concession-Rigidity Index)", "CRI", "steelblue")
    _plot(axes[0, 1], cri_stars, "CRI* (Data-driven)", "CRI*", "darkorange")
    _plot(axes[1, 0], burstiness, "Burstiness (τ)", "τ", "green")
    _plot(axes[1, 1], anchors, "Anchor Distance (first proposal utility)", "Utility", "purple")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "tanh_metrics_summary.png"), dpi=130, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
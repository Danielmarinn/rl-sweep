"""
RL Sweep — Results Plotter
==========================
Run after rl_sweep.py finishes (or at any point for partial results):

    python plot_results.py

Produces 6 publication-quality plots in results/plots/ and a summary table.

Fixes vs original:
  - KeyError 'batch_size': removed absent columns from coerce list
  - KeyError 'ent_coef' : same
  - LunarLander-v2 typo  : corrected to LunarLander-v3
  - print_summary crash  : removed batch_size/ent_coef references
  - Matplotlib deprecation: get_cmap → colormaps.get_cmap
  - New plots: violin distributions, seed-variance scatter, collapse spotlight,
    gamma×LR interaction
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_CSV = Path("results/runs.csv")
CURVES_CSV  = Path("results/curves.csv")
PLOTS_DIR   = Path("results/plots")

# ── Style ──────────────────────────────────────────────────────────────────────
ALGO_COLORS = {"PPO": "#2166ac", "DQN": "#d6604d"}
ENV_ORDER   = ["CartPole-v1", "LunarLander-v3", "Acrobot-v1"]   # FIX: was v2

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
})


# ── Data loading ───────────────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Has the sweep run?")
        sys.exit(1)

    df      = pd.read_csv(RESULTS_CSV)
    success = df[df["status"] == "success"].copy()

    # FIX: original list contained 'batch_size' and 'ent_coef', which are
    # absent from runs.csv → caused KeyError crash before any plot rendered.
    numeric_cols = [
        "mean_reward", "std_reward", "learning_rate",
        "gamma", "n_steps", "exploration_fraction", "duration_s",
    ]
    for col in numeric_cols:
        if col in success.columns:
            success[col] = pd.to_numeric(success[col], errors="coerce")

    n_total   = len(df)
    n_success = len(success)
    n_failed  = (df["status"] == "failed").sum()
    print(f"Loaded {n_total} runs  →  {n_success} success  |  {n_failed} failed\n")
    return success


def load_curves() -> pd.DataFrame | None:
    if not CURVES_CSV.exists():
        return None
    df = pd.read_csv(CURVES_CSV)
    df["timestep"]    = pd.to_numeric(df["timestep"],    errors="coerce")
    df["mean_reward"] = pd.to_numeric(df["mean_reward"], errors="coerce")
    return df.dropna(subset=["timestep", "mean_reward"])


# ── Plot 1: Violin distributions ───────────────────────────────────────────────

def plot_violin_distributions(df: pd.DataFrame) -> None:
    """
    Violin plot of reward distributions per (env, algo) across all configs.
    More informative than a bar chart — shows the full spread, not just the peak.
    """
    envs = [e for e in ENV_ORDER if e in df["env"].unique()]
    fig, axes = plt.subplots(1, len(envs), figsize=(5.5 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        sub            = df[df["env"] == env]
        algos_present  = [a for a in ["PPO", "DQN"] if a in sub["algo"].unique()]
        data           = [sub[sub["algo"] == a]["mean_reward"].dropna().values
                          for a in algos_present]
        colors         = [ALGO_COLORS[a] for a in algos_present]

        vp = ax.violinplot(data, positions=range(len(algos_present)),
                           showmedians=True, showextrema=True, widths=0.65)
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.55)
        vp["cmedians"].set_color("white")
        vp["cmedians"].set_linewidth(2.5)
        for part in ["cbars", "cmins", "cmaxes"]:
            if part in vp:
                vp[part].set_color("#444")

        for i, (algo, d) in enumerate(zip(algos_present, data)):
            ax.scatter([i], [d.max()], color=ALGO_COLORS[algo],
                       s=130, zorder=5, edgecolors="white", linewidth=1.5,
                       label=f"{algo} best: {d.max():.0f}")

        ax.set_xticks(range(len(algos_present)))
        ax.set_xticklabels(algos_present, fontsize=12, fontweight="bold")
        ax.set_title(env, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel("Mean Reward (all configs)", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Reward Distribution Across All Hyperparameter Configurations\n"
        "Violin = full distribution  |  dot = best config  |  line = median",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, "01_violin_distributions.png")


# ── Plot 2: Seed variance scatter ─────────────────────────────────────────────

def plot_seed_variance(df: pd.DataFrame) -> None:
    """
    For each configuration, plot mean reward vs inter-seed std.
    Reveals which configs are reliable vs. initialisation-sensitive.
    """
    envs = [e for e in ENV_ORDER if e in df["env"].unique()]
    fig, axes = plt.subplots(1, len(envs), figsize=(5.5 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        sub = df[df["env"] == env]
        for algo in ["PPO", "DQN"]:
            asub = sub[sub["algo"] == algo]
            if asub.empty:
                continue
            group_cols = [c for c in
                          ["learning_rate", "gamma", "n_steps", "exploration_fraction"]
                          if c in asub.columns and asub[c].notna().any()]
            grp   = asub.groupby(group_cols)["mean_reward"]
            means = grp.mean()
            stds  = grp.std().fillna(0)
            ax.scatter(means.values, stds.values,
                       color=ALGO_COLORS[algo], alpha=0.75, s=85,
                       edgecolors="white", linewidth=0.8, label=algo)

        ax.axhline(y=50, color="crimson", linestyle=":", linewidth=1.5,
                   alpha=0.75, label="σ = 50 threshold")
        ax.set_xlabel("Mean Reward (avg over seeds)", fontsize=9)
        ax.set_ylabel("Std between seeds", fontsize=9)
        ax.set_title(env, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Per-Configuration Seed Variance vs. Mean Reward\n"
        "High σ = result depends on initialisation, not just hyperparameters",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, "02_seed_variance.png")


# ── Plot 3: Learning rate sensitivity ─────────────────────────────────────────

def plot_lr_sensitivity(df: pd.DataFrame) -> None:
    envs  = [e for e in ENV_ORDER if e in df["env"].unique()]
    algos = [a for a in ["PPO", "DQN"] if a in df["algo"].unique()]

    fig, axes = plt.subplots(
        len(algos), len(envs),
        figsize=(5.5 * len(envs), 4.5 * len(algos)),
        squeeze=False,
    )

    for i, algo in enumerate(algos):
        for j, env in enumerate(envs):
            ax  = axes[i][j]
            sub = df[(df["algo"] == algo) & (df["env"] == env)]
            if sub.empty:
                ax.set_visible(False)
                continue

            grp   = sub.groupby("learning_rate")["mean_reward"]
            means = grp.mean()
            stds  = grp.std().fillna(0)
            lrs   = means.index.values
            color = ALGO_COLORS[algo]

            ax.fill_between(lrs, means - stds, means + stds,
                            color=color, alpha=0.18)
            ax.plot(lrs, means.values, "o-", color=color,
                    linewidth=2.2, markersize=8)

            for lr, m in zip(lrs, means.values):
                ax.annotate(f"{m:.0f}", (lr, m),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=8, color=color)

            ax.set_xscale("log")
            ax.set_title(f"{algo} — {env}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Learning Rate (log scale)", fontsize=8)
            ax.set_ylabel("Mean Reward", fontsize=8)

    fig.suptitle(
        "Learning Rate Sensitivity  (mean ± std across all other hyperparameters)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "03_lr_sensitivity.png")


# ── Plot 4: Gamma × LR interaction on LunarLander ────────────────────────────

def plot_gamma_lr_interaction(df: pd.DataFrame) -> None:
    """
    Shows the combined effect of gamma and learning rate on LunarLander,
    the environment where the interaction is most pronounced.
    """
    env = "LunarLander-v3"
    if env not in df["env"].unique():
        return

    ll    = df[df["env"] == env]
    algos = [a for a in ["PPO", "DQN"] if a in ll["algo"].unique()]
    fig, axes = plt.subplots(1, len(algos), figsize=(7 * len(algos), 5))
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        sub = ll[ll["algo"] == algo]
        color = ALGO_COLORS[algo]
        for gamma, marker, alpha, ls in [(0.99, "o", 0.55, "-"),
                                          (0.999, "s", 1.0, "--")]:
            gsub  = sub[sub["gamma"] == gamma].groupby("learning_rate")["mean_reward"]
            means = gsub.mean()
            stds  = gsub.std().fillna(0)
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        marker=marker, linestyle=ls, color=color,
                        capsize=5, linewidth=2.2, markersize=9, alpha=alpha,
                        label=f"γ = {gamma}")

        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate", fontsize=10)
        ax.set_ylabel("Mean Reward", fontsize=10)
        ax.set_title(f"{algo} — {env}\nγ interaction with learning rate",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=10)

    fig.suptitle(
        "Discount Factor γ × Learning Rate Interaction on LunarLander-v3\n"
        "γ = 0.999 consistently outperforms γ = 0.99 (delayed terminal reward structure)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    _save(fig, "04_gamma_lr_lunar.png")


# ── Plot 5: Acrobot collapse spotlight ────────────────────────────────────────

def plot_acrobot_collapse(df: pd.DataFrame, curves: pd.DataFrame | None) -> None:
    """
    Two-panel figure: all Acrobot results with collapse highlighted (left),
    and the divergent seed trajectories for the collapsing config (right).
    """
    acro = df[df["env"] == "Acrobot-v1"]
    if acro.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: all configs, scatter
    ax = axes[0]
    acro_sorted = acro.sort_values("mean_reward")
    plot_colors = ["#d62728" if v <= -490 else "#2166ac"
                   for v in acro_sorted["mean_reward"]]
    ax.scatter(range(len(acro_sorted)), acro_sorted["mean_reward"].values,
               c=plot_colors, s=75, alpha=0.85, edgecolors="white",
               linewidth=0.8, zorder=3)
    ax.axhline(-500, color="#d62728", linestyle="--", linewidth=1.5,
               alpha=0.65, label="Collapse floor (−500)")
    ax.axhline(acro["mean_reward"].max(), color="green", linestyle="--",
               linewidth=1.5, alpha=0.65,
               label=f"Best config ({acro['mean_reward'].max():.1f})")
    ax.set_xlabel("Run index", fontsize=10)
    ax.set_ylabel("Final Mean Reward", fontsize=10)
    ax.set_title("Acrobot-v1 — All 24 PPO Runs\nRed = catastrophic collapse",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Right panel: seed=0 vs seed=1 for the collapsing config
    ax = axes[1]
    collapse_id = "Acrobot-v1__PPO__gamma=0.999_learning_rate=0.0003_n_steps=512__s1"
    sibling_id  = "Acrobot-v1__PPO__gamma=0.999_learning_rate=0.0003_n_steps=512__s0"

    if curves is not None:
        for rid, label, color, lw in [
            (collapse_id, "seed=1  →  −500 (collapse)", "#d62728", 2.5),
            (sibling_id,  "seed=0  →  −73.6 (success)",  "#2166ac", 2.5),
        ]:
            c = curves[curves["run_id"] == rid].sort_values("timestep")
            if not c.empty:
                ax.plot(c["timestep"] / 1000, c["mean_reward"],
                        color=color, linewidth=lw, label=label)
        ax.axhline(-500, color="#d62728", linestyle=":", alpha=0.4, linewidth=1)
    else:
        ax.text(0.5, 0.5, "curves.csv not available",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Timestep (k)", fontsize=10)
    ax.set_ylabel("Mean Reward", fontsize=10)
    ax.set_title("Identical Hyperparameters — Seed 0 vs Seed 1\n"
                 "γ=0.999, lr=3×10⁻⁴, n_steps=512",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(
        "The Acrobot Collapse: Bootstrap Cascade Under High γ, Short Rollouts, ent_coef=0\n"
        "Same config — 426 reward-point gap between seeds",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, "05_acrobot_collapse.png")


# ── Plot 6: Learning curves (top 3 + worst per algo×env) ──────────────────────

def plot_learning_curves(runs_df: pd.DataFrame, curves_df: pd.DataFrame) -> None:
    envs  = [e for e in ENV_ORDER if e in runs_df["env"].unique()]
    algos = [a for a in ["PPO", "DQN"] if a in runs_df["algo"].unique()]

    fig, axes = plt.subplots(
        len(algos), len(envs),
        figsize=(6.5 * len(envs), 5 * len(algos)),
        squeeze=False,
    )

    for i, algo in enumerate(algos):
        for j, env in enumerate(envs):
            ax   = axes[i][j]
            runs = runs_df[(runs_df["algo"] == algo) & (runs_df["env"] == env)]
            if runs.empty:
                ax.set_visible(False)
                continue

            config_cols = [c for c in
                           ["learning_rate", "n_steps", "gamma", "exploration_fraction"]
                           if c in runs.columns and runs[c].notna().any()]
            ranked = (runs.groupby(config_cols)["mean_reward"]
                         .mean()
                         .sort_values(ascending=False)
                         .reset_index())

            top_n = min(3, len(ranked))
            show  = pd.concat([ranked.head(top_n), ranked.tail(1)]).drop_duplicates()

            # FIX: use colormaps instead of deprecated get_cmap
            cmap  = matplotlib.colormaps.get_cmap("tab10")

            for k, (_, cfg_row) in enumerate(show.iterrows()):
                mask = pd.Series([True] * len(runs), index=runs.index)
                for col in config_cols:
                    mask &= (runs[col] == cfg_row[col])

                run_ids     = runs[mask]["run_id"].tolist()
                seed_curves = curves_df[curves_df["run_id"].isin(run_ids)]
                if seed_curves.empty:
                    continue

                all_ts       = np.sort(seed_curves["timestep"].unique())
                seed_rewards = []
                for rid in run_ids:
                    sc = seed_curves[seed_curves["run_id"] == rid].sort_values("timestep")
                    if sc.empty:
                        continue
                    seed_rewards.append(
                        np.interp(all_ts, sc["timestep"].values, sc["mean_reward"].values)
                    )

                if not seed_rewards:
                    continue

                sr         = np.array(seed_rewards)
                mean_curve = sr.mean(axis=0)
                std_curve  = sr.std(axis=0)

                rank_str = "best" if k < top_n else "worst"
                lr_val   = cfg_row.get("learning_rate", "")
                g_val    = cfg_row.get("gamma", "")
                label    = f"[{rank_str}] lr={lr_val:.0e} γ={g_val}"
                color    = cmap(k / max(len(show) - 1, 1))

                ax.plot(all_ts, mean_curve, color=color, linewidth=2.2, label=label)
                ax.fill_between(all_ts,
                                mean_curve - std_curve,
                                mean_curve + std_curve,
                                color=color, alpha=0.14)

            ax.set_title(f"{algo} — {env}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Timestep", fontsize=8)
            ax.set_ylabel("Mean Reward", fontsize=8)
            ax.legend(fontsize=7, loc="lower right")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")
            )

    fig.suptitle(
        "Training Curves — Top 3 & Worst Config per (Algo, Env)\n"
        "Shaded = ±1 std across seeds",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "06_learning_curves.png")


# ── Terminal summary ───────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print("  SWEEP SUMMARY")
    print(sep)
    print(f"  Total successful runs : {len(df)}")
    print(f"  Environments          : {', '.join(df['env'].unique())}")
    print(f"  Algorithms            : {', '.join(df['algo'].unique())}")
    print(f"  Seeds per combo       : {df['seed'].nunique()}")
    print()
    print(f"  {'ENV':<20} {'ALGO':<5} {'BEST':>10}  {'MEAN':>8}  {'STD':>6}  "
          f"{'LR':>8}  {'GAMMA':>6}  EXTRA")
    print(sep)

    for env in ENV_ORDER:
        sub = df[df["env"] == env]
        if sub.empty:
            continue
        for algo in sorted(sub["algo"].unique()):
            asub = sub[sub["algo"] == algo]
            best = asub.loc[asub["mean_reward"].idxmax()]
            # FIX: removed batch_size / ent_coef references
            if algo == "PPO":
                n_s = int(best["n_steps"]) if pd.notna(best.get("n_steps")) else "?"
                extra = f"n_steps={n_s}"
            else:
                exp = best.get("exploration_fraction", "?")
                extra = f"exp_frac={exp}"
            print(
                f"  {env:<20} {algo:<5} {best['mean_reward']:>10.2f}  "
                f"{asub['mean_reward'].mean():>8.2f}  "
                f"{asub['mean_reward'].std():>6.2f}  "
                f"{best['learning_rate']:>8.0e}  "
                f"{best['gamma']:>6}  {extra}"
            )

    print(sep)
    avg_t  = df["duration_s"].mean()
    total_h = df["duration_s"].sum() / 3600
    print(f"  Avg run time : {avg_t:.0f}s  |  Total compute : {total_h:.1f} CPU-hours")
    print(sep)


# ── Utilities ──────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    if df.empty:
        print("No successful runs to plot yet.")
        return

    curves = load_curves()

    print("Generating plots…")
    plot_violin_distributions(df)
    plot_seed_variance(df)
    plot_lr_sensitivity(df)
    plot_gamma_lr_interaction(df)
    plot_acrobot_collapse(df, curves)

    if curves is not None and not curves.empty:
        plot_learning_curves(df, curves)
    else:
        print("  ⚠  curves.csv not found — skipping learning curves.")

    print_summary(df)
    print(f"\nAll plots saved to: {PLOTS_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
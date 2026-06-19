#!/usr/bin/env python3
"""
Genera los gráficos del informe T3 a partir de los CSVs de results/jobs/.
Uso: python results/generate_figures.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
})


def load_results(jobs_dir):
    rows = []
    for path in sorted(jobs_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_job"] = path.stem
                row["_file"] = path.name
                rows.append(row)
    return rows


def val(row, key):
    return float(row[key])


def fig1_total_time(rows, out):
    official = [r for r in rows if "official" in r["_job"]]
    grids = {}
    for r in official:
        pr, pc, T = int(r["pr"]), int(r["pc"]), int(r["threads"])
        key = f"{pr}x{pc}"
        grids.setdefault(key, {"T": [], "total": [], "procs": pr * pc})
        grids[key]["T"].append(T)
        grids[key]["total"].append(val(r, "elapsed_sec"))

    sorted_grids = sorted(grids.items(), key=lambda x: x[1]["procs"])
    labels = [g[0] for g in sorted_grids]

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.25
    x = np.arange(len(labels))

    for idx, T_val in enumerate([1, 2, 4]):
        vals = []
        for g in sorted_grids:
            match = [v for t, v in zip(g[1]["T"], g[1]["total"]) if t == T_val]
            vals.append(match[0] if match else 0)
        bars = ax.bar(x + (idx - 1) * width, vals, width, label=f"T = {T_val}")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Configuracion de grid (pr x pc)")
    ax.set_ylabel("Tiempo total (s)")
    ax.set_title("Figura 1: Tiempo total de ejecucion por configuracion y numero de threads")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig2_speedup(rows, out):
    official = [r for r in rows if "official" in r["_job"]]
    T_base = None
    for r in official:
        if int(r["pr"]) == 1 and int(r["pc"]) == 1 and int(r["threads"]) == 1:
            T_base = val(r, "elapsed_sec")
            break

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {1: "o", 2: "s", 4: "^"}
    colors = {1: "#2196F3", 2: "#FF9800", 4: "#4CAF50"}

    for T_val in [1, 2, 4]:
        procs_list, speedup_list = [], []
        for r in official:
            if int(r["threads"]) == T_val:
                p = int(r["pr"]) * int(r["pc"])
                procs_list.append(p)
                speedup_list.append(T_base / val(r, "elapsed_sec"))
        paired = sorted(zip(procs_list, speedup_list))
        px, sy = zip(*paired)
        ax.plot(px, sy, marker=markers[T_val], color=colors[T_val],
                label=f"T = {T_val}", linewidth=2, markersize=8,
                markeredgecolor="white", markeredgewidth=1)

    procs_ideal = [1, 2, 4, 8, 16]
    ax.plot(procs_ideal, procs_ideal, "--", color="gray", alpha=0.5, label="Speedup ideal")
    ax.set_xlabel("Numero de procesos MPI (pr x pc)")
    ax.set_ylabel("Speedup (T_base / T_config)")
    ax.set_title("Figura 2: Speedup relativo a la configuracion base (1x1, T=1)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig3_grid_shape(rows, out):
    official = [r for r in rows if "official" in r["_job"]]
    configs_4 = []
    for r in official:
        if int(r["pr"]) * int(r["pc"]) == 4 and int(r["threads"]) == 4:
            configs_4.append(r)
    configs_4.sort(key=lambda r: (int(r["pr"]), int(r["pc"])))

    labels = [f"{r['pr']}x{r['pc']}" for r in configs_4]
    totals = [val(r, "elapsed_sec") for r in configs_4]
    per_iters = [val(r, "time_per_iter") for r in configs_4]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.35

    bars = ax1.bar(x - width / 2, totals, width, label="Tiempo total (s)", color=colors, edgecolor="white")
    for bar, val_ in zip(bars, totals):
        ax1.annotate(f"{val_:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, per_iters, width, label="Tiempo/iteracion (s)",
                    color=colors, alpha=0.5, hatch="//")

    ax1.set_xlabel("Configuracion de grid (pr x pc)")
    ax1.set_ylabel("Tiempo total (s)")
    ax2.set_ylabel("Tiempo por iteracion (s)")
    ax1.set_title("Figura 3: Efecto de la forma del grid (pr x pc = 4, T = 4)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig4_profile(rows, out):
    r = None
    for row in rows:
        if row["_job"] == "nmf_p1x1_t1_official":
            r = row
            break
    if r is None:
        print("  WARN: nmf_p1x1_t1_official not found, skipping fig4")
        return

    labels = ["Error\n$\\|A-WH^\\top\\|_F$", "$A^\\top W$", "$AH$",
              "Solve $W$", "$H^\\top H$", "$W^\\top W$", "Solve $H$"]
    times = [val(r, "profile_error"), val(r, "profile_atw"), val(r, "profile_ah"),
             val(r, "profile_solve_w"), val(r, "profile_hth"),
             val(r, "profile_wtw"), val(r, "profile_solve_h")]
    total = sum(times)
    pcts = [100.0 * t / total for t in times]
    colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1", "#6D4C41"]

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))

    ax_pie.pie(pcts, labels=None, autopct="%1.1f%%", colors=colors, startangle=90, pctdistance=0.8)
    ax_pie.legend(labels, loc="center left", bbox_to_anchor=(-0.3, 0.5), fontsize=9)
    ax_pie.set_title("Distribucion del tiempo (1x1, T=1)")

    ax_bar.barh(labels[::-1], times[::-1], color=colors[::-1], edgecolor="white")
    ax_bar.set_xlabel("Tiempo (s)")
    ax_bar.set_title("Tiempo absoluto por operacion (1x1, T=1)")
    for i, (t, label) in enumerate(zip(times[::-1], labels[::-1])):
        ax_bar.text(t + 0.1, i, f"{t:.2f}s", va="center", fontsize=9)

    plt.suptitle("Figura 4: Perfil de ejecucion - Cuello de botella", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig5_numa(rows, out):
    numa_same = None
    numa_split = None
    for r in rows:
        if "numa_same_socket" in r["_job"]:
            numa_same = r
        if "numa_socket_split" in r["_job"]:
            numa_split = r

    if not numa_same or not numa_split:
        print("  WARN: NUMA results not found, skipping fig5")
        return

    labels = ["Mismo socket\n(--ntasks-per-socket=2)", "Sockets distintos\n(--ntasks-per-socket=1)"]
    times = [val(numa_same, "elapsed_sec"), val(numa_split, "elapsed_sec")]
    colors = ["#E53935", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, times, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Tiempo total (s)")
    ax.set_title("Figura 5: Efecto de la afinidad NUMA (pr=2, pc=1, T=2)")
    ax.grid(axis="y", alpha=0.3)

    for bar, t in zip(bars, times):
        ax.annotate(f"{t:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    diff_pct = (times[0] - times[1]) / times[0] * 100
    ax.annotate("", xy=(1, times[1]), xytext=(0, times[0]),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(0.5, (times[0] + times[1]) / 2, f"-{diff_pct:.0f}%", ha="center", va="center",
            fontsize=12, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig6_heatmap(rows, out):
    official = [r for r in rows if "official" in r["_job"]]
    grid_map = {}
    for r in official:
        if int(r["threads"]) == 4:
            pr, pc = int(r["pr"]), int(r["pc"])
            grid_map[(pr, pc)] = val(r, "elapsed_sec")

    prows = sorted(set(pr for pr, _ in grid_map.keys()))
    pcols = sorted(set(pc for _, pc in grid_map.keys()))
    matrix = np.full((len(prows), len(pcols)), np.nan)

    for (pr, pc), t in grid_map.items():
        ri = prows.index(pr)
        ci = pcols.index(pc)
        matrix[ri, ci] = t

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=3, vmax=10)
    ax.set_xticks(range(len(pcols)))
    ax.set_xticklabels([f"pc={pc}" for pc in pcols])
    ax.set_yticks(range(len(prows)))
    ax.set_yticklabels([f"pr={pr}" for pr in prows])
    ax.set_title("Figura 6: Mapa de calor del tiempo total (s) - T=4")

    for ri in range(len(prows)):
        for ci in range(len(pcols)):
            v = matrix[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.2f}s", ha="center", va="center", fontsize=12,
                        fontweight="bold", color="white" if v > 6 else "black")

    plt.colorbar(im, ax=ax, label="Tiempo total (s)")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    repo = Path(__file__).resolve().parent.parent
    jobs_dir = repo / "results" / "jobs"
    out_dir = repo / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not list(jobs_dir.glob("*.csv")):
        print("ERROR: No CSV files in results/jobs/")
        sys.exit(1)

    print("Loading results...")
    rows = load_results(jobs_dir)
    print(f"  {len(rows)} rows loaded")

    print("Generating figures...")
    fig1_total_time(rows, out_dir / "fig1_tiempo_total.png")
    fig2_speedup(rows, out_dir / "fig2_speedup.png")
    fig3_grid_shape(rows, out_dir / "fig3_forma_grid.png")
    fig4_profile(rows, out_dir / "fig4_perfil.png")
    fig5_numa(rows, out_dir / "fig5_numa.png")
    fig6_heatmap(rows, out_dir / "fig6_heatmap.png")

    print(f"\nDone! Figures saved to {out_dir}")


if __name__ == "__main__":
    main()

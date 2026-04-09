from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

from environment import build_env_base
from IPython.display import HTML, display


def make_edge(cfg_path: str):
    env = build_env_base(cfg_path)
    return env.edge_areas[0]


def run_qoe_vs_requests(
    cfg_path: str,
    req_list,
    cpu_to_ids_ratio: float,
    attack_intensity_per_sec: float,  # flows/sec — converted to flows/step internally
    attack_mode: str,                 # "cpu_only" | "bw_only" | "both"
    t: int = 0,
):
    edge = make_edge(cfg_path)

    # IDS / VA CPU split
    edge.ids_cpu = float(cpu_to_ids_ratio) * edge.budget.cpu
    edge.va_cpu  = edge.budget.cpu - edge.ids_cpu

    dt = edge.slot_ms / 1000.0
    flows_per_step = attack_intensity_per_sec * dt

    # --- backup attacker state ---
    atk_backups = []
    for a in edge.attackers:
        atk_backups.append({
            "atk":           a,
            "cycle_per_flow": float(getattr(a, "cycle_per_flow", 0.0)),
            "bw_per_flow":    float(getattr(a, "bw_per_flow",    0.0)),
            "start":          int(getattr(a,   "start",          0)),
            "active_len":     int(getattr(a,   "active_len",     0)),
            "scaling":        float(getattr(a, "scaling",        1.0)),
            "_flows":         a._flows.copy() if hasattr(a, "_flows") else None,
            "_flows_ema":     a._flows_ema.copy() if hasattr(a, "_flows_ema") else None,
        })

    # --- backup IDS FPR and zero it out so benign users are never dropped ---
    orig_acc_tpr_fpr = {k: v for k, v in edge.ids.acc_tpr_fpr.items()}

    # --- backup user methods ---
    orig_num_requests_at = [u.num_requests_at for u in edge.users]


    try:
        # Zero FPR so benign users are never stochastically dropped
        edge.ids.acc_tpr_fpr = {k: (tpr, 0.0) for k, (tpr, _fpr) in orig_acc_tpr_fpr.items()}

        # Override attackers: constant intensity, selected mode
        for a in edge.attackers:
            a.start      = 0          # make t=0 fall inside active window
            a.scaling    = 1.0
            a._flows     = np.full(a.active_len, float(flows_per_step), dtype=np.float32)
            a._flows_ema = np.full(a.active_len, float(flows_per_step), dtype=np.float32)
            a.bw_per_flow = 0.0
            a.latency_per_flow = 0.5
            a.cycle_per_flow = a.latency_per_flow * edge.cpu_cycle_per_ms * 8.0
            if attack_mode == "cpu_only":
                a.bw_per_flow = 0.0
            elif attack_mode == "bw_only":
                a.cycle_per_flow = 0.0
            elif attack_mode == "both":
                pass  # keep both as configured
            else:
                raise ValueError(f"unknown attack_mode={attack_mode!r}")

        qoes = []
        for N in req_list:
            # Override user arrivals: first user = N requests/step, rest = 0
            for i, u in enumerate(edge.users):
                u.num_requests_at = (lambda _t, _N=N: int(_N)) if i == 0 else (lambda _t: 0)

            cache = edge.step_local(t)
            qoes.append(float(cache.get("qoe", 0.0)))

        return np.asarray(req_list, dtype=np.float32), np.asarray(qoes, dtype=np.float32)

    finally:
        # Restore IDS FPR
        edge.ids.acc_tpr_fpr = orig_acc_tpr_fpr

        # Restore users
        for u, fn in zip(edge.users, orig_num_requests_at):
            u.num_requests_at = fn

        # Restore attackers
        for bk in atk_backups:
            a = bk["atk"]
            a.cycle_per_flow = bk["cycle_per_flow"]
            a.bw_per_flow    = bk["bw_per_flow"]
            a.start          = bk["start"]
            a.active_len     = bk["active_len"]
            a.scaling        = bk["scaling"]
            if bk["_flows"] is not None:
                a._flows     = bk["_flows"]
            if bk["_flows_ema"] is not None:
                a._flows_ema = bk["_flows_ema"]


def main():
    cfg_path = "./configs/simulation_0.yaml"

    # x-axis: user requests per step
    req_list = list(range(0, 25))

    # IDS CPU allocations (cores)
    ids_allocations  = [0.0, 1.0, 2.0, 3.0, 4.0]
    total_cpu        = make_edge(cfg_path).budget.cpu
    cpu_to_ids_ratios = [a / total_cpu for a in ids_allocations]

    # Slider 1: attack intensity in flows/sec (matches lambda_base range 7000–14000)
    attack_intensities = list(range(0, 15001, 500))
    # Slider 2: attack mode
    attack_modes = ["cpu_only", "bw_only", "both"]

    fig = go.Figure()

    n_modes = len(attack_modes)
    n_atk   = len(attack_intensities)
    n_ids   = len(ids_allocations)

    for mode in attack_modes:
        for atk in attack_intensities:
            for ids_alloc, r in zip(ids_allocations, cpu_to_ids_ratios):
                x, y = run_qoe_vs_requests(
                    cfg_path=cfg_path,
                    req_list=req_list,
                    cpu_to_ids_ratio=r,
                    attack_intensity_per_sec=float(atk),
                    attack_mode=mode,
                    t=0,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"IDS={ids_alloc:.1f}c",
                        hovertemplate="N=%{x}<br>QoE=%{y:.4f}<extra></extra>",
                        visible=False,
                        legendgroup=f"ids_{ids_alloc:.1f}",
                    )
                )
def main():
    cfg_path = "./configs/simulation_0.yaml"

    # x-axis: user requests per step
    req_list = list(range(0, 25))

    # IDS CPU allocations (cores)
    ids_allocations  = [0.0, 1.0, 2.0]
    total_cpu        = make_edge(cfg_path).budget.cpu
    cpu_to_ids_ratios = [a / total_cpu for a in ids_allocations]

    # Slider 1: attack intensity in flows/sec (matches lambda_base range 7000–14000)
    attack_intensities = list(range(0, 15001, 1000))
    # Slider 2: attack mode
    attack_modes = ["cpu_only", "bw_only", "both"]
    attack_modes = ["cpu_only"]

    fig = go.Figure()

    n_modes = len(attack_modes)
    n_atk   = len(attack_intensities)
    n_ids   = len(ids_allocations)

    for mode in attack_modes:
        for atk in attack_intensities:
            for ids_alloc, r in zip(ids_allocations, cpu_to_ids_ratios):
                x, y = run_qoe_vs_requests(
                    cfg_path=cfg_path,
                    req_list=req_list,
                    cpu_to_ids_ratio=r,
                    attack_intensity_per_sec=float(atk),
                    attack_mode=mode,
                    t=0,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"IDS={ids_alloc:.1f}c",
                        hovertemplate="N=%{x}<br>QoE=%{y:.4f}<extra></extra>",
                        visible=False,
                        legendgroup=f"ids_{ids_alloc:.1f}",
                    )
                )

    def set_visible(mode_idx: int, atk_idx: int):
        vis = [False] * (n_modes * n_atk * n_ids)
        base = mode_idx * (n_atk * n_ids) + atk_idx * n_ids
        for j in range(n_ids):
            vis[base + j] = True
        return vis

    # Initial view
    init_mode_idx = 0
    init_atk_idx  = 0
    for i, v in enumerate(set_visible(init_mode_idx, init_atk_idx)):
        fig.data[i].visible = v

    # Slider 1: attack intensity
    atk_steps = []
    for ai, atk in enumerate(attack_intensities):
        atk_steps.append(dict(
            method="update",
            label=str(atk),
            args=[
                {"visible": set_visible(init_mode_idx, ai)},
                {"title": f"QoE vs Requests | mode={attack_modes[init_mode_idx]} | attack={atk} flows/s"},
            ],
        ))

    # Slider 2: attack mode
    mode_steps = []
    for mi, mode in enumerate(attack_modes):
        mode_steps.append(dict(
            method="update",
            label=mode,
            args=[
                {"visible": set_visible(mi, init_atk_idx)},
                {"title": f"QoE vs Requests | mode={mode} | attack={attack_intensities[init_atk_idx]} flows/s"},
            ],
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"VA Performance vs Requests | "
                f"mode={attack_modes[init_mode_idx]} | "
                f"attack={attack_intensities[init_atk_idx]} flows/s"
            ),
            font=dict(size=22),
        ),
        xaxis=dict(
            title=dict(text="User Requests per Step (N)", font=dict(size=20)),
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            title=dict(text="VA Performance (QoE)", font=dict(size=20)),
            range=[0, 1.05],
            tickfont=dict(size=16),
        ),
        hovermode="x unified",
        legend=dict(
            title=dict(text="IDS CPU allocation", font=dict(size=20)),
            font=dict(size=16),
            x=1.12,
            y=1.0,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=80, r=180, t=100, b=120),
        sliders=[
            dict(
                active=init_atk_idx,
                currentvalue={"prefix": "Attack intensity (flows/s): ", "font": {"size": 18}},
                pad={"t": 40},
                steps=atk_steps,
                x=0.05,
                len=0.9,
            ),
            dict(
                active=init_mode_idx,
                currentvalue={"prefix": "Attack mode: ", "font": {"size": 18}},
                pad={"t": 90},
                steps=mode_steps,
                x=0.05,
                len=0.9,
            ),
        ],
    )

    out_path = "qoe_vs_requests_two_sliders.html"
    plot(fig, filename=out_path, auto_open=False)

    with open(out_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    display(HTML(html_content))


if __name__ == "__main__":
    main()

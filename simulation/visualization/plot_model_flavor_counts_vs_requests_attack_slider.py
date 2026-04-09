from __future__ import annotations
from IPython.display import HTML

import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

from environment import build_env_base


def make_edge(cfg_path: str):
    env = build_env_base(cfg_path)
    return env.edge_areas[0]


def _extract_model_flavor_counts(cache: dict, flavor_order: list[str]) -> dict[str, float]:
    counts = cache.get("od_plan", {})
    if counts is None:
        counts = {}
    return {f: float(counts.get(f, 0.0)) for f in flavor_order}


def run_model_flavor_counts_vs_requests(
    cfg_path: str,
    req_list,
    cpu_to_ids_ratio: float,
    attack_intensity_per_sec: float,  # flows/sec — converted to flows/step internally
    attack_mode: str,                  # "cpu_only" | "bw_only" | "both"
    flavor_order: list[str],
    t: int = 0,
):
    edge = make_edge(cfg_path)

    edge.ids_cpu = float(cpu_to_ids_ratio) * edge.budget.cpu
    edge.va_cpu  = edge.budget.cpu - edge.ids_cpu

    dt = edge.slot_ms / 1000.0
    flows_per_step = attack_intensity_per_sec * dt

    # --- backup attacker state ---
    atk_backups = []
    for a in edge.attackers:
        atk_backups.append({
            "atk":            a,
            "cycle_per_flow": float(getattr(a, "cycle_per_flow", 0.0)),
            "bw_per_flow":    float(getattr(a, "bw_per_flow",    0.0)),
            "start":          int(getattr(a,   "start",          0)),
            "active_len":     int(getattr(a,   "active_len",     0)),
            "scaling":        float(getattr(a, "scaling",        1.0)),
            "_flows":         a._flows.copy() if hasattr(a, "_flows") else None,
            "_flows_ema":     a._flows_ema.copy() if hasattr(a, "_flows_ema") else None,
        })

    # --- backup IDS FPR ---
    orig_acc_tpr_fpr = {k: v for k, v in edge.ids.acc_tpr_fpr.items()}

    # --- backup user methods ---
    orig_num_requests_at = [u.num_requests_at for u in edge.users]

    try:
        # Zero FPR so benign users are never stochastically dropped
        edge.ids.acc_tpr_fpr = {k: (tpr, 0.0) for k, (tpr, _fpr) in orig_acc_tpr_fpr.items()}

        # Override attackers: constant intensity, selected mode
        for a in edge.attackers:
            a.start           = 0
            a.scaling         = 1.0
            a._flows          = np.full(a.active_len, float(flows_per_step), dtype=np.float32)
            a._flows_ema      = np.full(a.active_len, float(flows_per_step), dtype=np.float32)
            a.bw_per_flow     = 0.0
            a.latency_per_flow = 0.5
            a.cycle_per_flow  = a.latency_per_flow * edge.cpu_cycle_per_ms * 8.0

            if attack_mode == "cpu_only":
                a.bw_per_flow = 0.0
            elif attack_mode == "bw_only":
                a.cycle_per_flow = 0.0
            elif attack_mode == "both":
                pass
            else:
                raise ValueError(f"unknown attack_mode={attack_mode!r}")

        y_by_series = {f: [] for f in flavor_order}
        y_by_series["dropped"] = []
        qoe_list = []

        for N in req_list:
            for i, u in enumerate(edge.users):
                u.num_requests_at = (lambda _t, _N=N: int(_N)) if i == 0 else (lambda _t: 0)

            cache = edge.step_local(t)
            counts = _extract_model_flavor_counts(cache, flavor_order)

            total_req_in = float(cache.get("local_num_request", N))
            served = float(sum(counts.values()))
            dropped = max(0.0, total_req_in - served)

            for f in flavor_order:
                y_by_series[f].append(float(counts[f]))
            y_by_series["dropped"].append(dropped)
            qoe_list.append(float(cache.get("qoe", 0.0)))

        x = np.asarray(req_list, dtype=np.float32)
        y_by_series = {k: np.asarray(v, dtype=np.float32) for k, v in y_by_series.items()}
        return x, y_by_series, np.asarray(qoe_list, dtype=np.float32)

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

    # x-axis: user requests per step (after * dt fix, mu_max=60 → ~12 req/step at dt=0.2)
    req_list = list(range(0, 25))

    ids_alloc = 3.0
    edge0     = make_edge(cfg_path)
    total_cpu = edge0.budget.cpu
    cpu_to_ids_ratio = ids_alloc / total_cpu

    flavor_order = list(edge0.pipeline.det_cycles.keys())
    if len(flavor_order) != 3:
        raise ValueError(f"Expected 3 model flavors, got {len(flavor_order)}: {flavor_order}")

    series_order = flavor_order + ["dropped"]

    legend_name_map = {
        flavor_order[0]: "light",
        flavor_order[1]: "medium",
        flavor_order[2]: "heavy",
        "dropped": "dropped",
    }

    color_map = {
        flavor_order[0]: "#636EFA",
        flavor_order[1]: "#00CC96",
        flavor_order[2]: "#FFA15A",
        "dropped": "#FF0000",
    }

    # Attack intensity in flows/sec — matches lambda_base range [7000, 14000]
    attack_intensities = list(range(0, 15001, 1000))
    attack_mode        = "cpu_only"

    fig = go.Figure()

    n_atk          = len(attack_intensities)
    n_stack_series = len(series_order)
    traces_per_atk = n_stack_series + 1  # stacked bars + QoE line

    for atk in attack_intensities:
        x, y_by_series, qoe_arr = run_model_flavor_counts_vs_requests(
            cfg_path=cfg_path,
            req_list=req_list,
            cpu_to_ids_ratio=cpu_to_ids_ratio,
            attack_intensity_per_sec=float(atk),
            attack_mode=attack_mode,
            flavor_order=flavor_order,
            t=0,
        )

        for series_name in series_order:
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y_by_series[series_name],
                    name=legend_name_map.get(series_name, series_name),
                    marker=dict(color=color_map.get(series_name)),
                    hovertemplate=(
                        "N=%{x}<br>"
                        f"Series={legend_name_map.get(series_name, series_name)}<br>"
                        "Count=%{y:.0f}<extra></extra>"
                    ),
                    visible=False,
                    legendgroup=series_name,
                    yaxis="y",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=qoe_arr,
                mode="lines+markers",
                name="VA Performance",
                line=dict(color="black", width=4),
                marker=dict(size=8, color="black"),
                hovertemplate="N=%{x}<br>VA Performance=%{y:.4f}<extra></extra>",
                visible=False,
                legendgroup="qoe",
                yaxis="y2",
            )
        )

    def set_visible(atk_idx: int):
        vis = [False] * (n_atk * traces_per_atk)
        base = atk_idx * traces_per_atk
        for j in range(traces_per_atk):
            vis[base + j] = True
        return vis

    init_atk_idx = 0
    for i, v in enumerate(set_visible(init_atk_idx)):
        fig.data[i].visible = v

    atk_steps = []
    for ai, atk in enumerate(attack_intensities):
        atk_steps.append(dict(
            method="update",
            label=str(atk),
            args=[
                {"visible": set_visible(ai)},
                {"title": (
                    f"Detector flavor counts vs Requests | "
                    f"IDS={ids_alloc:.1f}c | mode={attack_mode} | attack={atk} flows/s"
                )},
            ],
        ))

    fig.update_layout(
        title=(
            f"Detector flavor counts vs Requests | "
            f"IDS={ids_alloc:.1f}c | mode={attack_mode} | "
            f"attack={attack_intensities[init_atk_idx]} flows/s"
        ),
        barmode="stack",
        xaxis=dict(
            title=dict(text="User Requests per Step (N)", font=dict(size=20)),
            tickmode="array",
            tickvals=req_list,
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            title=dict(text="Request count", font=dict(size=20)),
            tickfont=dict(size=16),
        ),
        yaxis2=dict(
            title=dict(text="VA Performance", font=dict(size=20)),
            overlaying="y",
            side="right",
            range=[0, 1.05],
            tickfont=dict(size=16),
        ),
        hovermode="x unified",
        legend=dict(
            title=dict(text="Model Flavor", font=dict(size=20)),
            font=dict(size=16),
            x=1.12,
            y=1.0,
            xanchor="left",
            yanchor="top",
        ),
        sliders=[
            dict(
                active=init_atk_idx,
                currentvalue={"prefix": "Attack intensity (flows/s): ", "font": {"size": 18}},
                pad={"t": 80},
                steps=atk_steps,
                x=0.05,
                len=0.9,
            ),
        ],
        margin=dict(l=80, r=90, t=100, b=80),
    )

    out_path = f"flavor_vs_requests_atk_slider_ids_{ids_alloc}.html"
    plot(fig, filename=out_path, auto_open=False)

    with open(out_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    display(HTML(html_content))


if __name__ == "__main__":
    main()

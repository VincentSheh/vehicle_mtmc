"""
Convert ground-truth tracking CSV → per-epoch request arrival rate trace.

Usage:
    python gt_to_trace.py gt.csv                          # defaults: 30fps, 200ms epochs
    python gt_to_trace.py gt.csv --fps 25 --epoch_ms 200
    python gt_to_trace.py gt.csv --fps 30 --epoch_ms 200 --mu_min 80 --mu_max 160

Input CSV format (per-frame bounding boxes):
    frame, bbox_topleft_x, bbox_topleft_y, bbox_width, bbox_height, track_id, color, type

Outputs:
    trace_arrival_rate.csv   — epoch-level arrival rate trace
    trace_overview.png       — full trace plot (3 panels)
    trace_zoom.png           — zoomed detail
    trace_distribution.png   — histogram of rates
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ── CLI ──────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="GT tracking CSV → arrival rate trace")
p.add_argument("csv", type=str, default="/home/disk_4TB/vincent/edgeids/edgeids/vehicle_mtmc/output/bellevue_150th_eastgate_full/gt.csv", 
               help="Path to ground-truth CSV")
p.add_argument("--fps", type=float, default=5.0, help="Video frame rate (default: 30)")
p.add_argument("--epoch_ms", type=float, default=1000.0, help="Epoch width in ms (default: 200)")
p.add_argument("--mu_min", type=float, default=80.0, help="Target min request rate (req/s)")
p.add_argument("--mu_max", type=float, default=160.0, help="Target max request rate (req/s)")
p.add_argument("--count_mode", choices=["objects", "new_tracks"], default="objects",
               help="'objects' = unique objects per frame; 'new_tracks' = newly appearing tracks per epoch")
p.add_argument("--out_prefix", default="trace", help="Output filename prefix")
args = p.parse_args()

# ── Load ─────────────────────────────────────────────────────────────
print(f"Loading {args.csv} ...")
df = pd.read_csv(args.csv)
print(f"  Rows: {len(df):,}")
print(f"  Columns: {list(df.columns)}")
print(f"  Frame range: {df['frame'].min()} – {df['frame'].max()}")
print(f"  Unique tracks: {df['track_id'].nunique()}")
print(f"  Unique frames: {df['frame'].nunique()}")

frame_min = df["frame"].min()
frame_max = df["frame"].max()
total_frames = frame_max - frame_min + 1
video_duration_s = total_frames / args.fps

print(f"\n  Total frames: {total_frames:,}")
print(f"  Video duration: {video_duration_s:.1f} s ({video_duration_s/3600:.2f} h)")

# ── Per-frame object count ───────────────────────────────────────────
if args.count_mode == "objects":
    per_frame = df.groupby("frame")["track_id"].nunique().reset_index()
    per_frame.columns = ["frame", "count"]
else:
    # New tracks: first appearance of each track_id
    first_seen = df.groupby("track_id")["frame"].min().reset_index()
    first_seen.columns = ["track_id", "frame"]
    per_frame = first_seen.groupby("frame").size().reset_index(name="count")

# Fill all frames (including those with 0 objects)
all_frames = pd.DataFrame({"frame": np.arange(frame_min, frame_max + 1)})
per_frame = all_frames.merge(per_frame, on="frame", how="left").fillna(0)
per_frame["count"] = per_frame["count"].astype(float)

# ── Omit ranges 4-11h and 15-20h ─────────────────────────────────────
print(f"\nFiltering: Omit 4h-11h and 15h-20h ranges...")
per_frame["time_hr"] = (per_frame["frame"] - frame_min) / args.fps / 3600.0

# ── Export Full Data (unfiltered) ───────────────────────────────────
_fpe = int(args.fps * args.epoch_ms / 1000.0)
_n_epochs = len(per_frame) // _fpe
_counts = per_frame["count"].values[: _n_epochs * _fpe].reshape(_n_epochs, _fpe).mean(axis=1)
_times = np.arange(_n_epochs) * (args.epoch_ms / 1000.0)

out_full_raw = f"{args.out_prefix}_full_raw_counts.csv"
pd.DataFrame({"time_s": _times, "raw_count_per_epoch": _counts}).to_csv(out_full_raw, index=False, float_format="%.4f")
print(f"Saved full raw counts (inc. omitted) to: {out_full_raw}")

# ── Export 50s Averages (unfiltered) ────────────────────────────────
_win_s = 50.0
_win_ep = int(_win_s / (args.epoch_ms / 1000.0))
if _win_ep > 0:
    _n_win = len(_counts) // _win_ep
    _avg_50s = [_counts[i*_win_ep:(i+1)*_win_ep].mean() for i in range(_n_win)]
    _time_50s = np.arange(_n_win) * _win_s + (_win_s / 2.0)
    out_50s = f"{args.out_prefix}_50s_averages.csv"
    pd.DataFrame({"time_s": _time_50s, "avg_count_per_epoch": _avg_50s}).to_csv(out_50s, index=False, float_format="%.4f")
    print(f"Saved 50s averages to: {out_50s}")

mask = (
    (per_frame["time_hr"] <= 4.0) | 
    ((per_frame["time_hr"] >= 11.0) & (per_frame["time_hr"] <= 15.0)) | 
    (per_frame["time_hr"] >= 20.0)
)
per_frame = per_frame[mask]
print(f"  Remaining frames: {len(per_frame):,}")



# ── Aggregate to epochs ──────────────────────────────────────────────
frames_per_epoch = int(args.fps * args.epoch_ms / 1000.0)
n_epochs = len(per_frame) // frames_per_epoch

print(f"\n  Frames per epoch: {frames_per_epoch}")
print(f"  Total epochs: {n_epochs:,}")
print(f"  Epoch duration: {args.epoch_ms:.0f} ms")

counts = per_frame["count"].values[: n_epochs * frames_per_epoch]
epoch_counts = counts.reshape(n_epochs, frames_per_epoch).mean(axis=1)
epoch_time_s = np.arange(n_epochs) * (args.epoch_ms / 1000.0)
epoch_time_min = epoch_time_s / 60.0
epoch_time_hr = epoch_time_s / 3600.0

# ── Scale to target request rate range ───────────────────────────────
# Use percentile-based scaling to be robust to outliers
p2, p98 = np.percentile(epoch_counts, [2, 98])
scaled = args.mu_min + (epoch_counts - p2) / (p98 - p2 + 1e-9) * (args.mu_max - args.mu_min)
scaled = np.clip(scaled, args.mu_min * 0.8, args.mu_max * 1.2)  # allow slight overshoot

# Rolling means
edf = pd.DataFrame({
    "time_s": epoch_time_s,
    "raw_count": epoch_counts,
    "rate_scaled": scaled,
})
for w in [5, 20, 50, 200]:
    edf[f"raw_roll_{w}"] = edf["raw_count"].rolling(w, center=True, min_periods=1).mean()
    edf[f"rate_roll_{w}"] = edf["rate_scaled"].rolling(w, center=True, min_periods=1).mean()

# ── Save CSV ─────────────────────────────────────────────────────────
out_csv = f"{args.out_prefix}_arrival_rate.csv"
edf.to_csv(out_csv, index=False, float_format="%.4f")
print(f"\nSaved: {out_csv}")

# ── Stats ────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"TRACE STATISTICS")
print(f"{'='*50}")
print(f"  Duration:       {epoch_time_s[-1]:.1f} s  ({epoch_time_hr[-1]:.2f} h)")
print(f"  Epochs:         {n_epochs:,}")
print(f"  Count mode:     {args.count_mode}")
print(f"")
print(f"  Raw count/epoch:")
print(f"    Min:    {epoch_counts.min():.2f}")
print(f"    Max:    {epoch_counts.max():.2f}")
print(f"    Mean:   {epoch_counts.mean():.2f}")
print(f"    Std:    {epoch_counts.std():.2f}")
print(f"    CV:     {epoch_counts.std()/epoch_counts.mean():.2f}")
print(f"")
print(f"  Scaled rate [{args.mu_min}, {args.mu_max}] req/s:")
print(f"    Min:    {scaled.min():.1f}")
print(f"    Max:    {scaled.max():.1f}")
print(f"    Mean:   {scaled.mean():.1f}")
print(f"    Std:    {scaled.std():.1f}")
print(f"    CV:     {scaled.std()/scaled.mean():.2f}")

# ── Colors ───────────────────────────────────────────────────────────
C_RAW    = "#a3c4dc"
C_R20    = "#2b7a78"
C_R50    = "#e07b54"
C_R200   = "#1b1b2f"
C_GRAY   = "#888888"

# ── Decide x-axis unit ──────────────────────────────────────────────
if epoch_time_hr[-1] > 2:
    xvals, xlabel = epoch_time_hr, "Time (h)"
elif epoch_time_s[-1] > 300:
    xvals, xlabel = epoch_time_min, "Time (min)"
else:
    xvals, xlabel = epoch_time_s, "Time (s)"

# ── Plot 1: Overview (3 panels) ─────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(20, 14))

# Panel A: Raw epoch counts
ax = axes[0]
ax.fill_between(xvals, 0, epoch_counts, color=C_RAW, alpha=0.35, linewidth=0)
ax.plot(xvals, edf["raw_roll_20"], color=C_R20, lw=1.0, alpha=0.7, label="Roll 20 (4s)")
ax.plot(xvals, edf["raw_roll_200"], color=C_R200, lw=1.8, label="Roll 200 (40s)")
ax.set_ylabel("Objects / epoch", fontsize=12)
ax.set_title("(a) Raw Object Count per Epoch (200ms)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim(xvals[0], xvals[-1])
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)

# Panel B: Scaled request rate
ax = axes[1]
ax.fill_between(xvals, 0, scaled, color=C_RAW, alpha=0.3, linewidth=0)
ax.plot(xvals, edf["rate_roll_50"], color=C_R50, lw=1.2, alpha=0.8, label="Roll 50 (10s)")
ax.plot(xvals, edf["rate_roll_200"], color=C_R200, lw=1.8, label="Roll 200 (40s)")
ax.axhline(args.mu_min, color=C_GRAY, ls="--", alpha=0.5)
ax.axhline(args.mu_max, color=C_GRAY, ls="--", alpha=0.5)
ax.set_ylabel("Requests / sec", fontsize=12)
ax.set_title(f"(b) Rate-Scaled to [{args.mu_min:.0f}, {args.mu_max:.0f}] req/s",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim(xvals[0], xvals[-1])
ax.set_ylim(0, args.mu_max * 1.3)
ax.grid(axis="y", alpha=0.3)

# Panel C: Long-term trend only (heavy smoothing)
ax = axes[2]
ax.plot(xvals, edf["rate_roll_200"], color=C_R200, lw=2.5, label="Roll 200 (40s)")
ax.fill_between(xvals,
                edf["rate_roll_200"] - scaled.std() * 0.5,
                edf["rate_roll_200"] + scaled.std() * 0.5,
                color=C_R200, alpha=0.15, label="±0.5σ band")
ax.axhline(args.mu_min, color=C_GRAY, ls="--", alpha=0.5)
ax.axhline(args.mu_max, color=C_GRAY, ls="--", alpha=0.5)
ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel("Requests / sec", fontsize=12)
ax.set_title("(c) Long-Term Demand Trend (Smoothed)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim(xvals[0], xvals[-1])
ax.set_ylim(args.mu_min * 0.5, args.mu_max * 1.3)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout(h_pad=2)
out_overview = f"{args.out_prefix}_overview.png"
fig.savefig(out_overview, dpi=150, bbox_inches="tight")
print(f"Saved: {out_overview}")

# ── Plot 2: Zoomed window ───────────────────────────────────────────
zoom_frac = 0.05  # first 5%
zoom_n = max(200, int(n_epochs * zoom_frac))
zoom_x = xvals[:zoom_n]
zoom_raw = epoch_counts[:zoom_n]
zoom_scaled = scaled[:zoom_n]
zoom_roll = edf["rate_roll_20"].values[:zoom_n]

fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(16, 8))

ax2a.bar(zoom_x, zoom_raw, width=(xvals[1] - xvals[0]) * 0.85,
         color=C_RAW, alpha=0.6, label="Raw count")
ax2a.plot(zoom_x, edf["raw_roll_20"].values[:zoom_n],
          color=C_R200, lw=2, label="Roll 20")
ax2a.set_ylabel("Objects / epoch", fontsize=12)
ax2a.set_title("Zoomed: Raw Object Count", fontsize=13, fontweight="bold")
ax2a.legend(fontsize=10)
ax2a.set_xlim(zoom_x[0], zoom_x[-1])
ax2a.set_ylim(bottom=0)
ax2a.grid(axis="y", alpha=0.3)

ax2b.bar(zoom_x, zoom_scaled, width=(xvals[1] - xvals[0]) * 0.85,
         color=C_RAW, alpha=0.6, label="Scaled rate")
ax2b.plot(zoom_x, zoom_roll, color=C_R200, lw=2, label="Roll 20")

# ── 50s Average Dots ──────────────────────────────────────────────
win_epochs = int(50.0 / (args.epoch_ms / 1000.0))
if win_epochs > 0:
    n_win = len(zoom_scaled) // win_epochs
    if n_win > 0:
        dot_idx = np.arange(n_win) * win_epochs + (win_epochs // 2)
        dot_x = zoom_x[dot_idx]
        dot_y = [zoom_scaled[i*win_epochs:(i+1)*win_epochs].mean() for i in range(n_win)]
        ax2b.scatter(dot_x, dot_y, color=C_R50, s=50, edgecolors="white", lw=1, 
                     zorder=10, label="50s Window Avg")

ax2b.set_xlabel(xlabel, fontsize=12)
ax2b.set_ylabel("Requests / sec", fontsize=12)
ax2b.set_title("Zoomed: Scaled Request Rate", fontsize=13, fontweight="bold")
ax2b.legend(fontsize=10)
ax2b.set_xlim(zoom_x[0], zoom_x[-1])
ax2b.set_ylim(0, args.mu_max * 1.3)
ax2b.grid(axis="y", alpha=0.3)

fig2.tight_layout(h_pad=2)
out_zoom = f"{args.out_prefix}_zoom.png"
fig2.savefig(out_zoom, dpi=150, bbox_inches="tight")
print(f"Saved: {out_zoom}")

# ── Plot 3: Distribution ────────────────────────────────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

ax3a.hist(epoch_counts, bins=60, color=C_R20, alpha=0.75, edgecolor="white", lw=0.5)
ax3a.axvline(epoch_counts.mean(), color=C_R50, ls="--", lw=2,
             label=f"Mean = {epoch_counts.mean():.2f}")
ax3a.set_xlabel("Objects / epoch", fontsize=12)
ax3a.set_ylabel("Frequency", fontsize=12)
ax3a.set_title("Raw Count Distribution", fontsize=13, fontweight="bold")
ax3a.legend(fontsize=10)

ax3b.hist(scaled, bins=60, color=C_R20, alpha=0.75, edgecolor="white", lw=0.5)
ax3b.axvline(scaled.mean(), color=C_R50, ls="--", lw=2,
             label=f"Mean = {scaled.mean():.1f}")
ax3b.axvline(args.mu_min, color=C_GRAY, ls=":", lw=1.5, label=f"μ_min = {args.mu_min:.0f}")
ax3b.axvline(args.mu_max, color=C_GRAY, ls=":", lw=1.5, label=f"μ_max = {args.mu_max:.0f}")
ax3b.set_xlabel("Requests / sec", fontsize=12)
ax3b.set_ylabel("Frequency", fontsize=12)
ax3b.set_title("Scaled Rate Distribution", fontsize=13, fontweight="bold")
ax3b.legend(fontsize=10)

fig3.tight_layout()
out_dist = f"{args.out_prefix}_distribution.png"
fig3.savefig(out_dist, dpi=150, bbox_inches="tight")
print(f"Saved: {out_dist}")

print(f"\nDone. All outputs saved with prefix '{args.out_prefix}_'")
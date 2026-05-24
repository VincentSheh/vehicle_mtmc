#!/usr/bin/env python3
"""
Process ONE shard of Google Borg v2019 collection_events.
Downloads nothing — just point it at the .json.gz you already downloaded.

Usage:
  wget https://storage.googleapis.com/clusterdata_2019_a/collection_events-000000000000.json.gz
  python process_borg_shard.py collection_events-000000000000.json.gz
"""
import sys, os, gzip, json
from collections import defaultdict
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    os.system(f"{sys.executable} -m pip install matplotlib -q")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_borg_shard.py <collection_events-*.json.gz>")
        print("\nDownload one shard first:")
        print("  wget https://storage.googleapis.com/clusterdata_2019_a/collection_events-000000000000.json.gz")
        sys.exit(1)

    fpath = sys.argv[1]
    bin_width = 60  # seconds

    # ── Parse ──────────────────────────────────────────────────────
    print(f"Parsing {fpath} ...")
    job_submits = {}  # collection_id -> time_seconds
    total = 0
    skipped = 0

    opener = gzip.open if fpath.endswith(".gz") else open
    with opener(fpath, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Filter: type=0 (SUBMIT), collection_type=0 (job), time>0
            if int(rec.get("type", -1)) != 0:
                continue
            if int(rec.get("collection_type", -1)) != 0:
                continue

            ts_us = rec.get("time")
            cid = rec.get("collection_id")
            if ts_us is None or cid is None:
                continue

            ts_s = int(ts_us) / 1e6
            if ts_s <= 0:
                continue

            cid = str(cid)
            if cid not in job_submits or ts_s < job_submits[cid]:
                job_submits[cid] = ts_s

            if total % 500_000 == 0:
                print(f"  {total:,} lines, {len(job_submits):,} job SUBMITs so far")

    print(f"Done: {total:,} lines, {skipped:,} skipped, {len(job_submits):,} unique job SUBMITs")

    if not job_submits:
        print("No job SUBMIT events found. Check your file.")
        sys.exit(1)

    # ── Bin ────────────────────────────────────────────────────────
    arrivals = defaultdict(int)
    for ts in job_submits.values():
        arrivals[int(ts // bin_width) * bin_width] += 1

    times = np.array(sorted(arrivals.keys()))
    counts = np.array([arrivals[t] for t in times])
    times_h = times / 3600.0
    rate = counts / bin_width

    # ── Stats ──────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Time span:      {times_h[0]:.1f} – {times_h[-1]:.1f} hours")
    print(f"Total jobs:     {int(counts.sum()):,}")
    print(f"Mean rate:      {rate.mean():.2f} jobs/s")
    print(f"Peak rate:      {rate.max():.2f} jobs/s")
    print(f"Median rate:    {np.median(rate):.2f} jobs/s")
    print(f"{'─'*50}")

    # ── Export CSV ─────────────────────────────────────────────────
    csv_path = fpath.replace(".json.gz", "_job_arrivals.csv").replace(".json", "_job_arrivals.csv")
    with open(csv_path, "w") as f:
        f.write("time_s,time_h,job_count,rate_per_s\n")
        for t, c, r in zip(times, counts, rate):
            f.write(f"{t},{t/3600:.4f},{c},{r:.4f}\n")
    print(f"Saved arrival trace: {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="white")

    # Top: arrival rate
    ax = axes[0]
    window = max(1, 300 // bin_width)
    smoothed = np.convolve(rate, np.ones(window)/window, mode="same")
    ax.fill_between(times_h, 0, rate, alpha=0.15, color="#2563EB")
    ax.plot(times_h, smoothed, linewidth=1, color="#2563EB",
            label=f"{window*bin_width//60}-min moving avg")
    ax.set_ylabel("Job submission rate (jobs/s)")
    ax.set_title("Google Borg v2019 — Job Arrivals (collection_events, type=SUBMIT, jobs only)",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(times_h[0], times_h[-1])

    # Day markers
    for d in range(1, int(times_h[-1] // 24) + 1):
        ax.axvline(d * 24, color="#CBD5E1", linestyle="--", linewidth=0.7)

    # Bottom: cumulative
    ax2 = axes[1]
    cum = np.cumsum(counts)
    ax2.plot(times_h, cum, linewidth=1.5, color="#0F766E")
    ax2.fill_between(times_h, 0, cum, alpha=0.1, color="#0F766E")
    ax2.set_xlabel("Time (hours from trace start)")
    ax2.set_ylabel("Cumulative job count")
    ax2.set_title("Cumulative Job Arrivals")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(times_h[0], times_h[-1])

    plt.tight_layout()
    plot_path = fpath.replace(".json.gz", "_job_arrivals.png").replace(".json", "_job_arrivals.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot:          {plot_path}")


if __name__ == "__main__":
    main()
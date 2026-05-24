import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Attempt to get the directory of the current script
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

# Add parent directory to path to import request.py from the root
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, "..")))
from request import User

def visualize_user_trace(csv_path, t_max=10000, slot_ms=200, target_avg=None, arrival_col="raw_count", seed=None):
    if not os.path.exists(csv_path):
        # Fallback to checking in the current working directory if relative path fails
        alt_path = os.path.join(os.getcwd(), csv_path)
        if not os.path.exists(alt_path):
            print(f"Error: {csv_path} not found.")
            return
        csv_path = alt_path

    # Initialize User in trace mode
    # User class now handles target_avg range sampling internally
    user = User(
        user_id="viz_user",
        slot_ms=slot_ms,
        t_max=t_max,
        seed=seed,
        source_mode="trace",
        csv_path=csv_path,
        arrival_col=arrival_col,
        random_slice=True,
        target_avg=target_avg
    )

    print(f"Loading trace from {csv_path} (col={arrival_col}, target_avg={user.target_avg:.2f})...")


    # Collect requests over t_max steps
    requests = []
    for t in range(t_max):
        requests.append(user.num_requests_at(t))
    
    requests = np.array(requests)
    time_sec = np.arange(t_max) * (slot_ms / 1000.0)

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(time_sec, requests, label='Requests per Step', alpha=0.6, color='#a3c4dc')
    
    # Add a rolling average (50s window) for clarity
    win_steps = int(50.0 / (slot_ms / 1000.0))
    if win_steps > 1:
        rolling_avg = pd.Series(requests).rolling(window=win_steps, center=True).mean()
        plt.plot(time_sec, rolling_avg, label='50s Rolling Mean', color='#e07b54', lw=2)

    plt.title(f"User Request Trace Visualization: {os.path.basename(csv_path)}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of Requests (per step)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_png = os.path.join(output_dir, "user_trace_visualization.png")
    
    plt.savefig(output_png, dpi=150)
    print(f"Visualization saved to {output_png}")
    plt.show() 
    
    # Print statistics
    print(f"\nTrace Statistics (Steps: {t_max}, Slot: {slot_ms}ms):")
    print(f"  Min:  {np.min(requests)}")
    print(f"  Max:  {np.max(requests)}")
    print(f"  Mean: {np.mean(requests):.2f}")
    print(f"  Std:  {np.std(requests):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize request trace using User class")
    parser.add_argument("--csv", type=str, default="trace_arrival_rate.csv", help="Path to trace CSV")
    parser.add_argument("--t_max", type=int, default=30000, help="Number of steps to visualize")
    parser.add_argument("--slot_ms", type=float, default=200.0, help="Slot duration in ms")
    parser.add_argument("--target_min", type=float, default=80.0, help="Min target average (req/step)")
    parser.add_argument("--target_max", type=float, default=160.0, help="Max target average (req/step)")
    parser.add_argument("--col", type=str, default="raw_count", help="Column name in CSV")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for determinism (optional)")
    args = parser.parse_args()


    visualize_user_trace(
        csv_path=args.csv,
        t_max=args.t_max,
        slot_ms=args.slot_ms,
        target_avg=[args.target_min, args.target_max],
        arrival_col=args.col,
        seed=args.seed
    )


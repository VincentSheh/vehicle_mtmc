from mot.run_tracker import run_single_experiment
from config.defaults import get_cfg_defaults
from config.config_tools import expand_relative_paths
import os
import json
import csv

OUT_DIR = "output/bellevue_150th_eastgate_nomask/"
FRAME_STATS_DIR = OUT_DIR + "2017-09-11_07-08-31"
CSV_PATH = os.path.join(OUT_DIR, "results.csv")
os.makedirs(FRAME_STATS_DIR, exist_ok=True)

CSV_FIELDS = [
    "experiment_id",
    "detector",
    "base_resolution_w",
    "base_resolution_h",
    "reid_object_w",
    "reid_object_h",
    "IDF1",
    "MOTA",
    "latency_detection_filter_ms",
    "latency_reid_ms",
]
os.makedirs(OUT_DIR, exist_ok=True)
def fmt(x):
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x:.5f}"
    return x

def append_csv_row(row: dict):
    file_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def write_frame_stats_csv(frame_stats, exp_id):
    """
    Write per-frame stats to CSV.
    """
    out_path = os.path.join(FRAME_STATS_DIR, f"frame_stats_{exp_id}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "num_objects",
                "detector_latency_ms",
                "reid_latency_ms",
            ],
        )
        writer.writeheader()

        for row in frame_stats:
            writer.writerow({
                "frame": row["frame"],
                "num_objects": row["num_objects"],
                "detector_latency_ms": round(row["detector_latency_ms"], 5),
                "reid_latency_ms": round(row["reid_latency_ms"], 5),
            })

def run_experiment(overrides: dict):
    cfg = get_cfg_defaults()

    # load base YAML
    cfg.merge_from_file("config/bellevue_150th_eastgate/Bellevue_150th_Eastgate__2017-09-11_07-08-31.yaml")

    cfg.defrost()

    # apply overrides
    for k, v in overrides.items():
        if k.startswith("MOT."):
            setattr(cfg.MOT, k.replace("MOT.", ""), v)
        elif k == "OUTPUT_DIR":
            # cfg.OUTPUT_DIR = v
            pass
        else:
            raise ValueError(f"Unknown override key: {k}")

    cfg = expand_relative_paths(cfg)
    cfg.freeze()

    # profiling-safe call
    return run_single_experiment(cfg)
    
experiments = [
    {
        "MOT.DETECTOR": "yolo11x",
        "MOT.BASE_RESOLUTION": [1280, 736],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11x",
        "MOT.BASE_RESOLUTION": [960, 544],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11x",
        "MOT.BASE_RESOLUTION": [640, 384],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11l",
        "MOT.BASE_RESOLUTION": [1280, 736],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11l",
        "MOT.BASE_RESOLUTION": [960, 544],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11l",
        "MOT.BASE_RESOLUTION": [640, 384],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11l",
        "MOT.BASE_RESOLUTION": [1280, 736],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11l",
        "MOT.BASE_RESOLUTION": [960, 544],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11l",
        "MOT.BASE_RESOLUTION": [640, 384],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11m",
        "MOT.BASE_RESOLUTION": [1280, 736],
        "MOT.REID_OBJECT_SIZE": [64,64],
    },
    {
        "MOT.DETECTOR": "yolo11m",
        "MOT.BASE_RESOLUTION": [960, 544],
        "MOT.REID_OBJECT_SIZE": [64,64],
    },
    {
        "MOT.DETECTOR": "yolo11m",
        "MOT.BASE_RESOLUTION": [640, 384],
        "MOT.REID_OBJECT_SIZE": [64,64],
    },
    {
        "MOT.DETECTOR": "yolo11m",
        "MOT.BASE_RESOLUTION": [1280, 736],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11m",
        "MOT.BASE_RESOLUTION": [960, 544],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11m",
        "MOT.BASE_RESOLUTION": [640, 384],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11s",
        "MOT.BASE_RESOLUTION": [1280, 736],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11s",
        "MOT.BASE_RESOLUTION": [960, 544],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11s",
        "MOT.BASE_RESOLUTION": [640, 384],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11n",
        "MOT.BASE_RESOLUTION": [1280, 736],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11n",
        "MOT.BASE_RESOLUTION": [960, 544],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    {
        "MOT.DETECTOR": "yolo11n",
        "MOT.BASE_RESOLUTION": [640, 384],
        "MOT.REID_OBJECT_SIZE": [18,18],
    },
    
]
results = []

for i, exp in enumerate(experiments):
    metrics = run_experiment(exp)
    results.append(metrics)

    row = {
        "experiment_id": i,
        "detector": exp["MOT.DETECTOR"],
        "base_resolution_w": exp["MOT.BASE_RESOLUTION"][0],
        "base_resolution_h": exp["MOT.BASE_RESOLUTION"][1],
        "reid_object_w": exp["MOT.REID_OBJECT_SIZE"][0] if exp["MOT.REID_OBJECT_SIZE"] else -1,
        "reid_object_h": exp["MOT.REID_OBJECT_SIZE"][1] if exp["MOT.REID_OBJECT_SIZE"] else -1,
        "IDF1": fmt(metrics.get("IDF1")),
        "MOTA": fmt(metrics.get("MOTA")),
        "latency_detection_filter_ms": fmt(metrics["latency_ms"].get("detection_filter")),
        "latency_reid_ms": fmt(metrics["latency_ms"].get("reid")),
    }
    append_csv_row(row)
    write_frame_stats_csv(metrics["frame_stats"], f"{exp['MOT.DETECTOR']}_{exp['MOT.BASE_RESOLUTION'][1]}")    
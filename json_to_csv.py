import os
import json
import csv

JSON_DIR = "output/temp"
CSV_PATH = "output/temp/results_from_json.csv"

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

def fmt(x):
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x:.5f}"
    return x

json_files = sorted(
    f for f in os.listdir(JSON_DIR)
    if f.endswith(".json")
)

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    print(json_files)

    for fname in json_files:
        path = os.path.join(JSON_DIR, fname)
        with open(path, "r") as jf:
            data = json.load(jf)

        cfg = data.get("config", {})
        metrics = data.get("metrics", {})
        latency = metrics.get("latency_ms", {})

        base_res = cfg.get("MOT.BASE_RESOLUTION", [-1, -1])
        reid_sz = cfg.get("MOT.REID_OBJECT_SIZE", [])

        row = {
            "experiment_id": data.get("experiment_id"),
            "detector": cfg.get("MOT.DETECTOR"),
            "base_resolution_w": base_res[0],
            "base_resolution_h": base_res[1],
            "reid_object_w": reid_sz[0] if reid_sz else -1,
            "reid_object_h": reid_sz[1] if reid_sz else -1,
            "IDF1": fmt(metrics.get("IDF1")),
            "MOTA": fmt(metrics.get("MOTA")),
            "latency_detection_filter_ms": fmt(latency.get("detection_filter")),
            "latency_reid_ms": fmt(latency.get("reid")),
        }

        writer.writerow(row)

print(f"Converted {len(json_files)} JSON files to CSV with 5-decimal precision:")
print(CSV_PATH)
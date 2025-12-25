import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Union
import re

class Attacker:
    """
    Attacker time series df must include at least:
      - t (int)
      - attack_type (str)
      - lambda_req (float)  (requests per step or per second, treat as "rate")
    Optional:
      - cpu_per_req, mem_per_req, uplink_per_req (not required for this minimal emu)
    """
    def __init__(self, attacker_id: str, df: pd.DataFrame):
        self.attacker_id = attacker_id
        self.df = df.copy()
        if "t" not in self.df.columns:
            raise ValueError("Attacker df missing column 't'")
        if "attack_type" not in self.df.columns:
            raise ValueError("Attacker df missing column 'attack_type'")
        if "lambda_req" not in self.df.columns:
            raise ValueError("Attacker df missing column 'lambda_req'")

    def load_at(self, t: int) -> pd.DataFrame:
        return self.df[self.df["t"] == t].copy()


def _parse_detector_res_from_filename(p: Path) -> Tuple[str, int]:
    """
    Expected patterns like:
      frame_stats_yolo11l_384.csv
      frame_stats_yolo11x-720.csv
      obj_pred_yolo11s_736.csv

    Returns: (detector, base_resolution_h)
    """
    stem = p.stem.lower()
    # find yolo11? token and a following integer
    import re
    m = re.search(r"(yolo11[a-z])[_\-]?(\d+)", stem)
    if not m:
        raise ValueError(f"Cannot parse (detector, h) from filename: {p.name}")
    det = m.group(1)
    h = int(m.group(2))
    return det, h


class UserCamera:
    """
    UserCamera with configuration-aware object predictions.

    input_dir must contain:
      - frame_stats_*.csv          (object predictions per configuration)
      - mota_table.csv             (MOTA lookup table)

    Object prediction CSV format:
      - frame (int)
      - num_objects (int / float)

    Filename format:
      frame_stats_<detector>_<h>.csv
      frame_stats_yolo11l_384.csv
      frame_stats_yolo11x-736.csv

    MOTA table format:
      - detector
      - base_resolution_h
      - MOTA
    """

    def __init__(
        self,
        user_id: str,
        input_dir: str | Path,
    ):
        self.user_id = str(user_id)
        input_dir = Path(input_dir)

        if not input_dir.is_dir():
            raise ValueError(f"input_dir must be a directory: {input_dir}")

        # =================================================
        # Load object prediction CSVs
        # =================================================
        self._obj_lookup: Dict[Tuple[str, int], Dict[int, float]] = {}

        obj_files = sorted(input_dir.glob("frame_stats_*.csv"))
        if not obj_files:
            raise ValueError(f"No frame_stats_*.csv found in {input_dir}")

        for csv_path in obj_files:
            detector, h = self._parse_detector_res(csv_path.name)

            df = pd.read_csv(csv_path)
            if not {"frame", "num_objects"}.issubset(df.columns):
                raise ValueError(
                    f"{csv_path.name} must contain columns: frame, num_objects"
                )

            df = df[["frame", "num_objects"]].dropna().sort_values("frame")

            self._obj_lookup[(detector, h)] = {
                int(r.frame): float(r.num_objects)
                for r in df.itertuples(index=False)
            }

        # =================================================
        # Load MOTA table
        # =================================================
        mota_path = input_dir / "mota_table.csv"
        if not mota_path.exists():
            raise ValueError(f"Missing mota_table.csv in {input_dir}")

        mota_df = pd.read_csv(mota_path)

        required = {"detector", "base_resolution_h", "MOTA"}
        if not required.issubset(mota_df.columns):
            raise ValueError(f"mota_table.csv must contain columns {required}")

        self._mota_lookup: Dict[Tuple[str, int], float] = {
            (str(r.detector), int(r.base_resolution_h)): float(r.MOTA)
            for r in mota_df.itertuples(index=False)
        }

    # =================================================
    # Public API
    # =================================================

    def get_num_objects(self, t: int, detector: str, base_resolution_h: int) -> float:
        return self._obj_lookup.get(
            (str(detector), int(base_resolution_h)), {}
        ).get(int(t), 0.0)

    def get_mota(self, detector: str, base_resolution_h: int) -> float:
        return self._mota_lookup.get((str(detector), int(base_resolution_h)), 0.0)

    # =================================================
    # Helpers
    # =================================================

    @staticmethod
    def _parse_detector_res(filename: str) -> Tuple[str, int]:
        """
        Extract (detector, resolution_h) from filename.
        """
        m = re.search(r"(yolo11[a-z])[_\-]?(\d+)", filename.lower())
        if not m:
            raise ValueError(
                f"Cannot parse detector & resolution from filename: {filename}"
            )
        return m.group(1), int(m.group(2))
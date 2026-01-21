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
    def __init__(
        self,
        attacker_id,
        attack_type,
        ts_df,
        cpu_usage_const,
        non_defendable_bw_const,
        scaling,
        slot_ms,
        t_max,
        seed,
    ):
        self.attacker_id = attacker_id
        self.attack_type = attack_type
        self.cpu_usage_const = cpu_usage_const
        self.non_defendable_bw_const = non_defendable_bw_const
        self.slot_ms = slot_ms
        self.t_max=t_max
        
        # -----------------------------
        # Prepare dataframe
        # -----------------------------
        self.df = ts_df.reset_index(drop=True).copy()
        self.df["attack_type"] = attack_type 
        self.df["attacker_id"] = attacker_id 
        self.df["forward_bytes_per_sec"] = self.df["forward_bytes_per_sec"] / (1024*1024)

        required = {
            "forward_packets_per_sec",
            "forward_bytes_per_sec",
            "flows_per_sec",
            "attack_type",
            "attacker_id",
        }
        if not required.issubset(self.df.columns):
            raise ValueError(f"Attack trace missing columns: {required}")        

        step_scale = slot_ms / 1000.0  # seconds per step
        for col in (
            "flows_per_sec",
            "forward_packets_per_sec",
            "forward_bytes_per_sec",
        ):
            self.df[col] *= step_scale        
            self.df[col] *= scaling        

        # -----------------------------
        # Unit conversions
        # -----------------------------
        self.steps_per_sec = int(1000 // slot_ms)

        trace_len_sec = len(self.df)
        if trace_len_sec <= 0:
            raise ValueError("Empty attack trace")

        trace_len_steps = trace_len_sec * self.steps_per_sec

        if trace_len_steps > t_max:
            raise ValueError(
                f"Attack trace longer than episode: "
                f"{trace_len_steps} > {t_max}"
            )     
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self._init_start()
    def _init_start(self):
        trace_len_steps = len(self.df) * self.steps_per_sec
        max_start = self.t_max - trace_len_steps
        self.start = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_start()        

    def load_at(self, t: int) -> pd.DataFrame:
        """
        Map step t → second index in attack trace.
        """
        local_step = t - self.start
        if local_step < 0:
            return pd.DataFrame(columns=self.df.columns)

        sec_idx = local_step // self.steps_per_sec
        if sec_idx >= len(self.df):
            return pd.DataFrame(columns=self.df.columns)

        return self.df.iloc[[sec_idx]].copy()


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
    def __init__(
        self,
        user_id: str,
        input_dir: str | Path,
        t_max: int,
        seed: int,
    ):
        self.user_id = str(user_id)
        self.input_dir = Path(input_dir)
        self.t_max = int(t_max)

        if not self.input_dir.is_dir():
            raise ValueError(f"input_dir must be a directory: {self.input_dir}")

        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)

        self._obj_lookup: Dict[Tuple[str, int], Dict[int, float]] = {}
        self._mota_lookup: Dict[Tuple[str, int], float] = {}

        self._load_mota_table()
        self._build_obj_lookup()
        self._set_start()
        
    def _load_mota_table(self):
        """
        Load MOTA lookup table.

        Expected CSV format:
        - detector (str)
        - base_resolution_h (int)
        - MOTA (float)

        Builds:
        self._mota_lookup[(detector, base_resolution_h)] -> MOTA
        """
        mota_path = self.input_dir / "mota_table.csv"
        if not mota_path.exists():
            raise ValueError(f"Missing mota_table.csv in {self.input_dir}")

        mota_df = pd.read_csv(mota_path)

        required = {"detector", "base_resolution_h", "MOTA"}
        if not required.issubset(mota_df.columns):
            raise ValueError(
                f"mota_table.csv must contain columns {required}, "
                f"got {set(mota_df.columns)}"
            )

        # clean + type normalize
        mota_df = mota_df[["detector", "base_resolution_h", "MOTA"]].dropna()
        mota_df["detector"] = mota_df["detector"].astype(str)
        mota_df["base_resolution_h"] = mota_df["base_resolution_h"].astype(int)
        mota_df["MOTA"] = mota_df["MOTA"].astype(float)

        # ensure uniqueness
        if mota_df.duplicated(subset=["detector", "base_resolution_h"]).any():
            dups = mota_df[mota_df.duplicated(
                subset=["detector", "base_resolution_h"], keep=False
            )]
            raise ValueError(
                "Duplicate (detector, base_resolution_h) entries in mota_table.csv:\n"
                f"{dups}"
            )

        self._mota_lookup = {
            (row.detector, row.base_resolution_h): row.MOTA
            for row in mota_df.itertuples(index=False)
        }        
        
    def _build_obj_lookup(self):
        """
        Build object count lookup for each (detector, resolution).
        Randomly selects a contiguous window of length t_max
        using the instance RNG.
        """
        self._obj_lookup.clear()

        obj_files = sorted(self.input_dir.glob("frame_stats_*.csv"))
        if not obj_files:
            raise ValueError(f"No frame_stats_*.csv found in {self.input_dir}")

        for csv_path in obj_files:
            detector, h = self._parse_detector_res(csv_path.name)

            df = pd.read_csv(csv_path)
            if not {"frame", "num_objects"}.issubset(df.columns):
                raise ValueError(
                    f"{csv_path.name} must contain columns: frame, num_objects"
                )

            df = (
                df[["frame", "num_objects"]]
                .dropna()
                .sort_values("frame")
                .reset_index(drop=True)
            )

            if len(df) <= self.t_max:
                start = 0
                df_slice = df
            else:
                max_start = len(df) - self.t_max
                start = int(self.rng.integers(0, max_start + 1))
                df_slice = df.iloc[start : start + self.t_max]

            df_slice = df_slice.copy()
            df_slice["frame"] = np.arange(len(df_slice))  # reindex 0..t_max-1

            self._obj_lookup[(detector, h)] = {
                int(r.frame): float(r.num_objects)
                for r in df_slice.itertuples(index=False)
            }
    def _set_start(self):
        """
        Random start offset for this user within an episode.
        """
        if self.t_max <= 0:
            self.start = 0
            return

        max_start = self.t_max
        self.start = int(self.rng.integers(0, max_start))
        
    def reset(self, seed: int | None = None):
        """
        Reset user randomness for a new episode.
        """
        if seed is not None:
            self.base_seed = int(seed)
            self.rng = np.random.default_rng(self.base_seed)

        self._build_obj_lookup()
        self._set_start()                    
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
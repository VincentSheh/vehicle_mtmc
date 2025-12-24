import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

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


# =========================
# User (Camera)
# =========================

class UserCamera:
    """
    obj_pred_ts must include:
      - t (int)
      - detector (str)
      - base_resolution_h (int)
      - num_objects (float or int)
    mota_table must include either:
      A) columns: detector, base_resolution_h, MOTA
      or
      B) columns: detector, base_resolution_w, base_resolution_h, MOTA
    """
    def __init__(self, user_id: str, obj_pred_ts: pd.DataFrame, mota_table: pd.DataFrame):
        self.user_id = user_id
        self.obj_pred_ts = obj_pred_ts.copy()
        self.mota_table = mota_table.copy()

        for col in ["t", "detector", "base_resolution_h", "num_objects"]:
            if col not in self.obj_pred_ts.columns:
                raise ValueError(f"obj_pred_ts missing column '{col}'")

        if "MOTA" not in self.mota_table.columns:
            raise ValueError("mota_table missing column 'MOTA'")
        if "detector" not in self.mota_table.columns:
            raise ValueError("mota_table missing column 'detector'")
        if "base_resolution_h" not in self.mota_table.columns:
            raise ValueError("mota_table missing column 'base_resolution_h'")

        # Pre-index MOTA for O(1) lookup
        self._mota_lookup: Dict[Tuple[str, int], float] = {}
        for _, r in self.mota_table.iterrows():
            key = (str(r["detector"]), int(r["base_resolution_h"]))
            self._mota_lookup[key] = float(r["MOTA"])

    def get_num_objects(self, t: int, detector: str, base_resolution_h: int) -> float:
        rows = self.obj_pred_ts[
            (self.obj_pred_ts["t"] == t)
            & (self.obj_pred_ts["detector"] == detector)
            & (self.obj_pred_ts["base_resolution_h"] == base_resolution_h)
        ]
        if rows.empty:
            return 0.0
        return float(rows.iloc[0]["num_objects"])

    def get_mota(self, detector: str, base_resolution_h: int) -> float:
        return float(self._mota_lookup.get((detector, int(base_resolution_h)), 0.0))

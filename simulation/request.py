import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Union
import re
from dataclasses import dataclass
from typing import Optional, Union

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
        latency_per_flow,
        bw_per_flow,
        non_defendable_bw_const,
        scaling,
        slot_ms,
        t_max,
        seed,
        cpu_cycle_per_ms: float,
        cpu_cores: int,        
    ):
        self.attacker_id = attacker_id
        self.attack_type = attack_type
        self.latency_per_flow = latency_per_flow
        self.cycle_per_flow = latency_per_flow * float(cpu_cycle_per_ms) * int(cpu_cores)
        self.bw_per_flow = bw_per_flow
        self.non_defendable_bw_const = non_defendable_bw_const
        self.slot_ms = slot_ms
        self.t_max=t_max
        
        # -----------------------------
        # Prepare dataframe
        # -----------------------------
        self.df = ts_df.reset_index(drop=True).copy()
        self.df["attack_type"] = attack_type 
        self.df["attacker_id"] = attacker_id 
        self.df["forward_bytes_per_sec"] = self.df["forward_bytes_per_sec"] / (1024*1024) * scaling
        self.df["flows_per_sec"] = self.df["flows_per_sec"] * scaling

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



class User:
    """
    User request arrival time series.
    Input CSV is one row per second.
    """

    def __init__(
        self,
        user_id: Union[str, int],
        slot_ms: float,
        t_max: int,
        seed: int = 0,
        scaling: float = 1.0,
        arrival_col: str = "num_objects",   # rename later if you want
        synth_cfg=None,
    ):
        self.user_id = str(user_id)
        self.slot_ms = float(slot_ms)
        self.t_max = int(t_max)
        self.scaling = float(scaling)
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)

        self.arrival_col = str(arrival_col)
        
        self.synth_cfg = synth_cfg
        self._init_from_synthetic()        

    def _init_from_synthetic(self):
        cfg = self.synth_cfg
        self.steps_per_sec = int(round(1000.0 / self.slot_ms))
        if self.steps_per_sec <= 0:
            raise ValueError(f"Invalid slot_ms={self.slot_ms}")

        df = self.generate_req_trace(
            t_steps=self.t_max,
            slot_ms=self.slot_ms,
            rng=self.rng,                # key: uses User rng so reset controls mu0
            rw_sigma_per_sqrt_sec=cfg["rw_sigma_per_sqrt_sec"],
            mu_min=cfg["mu_min"],
            mu_max=cfg["mu_max"],
            kappa=cfg["kappa"],
            sigma=cfg["sigma"],
        )

        per_step = np.maximum(df["num_requests_per_step"].to_numpy(dtype=float) * self.scaling, 0.0)

        self.df = pd.DataFrame(
            {
                "t": np.arange(self.t_max, dtype=int),
                "user_id": self.user_id,
                "num_requests_per_step": per_step,
                "mu0": df["mu0"].to_numpy(dtype=float),
                "mu_t": df["mu_t"].to_numpy(dtype=float),
                "req_per_sec": df["req_per_sec"].to_numpy(dtype=float),
                "req_per_step_expected": df["req_per_step_expected"].to_numpy(dtype=float) * self.scaling,
            }
        )

    def generate_req_trace(
        self,
        t_steps: int,
        slot_ms: float,
        rng: np.random.Generator,

        # random-walk mean params
        rw_sigma_per_sqrt_sec: float = 0.8,
        mu_min: float = 5.0,
        mu_max: float = 40.0,

        # OU-like arrival params
        kappa: float = 0.02,
        sigma: float = 0.9,
    ):
        dt = slot_ms / 1000.0

        # 1) randomize mu0
        mu0 = float(rng.uniform(mu_min, mu_max))

        # 2) make mu random walk
        mu_series = np.empty(t_steps, dtype=float)
        mu_series[0] = mu0
        for t in range(1, t_steps):
            step =  rw_sigma_per_sqrt_sec * np.sqrt(dt) * rng.standard_normal()
            mu_series[t] = np.clip(mu_series[t - 1] + step, mu_min, mu_max)

        # 3) generate arrival rate with mean-reverting dynamics toward mu_t
        x = float(mu_series[0])
        req_per_sec = np.zeros(t_steps, dtype=float)

        for t in range(t_steps):
            mu_t = float(mu_series[t])
            x = x + kappa * (mu_t - x) + sigma * rng.standard_normal()
            x = max(0.0, x)
            req_per_sec[t] = x

        # convert per-second rate to per-step expectation
        req_per_step_expected = req_per_sec * dt
        req_per_step = np.maximum(req_per_step_expected, 0.0)

        df = pd.DataFrame(
            {
                "t": np.arange(t_steps, dtype=int),
                "mu0": mu0,
                "mu_t": mu_series,
                "req_per_sec": req_per_sec,
                "req_per_step_expected": req_per_step_expected,
                "num_requests_per_step": req_per_step,
            }
        )
        return df        


    def reset(self, seed: int | None = None):
        if seed is not None:
            self.base_seed = int(seed)
            self.rng = np.random.default_rng(self.base_seed)
        self._init_from_synthetic()

    def load_at(self, t: int) -> pd.DataFrame:
        local_step = int(t)
        if local_step < 0:
            return pd.DataFrame(columns=self.df.columns.tolist())

        return self.df.iloc[[local_step]].copy()

    def num_requests_at(self, t: int) -> float:
        # direct per-step lookup
        if t < 0 or t >= len(self.df):
            return 0.0
        return float(self.df.iloc[int(t)]["num_requests_per_step"])


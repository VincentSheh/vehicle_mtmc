import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Union
import re
from dataclasses import dataclass
from typing import Optional, Union
import math

def sinus_local(t_local: np.ndarray, T: float) -> np.ndarray:
    return 0.5 * (1.0 + np.sin(2.0 * np.pi * t_local / T))

def expo_local(t_local: np.ndarray, T: float) -> np.ndarray:
    x = np.exp(-3.0 * t_local / T)
    x = (x - np.exp(-3.0)) / (1.0 - np.exp(-3.0))
    return np.clip(x, 0.0, 1.0)

def pulse_wave_local(t_local: np.ndarray, T: float) -> np.ndarray:
    return (t_local < T / 2).astype(float)

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
        base_scaling,
        mean_rep,
        non_defendable_bw_const,
        slot_ms,
        t_max,
        seed,
        cpu_cycle_per_ms: float,
        cpu_cores: int,        
        pattern_type: str = "sinus",
        t_min: float = 15.0,
        t_max_pattern: float = 45.0,
        smooth_window: int = 10,
    ):
        self.attacker_id = attacker_id
        self.attack_type = attack_type
        self.episode_active = True
        self.latency_per_flow = latency_per_flow
        self.cycle_per_flow = latency_per_flow * float(cpu_cycle_per_ms) * int(cpu_cores)
        self.bw_per_flow = bw_per_flow
        self.non_defendable_bw_const = non_defendable_bw_const
        self.slot_ms = slot_ms
        self.t_max=t_max
        self.mean_rep = mean_rep
        self.rep = 1
        self.scaling = 1
        self.z_t = 0
        self.tau = 0
        self.tau_threshold = 4000

        self.pattern_type = pattern_type
        self.t_min_pattern = t_min
        self.t_max_pattern = t_max_pattern
        self.smooth_window = smooth_window
        
        # -----------------------------
        # Prepare dataframe
        # -----------------------------
        self.df = ts_df.reset_index(drop=True).copy()
        self.df["attack_type"] = attack_type 
        self.df["attacker_id"] = attacker_id 
        self.df["forward_bytes_per_sec"] = self.df["forward_bytes_per_sec"] / (1024*1024)
        self.df["flows_per_sec"] = self.df["flows_per_sec"] * base_scaling

        required = {
            "forward_packets_per_sec",
            "forward_bytes_per_sec",
            "flows_per_sec",
            "attack_type",
            "attacker_id",
        }
        if not required.issubset(self.df.columns):
            raise ValueError(f"Attack trace missing columns: {required}")        

        if self.pattern_type != "trace":
            self._estimate_baseline()

        step_scale = slot_ms / 1000.0  # seconds per step
        for col in (
            "flows_per_sec",
            "forward_packets_per_sec",
            "forward_bytes_per_sec",
        ):
            self.df[col] *= step_scale        

        # -----------------------------
        # Unit conversions
        # -----------------------------
        self.steps_per_sec = int(1000 // slot_ms)

        trace_len_sec = len(self.df)
        if trace_len_sec <= 0:
            raise ValueError("Empty attack trace")

        trace_len_steps = trace_len_sec * self.steps_per_sec

        if self.pattern_type == "trace" and trace_len_steps > t_max:
            raise ValueError(
                f"Attack trace longer than episode: "
                f"{trace_len_steps} > {t_max}"
            )     
        pad_sec = 1000
        pad_df = pd.DataFrame({
            "flows_per_sec": np.zeros(pad_sec, dtype=np.float32),
            "forward_packets_per_sec": np.zeros(pad_sec, dtype=np.float32),
            "forward_bytes_per_sec": np.zeros(pad_sec, dtype=np.float32),
            "attack_type": self.attack_type,
            "attacker_id": self.attacker_id,
        })

        # self.df = pd.concat([self.df, pad_df], ignore_index=True)            
        # -----------------------------
        # Precompute EMA and momentum on the trace timeline
        # -----------------------------
        hl = float(50.0)
        ema_col = "flows_per_sec"
        if hl <= 0:
            # degenerate: EMA = signal, momentum = diff(signal)
            ema = self.df[ema_col].astype(float)
        else:
            alpha = 1.0 - math.exp(math.log(0.5) / hl)
            ema = self.df[ema_col].astype(float).ewm(alpha=alpha, adjust=False).mean()

        self.df["flows_per_sec_ema"] = ema
        # self.df["flows_per_sec_ema_mom"] = self.df[f"flows_per_sec_ema"].diff().fillna(0.0)
        # self.df["flows_per_sec_ema_mom"] = self.df[f"flows_per_sec_ema"].diff(hl).fillna(0.0) / (hl)          
        # mom_raw = self.df["flows_per_sec_ema"].diff().fillna(0.0)
        # self.df["flows_per_sec_ema_mom"] = mom_raw.ewm(alpha=alpha, adjust=False).mean()        
        self._flows = self.df["flows_per_sec"].to_numpy(dtype=np.float32)
        self._flows_ema = self.df["flows_per_sec_ema"].to_numpy(dtype=np.float32)
        # self._flows_ema_mom = self.df["flows_per_sec_ema_mom"].to_numpy(dtype=np.float32)

                
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self._init_start()

    def _estimate_baseline(self):
        flows = self.df["flows_per_sec"].values
        flows = flows[flows > 0]
        if len(flows) == 0:
            self.lambda_base = 1.0
            self.noise_std = 0.1
            return

        trend = (
            pd.Series(flows)
            .rolling(window=self.smooth_window, min_periods=1, center=True)
            .mean()
            .values
        )
        residual = flows / np.maximum(trend, 1e-6)
        self.lambda_base = float(np.mean(trend))
        self.noise_std = float(np.std(residual - 1.0))
        self.noise_std = min(self.noise_std, 1.0) / 2

    def _generate_patterned_trace(self):
        self.active_len = self.t_max // 2
        dt = self.slot_ms / 1000.0
        t_full = np.arange(0, self.active_len * dt, dt)
        g = np.zeros_like(t_full, dtype=float)
        
        pattern_fn = {
            "sinus": sinus_local,
            "expo": expo_local,
            "pw": pulse_wave_local
        }.get(self.pattern_type, sinus_local)

        cursor = 0.0
        total_time = self.active_len * dt
        while cursor < total_time:
            T_k = float(self.rng.uniform(self.t_min_pattern, self.t_max_pattern))
            end = min(cursor + T_k, total_time)

            mask = (t_full >= cursor) & (t_full < end)
            t_local = t_full[mask] - cursor
            T_eff = max(end - cursor, dt)

            # Random scaling per peak (segment)
            local_peak_scaling = float(self.rng.uniform(0.8, 1.2)) # if self.pattern_type != "yoyo" else 0.8,1.2
            g[mask] = pattern_fn(t_local, T_eff) * local_peak_scaling
            cursor = end

        noise = self.rng.normal(loc=1.0, scale=self.noise_std, size=len(g))
        noise = np.clip(noise, 0.05, None)
        
        # lambda_t is flows per second
        lambda_t = self.lambda_base * g * noise
        
        # Convert to flows per step for Poisson sampling
        self._flows = self.rng.poisson(lam=lambda_t * dt).astype(np.float32)
        
        # Re-compute EMA on the generated flows
        hl = float(50.0)
        alpha = 1.0 - math.exp(math.log(0.5) / hl) if hl > 0 else 1.0
        self._flows_ema = pd.Series(self._flows).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float32)

    def _init_start(self):
        self.active_len = self.t_max // 2

        self._generate_patterned_trace()
        max_start = self.t_max - self.active_len
        self.start = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        self.rep = 1
        self.scaling = float(self.rng.uniform(0.8, 2.0))
        self.tau = 0



    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_start()        


    def load_at(self, t: int):
        if not (self.start <= t < self.start + self.active_len):
            return None

        local_step = t - self.start
        
        if self.pattern_type == "yoyo":
            # Update tau
            if self.z_t == 1:
                self.tau = 0
                intensity = float(self.lambda_base)
            elif self.z_t == -1:
                self.tau = 0
                intensity = float(self.lambda_base)
            else:
                self.tau += 1
                if self.tau > self.tau_threshold:
                    intensity = float(self.lambda_base)
                else:
                    intensity = 0
                # intensity = 200
            intensity *= self.scaling
            
            dt = self.slot_ms / 1000.0
            flows = self.rng.poisson(lam=intensity * dt)
            
            return {
                "attacker_id": self.attacker_id,
                "attack_type": self.attack_type,
                "flows_per_sec": float(flows) * self.scaling,
            }

        if local_step < len(self._flows):
            return {
                "attacker_id": self.attacker_id,
                "attack_type": self.attack_type,
                "flows_per_sec": float(self._flows[local_step]) * self.scaling,
            }
        return None

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
        arrival_col: str = "num_objects",   # rename later if you want
        synth_cfg=None,
    ):
        self.user_id = str(user_id)
        self.slot_ms = float(slot_ms)
        self.t_max = int(t_max)
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

        per_step = np.maximum(df["num_requests_per_step"].to_numpy(dtype=int), 0.0)

        self.df = pd.DataFrame(
            {
                "t": np.arange(self.t_max, dtype=int),
                "user_id": self.user_id,
                "num_requests_per_step": per_step,
                "mu0": df["mu0"].to_numpy(dtype=float),
                "mu_t": df["mu_t"].to_numpy(dtype=float),
                "req_per_sec": df["req_per_sec"].to_numpy(dtype=int),
                "req_per_step_expected": df["req_per_step_expected"].to_numpy(dtype=float),
            }
        )
        self._req = self.df["num_requests_per_step"].to_numpy(dtype=np.int32)
        

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
        req_per_step_expected = req_per_sec
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

    def load_at(self, t: int):
        if t < 0 or t >= self._req.shape[0]:
            return None
        return {
            "user_id": self.user_id,
            "num_requests_per_step": int(self._req[t]),
        }

    def num_requests_at(self, t: int) -> int:
        if t < 0 or t >= self._req.shape[0]:
            return 0
        return int(self._req[t])

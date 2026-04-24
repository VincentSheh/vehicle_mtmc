import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
from dataclasses import dataclass
import math

def sinus_local(t_local: np.ndarray, T: float) -> np.ndarray:
    return 0.5 * (1.0 + np.sin(2.0 * np.pi * t_local / T))

def expo_local(t_local: np.ndarray, T: float) -> np.ndarray:
    x = np.exp(-3.0 * t_local / T)
    x = (x - np.exp(-3.0)) / (1.0 - np.exp(-3.0))
    return np.clip(x, 0.0, 1.0)

def pulse_wave_local(t_local: np.ndarray, T: float) -> np.ndarray:
    return (t_local < T / 2).astype(float)

def static_local(t_local: np.ndarray, T: float) -> np.ndarray:
    return np.ones_like(t_local, dtype=float)

@dataclass(frozen=True)
class AttackTypeSpec:
    """Immutable characteristics for one attack type, sampled once per run."""
    type_id: int
    lambda_base: float           # peak flows/sec
    noise_std: float
    pattern_type: str            # "sinus" | "pw" | "expo" | "static" | "yoyo"
    t_min_pattern: float         # seconds
    t_max_pattern: float         # seconds
    latency_per_flow: float      # ms per flow
    bw_per_flow: float           # Mbps per flow
    non_defendable_bw_const: float
    active_len_factor: float = 0.5


class AttackTypeLibrary:
    """
    Holds N attack type specifications sampled once per run.

    Call _sample_all() (done in __init__) to populate.
    Each EdgeArea holds a reference and draws from P(a|e) at episode start.
    """

    def __init__(self, n_types: int, sampler_cfg: dict, rng: np.random.Generator):
        self.n_types = n_types
        self.sampler_cfg = sampler_cfg
        self._specs: List[AttackTypeSpec] = []
        self._sample_all(rng)

    def _sample_all(self, rng: np.random.Generator):
        cfg = self.sampler_cfg
        pattern_types = cfg.get("pattern_types", ["sinus", "pw", "expo", "static"])
        specs = []
        for i in range(self.n_types):
            lb_range = cfg.get("lambda_base")
            if "lambda_level" in cfg:
                lvl = cfg.get("level", "default")
                lb_range = cfg["lambda_level"].get(lvl, cfg["lambda_level"].get("default"))

            ns_range   = cfg["noise_std"]
            tcm_range  = cfg["t_cycle_min"]
            tcd_range  = cfg["t_cycle_delta"]
            lat_range  = cfg["latency_per_flow"]
            bw_range   = cfg["bw_per_flow"]
            t_min = float(rng.uniform(*tcm_range))
            t_max = t_min + float(rng.uniform(*tcd_range))
            specs.append(AttackTypeSpec(
                type_id=i,
                lambda_base=float(rng.uniform(*lb_range)),
                noise_std=float(np.clip(rng.uniform(*ns_range), 0.0, 1.0)),
                pattern_type=str(rng.choice(pattern_types)),
                t_min_pattern=t_min,
                t_max_pattern=t_max,
                latency_per_flow=float(rng.uniform(*lat_range)),
                bw_per_flow=float(rng.uniform(*bw_range)),
                non_defendable_bw_const=float(cfg.get("non_defendable_bw_const", 0.0)),
                active_len_factor=float(cfg.get("active_len_factor", 0.5)),
            ))
        self._specs = specs

    def get(self, type_id: int) -> AttackTypeSpec:
        return self._specs[type_id]


class Attacker:
    """
    Generates malicious request arrivals for one episode.

    Characteristics come from an AttackTypeSpec sampled at run start.
    Pattern generation (sinus/pw/expo/static/yoyo) and Poisson arrival
    sampling are identical to the previous implementation.
    """

    def __init__(
        self,
        attacker_id: str,
        spec: AttackTypeSpec,
        slot_ms: float,
        t_max: int,
        seed: int,
        cpu_cycle_per_ms: float,
        cpu_cores: int,
    ):
        self.attacker_id = attacker_id
        self.attack_type = f"type_{spec.type_id}"
        self.episode_active = True

        self.latency_per_flow = spec.latency_per_flow
        self.cycle_per_flow = spec.latency_per_flow * float(cpu_cycle_per_ms) * int(cpu_cores)
        self.bw_per_flow = spec.bw_per_flow
        self.non_defendable_bw_const = spec.non_defendable_bw_const

        self.slot_ms = slot_ms
        self.t_max = t_max
        self.active_len_factor = spec.active_len_factor
        self.rep = 1
        self.scaling = 1
        self.z_t = 0
        self.tau = 0
        self.tau_threshold = 4000

        self.pattern_type = spec.pattern_type
        self.t_min_pattern = spec.t_min_pattern
        self.t_max_pattern = spec.t_max_pattern
        self.lambda_base = spec.lambda_base
        self.noise_std = spec.noise_std

        self.steps_per_sec = int(1000 // slot_ms)

        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self._init_start()

    def _generate_patterned_trace(self):
        dt = self.slot_ms / 1000.0
        t_full = np.arange(0, self.active_len * dt, dt)
        g = np.zeros_like(t_full, dtype=float)
        
        pattern_fn = {
            "sinus": sinus_local,
            "expo": expo_local,
            "pw": pulse_wave_local,
            "static": static_local,
        }.get(self.pattern_type, sinus_local)

        cursor = 0.0
        total_time = self.active_len * dt
        while cursor < total_time:
            T_k = float(self.rng.uniform(self.t_min_pattern, self.t_max_pattern))
            end = min(cursor + T_k, total_time)

            mask = (t_full >= cursor) & (t_full < end)
            t_local = t_full[mask] - cursor
            T_eff = max(end - cursor, dt)

            # Random scaling per peak (segment) — always draw to keep RNG state consistent,
            # but don't apply for "static" (only per-step noise should contribute there).
            local_peak_scaling = float(self.rng.uniform(0.8, 1.2))
            if self.pattern_type != "static":
                g[mask] = pattern_fn(t_local, T_eff) * local_peak_scaling
            else:
                g[mask] = pattern_fn(t_local, T_eff)
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
        self.active_len = int(self.t_max * self.active_len_factor)
        max_start = self.t_max - self.active_len
        # Sample start and scaling BEFORE trace generation so these are
        # identical across pattern types for the same seed.
        self.start = int(self.rng.uniform(500, max(501, max_start)))
        self.scaling = float(self.rng.uniform(0.8, 1.4))
        self.rep = 1
        self.tau = 0
        self._generate_patterned_trace()



    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_start()        

    def get_state(self) -> dict:
        return {
            "z_t": self.z_t,
        }

    def set_state(self, state: dict):
        self.z_t = state["z_t"]

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
            # intensity *= self.scaling
            
            dt = self.slot_ms / 1000.0
            noise = float(np.clip(self.rng.normal(loc=1.0, scale=self.noise_std), 0.05, None))
            flows = self.rng.poisson(lam=intensity * noise * dt)
            
            return {
                "attacker_id": self.attacker_id,
                "attack_type": self.attack_type,
                "flows_per_step": float(flows) * self.scaling,
            }

        if local_step < len(self._flows):
            return {
                "attacker_id": self.attacker_id,
                "attack_type": self.attack_type,
                "flows_per_step": float(self._flows[local_step]) * self.scaling,
            }
        return None

import numpy as np
import pandas as pd
from typing import Union, Optional


class User:
    """
    User request arrival time series.

    Modes:
    - synthetic: generate arrivals from a synthetic stochastic process
    - trace: load arrivals from a reconstructed CSV and slice a chunk of length t_max
    """

    def __init__(
        self,
        user_id: Union[str, int],
        slot_ms: float,
        t_max: int,
        seed: int = 0,
        synth_cfg=None,
        source_mode: str = "synthetic",   # "synthetic" or "trace"
        csv_path: Optional[str] = "output/job_count_reconstructed.csv",
        arrival_col: str = "recon_value",
        random_slice: bool = True,
    ):
        self.user_id = str(user_id)
        self.slot_ms = float(slot_ms)
        self.t_max = int(t_max)
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)

        self.synth_cfg = synth_cfg
        self.source_mode = str(source_mode)
        self.csv_path = csv_path
        self.arrival_col = str(arrival_col)
        self.random_slice = bool(random_slice)

        self.steps_per_sec = int(round(1000.0 / self.slot_ms))
        if self.steps_per_sec <= 0:
            raise ValueError(f"Invalid slot_ms={self.slot_ms}")

        self.full_trace = None
        if self.source_mode == "trace":
            if not self.csv_path:
                raise ValueError("csv_path must be provided when source_mode='trace'")
            self.full_trace = self._load_full_trace()

        self._init_source()

    def _init_source(self):
        if self.source_mode == "synthetic":
            self._init_from_synthetic()
        elif self.source_mode == "trace":
            self._init_from_trace()
        else:
            raise ValueError(
                f"Unsupported source_mode={self.source_mode}. "
                f"Use 'synthetic' or 'trace'."
            )

    # =========================================================
    # Synthetic mode
    # =========================================================
    def _init_from_synthetic(self):
        if self.synth_cfg is None:
            raise ValueError("synth_cfg must be provided when source_mode='synthetic'")

        cfg = self.synth_cfg

        mu_min = cfg.get("mu_min", 5.0)
        mu_max = cfg.get("mu_max", 40.0)
        if "mu_level" in cfg:
            lvl = cfg.get("level", "default")
            mu_range = cfg["mu_level"].get(lvl, cfg["mu_level"].get("default"))
            if mu_range:
                mu_min, mu_max = mu_range

        df = self.generate_req_trace(
            t_steps=self.t_max,
            slot_ms=self.slot_ms,
            rng=self.rng,
            rw_sigma_per_sqrt_sec=cfg["rw_sigma_per_sqrt_sec"],
            mu_min=mu_min,
            mu_max=mu_max,
            kappa=cfg["kappa"],
            sigma=cfg["sigma"],
        )

        per_step = np.maximum(
            np.rint(df["num_requests_per_step"].to_numpy(dtype=float)),
            0
        ).astype(np.int32)

        self.df = pd.DataFrame(
            {
                "t": np.arange(self.t_max, dtype=int),
                "user_id": self.user_id,
                "num_requests_per_step": per_step,
                "mu0": df["mu0"].to_numpy(dtype=float),
                "mu_t": df["mu_t"].to_numpy(dtype=float),
                "req_per_sec": df["req_per_sec"].to_numpy(dtype=float),
                "req_per_step_expected": df["req_per_step_expected"].to_numpy(dtype=float),
            }
        )

        self.slice_start = None
        self.slice_end = None
        self._req = per_step

    def generate_req_trace(
        self,
        t_steps: int,
        slot_ms: float,
        rng: np.random.Generator,
        rw_sigma_per_sqrt_sec: float = 0.8,
        mu_min: float = 5.0,
        mu_max: float = 40.0,
        kappa: float = 0.02,
        sigma: float = 0.9,
    ):
        dt = slot_ms / 1000.0

        mu0 = float(rng.uniform(mu_min, mu_max))

        mu_series = np.empty(t_steps, dtype=float)
        mu_series[0] = mu0
        for t in range(1, t_steps):
            step = rw_sigma_per_sqrt_sec * np.sqrt(dt) * rng.standard_normal()
            mu_series[t] = np.clip(mu_series[t - 1] + step, mu_min, mu_max)

        x = float(mu_series[0])
        req_per_sec = np.zeros(t_steps, dtype=float)

        for t in range(t_steps):
            mu_t = float(mu_series[t])
            x = x + kappa * (mu_t - x) + sigma * rng.standard_normal()
            x = max(0.0, x)
            req_per_sec[t] = x

        req_per_step_expected = req_per_sec * dt
        req_per_step = np.maximum(req_per_step_expected, 0.0)

        return pd.DataFrame(
            {
                "t": np.arange(t_steps, dtype=int),
                "mu0": mu0,
                "mu_t": mu_series,
                "req_per_sec": req_per_sec,
                "req_per_step_expected": req_per_step_expected,
                "num_requests_per_step": req_per_step,
            }
        )

    # =========================================================
    # Trace mode
    # =========================================================
    def _load_full_trace(self) -> np.ndarray:
        values = pd.read_csv(self.csv_path, usecols=[self.arrival_col])[self.arrival_col]
        values = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        return np.maximum(values, 0.0)

    def _sample_start_index(self) -> int:
        n = self.full_trace.shape[0]
        if n < self.t_max:
            raise ValueError(
                f"Trace length ({n}) is smaller than requested t_max ({self.t_max})."
            )

        max_start = n - self.t_max
        if not self.random_slice or max_start == 0:
            return 0
        return int(self.rng.integers(0, max_start + 1))

    def _init_from_trace(self):
        start_idx = self._sample_start_index()
        end_idx = start_idx + self.t_max

        chunk = self.full_trace[start_idx:end_idx]
        self._req = np.maximum(np.rint(chunk), 0).astype(np.int32)

        self.slice_start = start_idx
        self.slice_end = end_idx
        self.df = None

    # =========================================================
    # Public methods
    # =========================================================
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.base_seed = int(seed)
            self.rng = np.random.default_rng(self.base_seed)

        self._init_source()

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
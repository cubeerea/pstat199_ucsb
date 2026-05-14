"""
PID controller implementations.

PerLayerPIDController: replicates the paper's per-layer PID (computes steering dirs
  via PID recurrence over DIM directions; direction is static during generation).

GlobalPIDController: Hank's contribution — single PID using static global vector r_bar.
  Error signal is constant e(k) = r_bar (default, per Q1 in notes/decisions.md).
"""
from __future__ import annotations

import torch
from torch import Tensor


class PerLayerPIDController:
    """
    Per-layer PID: pre-computes one steering direction per layer via PID
    recurrence over DIM directions. Directions are applied as static ActAdd
    during generation (no online update).
    """

    def __init__(
        self,
        ref_dirs: dict[int, Tensor],
        kp: float = 0.9,
        ki: float = 0.01,
        kd: float = 0.01,
    ):
        from src.dim import apply_pid_to_dirs
        self.steering_dirs = apply_pid_to_dirs(ref_dirs, kp=kp, ki=ki, kd=kd)

    def get_steering_dir(self, layer_idx: int) -> Tensor:
        return self.steering_dirs[layer_idx]


class GlobalPIDController:
    """
    Single PID using a static global refusal vector r_bar.
    Applied at every layer in the persistence window W.

    Error signal: constant e(k) = r_bar (Q1 default).
    The I-term accumulates linearly with layer index → integral windup test.
    """

    def __init__(
        self,
        r_bar: Tensor,
        kp: float = 0.9,
        ki: float = 0.01,
        kd: float = 0.01,
        window: list[int] | None = None,
    ):
        self.r_bar = r_bar
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.window = window or []
        self.reset()

    def reset(self):
        self.integral = torch.zeros_like(self.r_bar)
        self.prev_error: Tensor | None = None
        self._layer_order: list[int] = []
        self._steering_dirs: dict[int, Tensor] = {}
        # Diagnostic norms (set by precompute_steering_dirs; no forward pass needed)
        self.p_norms: dict[int, float] = {}
        self.i_norms: dict[int, float] = {}
        self.d_norms: dict[int, float] = {}
        self.integral_norms: dict[int, float] = {}  # pre-scale ||integral||

    def precompute_steering_dirs(self) -> dict[int, Tensor]:
        """
        Pre-compute one steering direction per layer in window using the
        constant-error PID recurrence. Returns dict layer_idx -> steering_dir.
        Stored in self._steering_dirs.
        """
        self.reset()
        for k in sorted(self.window):
            e = self.r_bar

            p_term = self.kp * e
            self.integral = self.integral + e
            self.integral_norms[k] = float(self.integral.norm())
            i_term = self.ki * self.integral

            if self.prev_error is None:
                d_term = torch.zeros_like(e)
            else:
                d_term = self.kd * (e - self.prev_error)
            self.prev_error = e.clone()

            self.p_norms[k] = float(p_term.norm())
            self.i_norms[k] = float(i_term.norm())
            self.d_norms[k] = float(d_term.norm())

            u = p_term + i_term + d_term
            self._steering_dirs[k] = u.clone()

        return self._steering_dirs

    def get_steering_dir(self, layer_idx: int) -> Tensor:
        if layer_idx not in self._steering_dirs:
            raise KeyError(f"Layer {layer_idx} not in window or precompute_steering_dirs() not called")
        return self._steering_dirs[layer_idx]


class GlobalPIDControllerAntiWindup(GlobalPIDController):
    """
    Global PID with I-term clamped to 2 * ||r_bar||.
    Ablation condition per CLAUDE.md §4.4.
    """

    def precompute_steering_dirs(self) -> dict[int, Tensor]:
        clamp_limit = 2.0 * self.r_bar.norm().item()
        self.reset()
        for k in sorted(self.window):
            e = self.r_bar
            p_term = self.kp * e
            self.integral = self.integral + e
            self.integral_norms[k] = float(self.integral.norm())  # pre-clamp
            i_clamped = torch.clamp(self.integral, min=-clamp_limit, max=clamp_limit)
            i_term = self.ki * i_clamped

            if self.prev_error is None:
                d_term = torch.zeros_like(e)
            else:
                d_term = self.kd * (e - self.prev_error)
            self.prev_error = e.clone()

            self.p_norms[k] = float(p_term.norm())
            self.i_norms[k] = float(i_term.norm())
            self.d_norms[k] = float(d_term.norm())

            u = p_term + i_term + d_term
            self._steering_dirs[k] = u.clone()

        return self._steering_dirs

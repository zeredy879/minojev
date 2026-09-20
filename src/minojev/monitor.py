"""Training observability and memory safety.

Every training phase reports a memory snapshot, a hard budget can abort a run
before the operating system is at risk, and progress is written to
``status.json`` (live, overwritten) plus ``metrics.jsonl`` (append-only) so a
run can be watched from another terminal:

    watch -n 2 cat runs/qwen/status.json
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


_last_cpu: dict = {"wall": None, "cpu": None}


def _process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def pressure_snapshot() -> dict:
    """CPU and system memory pressure, computed without external libraries.

    ``process_cpu_percent`` is the share of total machine CPU capacity used by
    this process since the previous call; ``load1_per_core`` uses the OS load
    average normalized by core count.
    """
    now = time.perf_counter()
    cpu_seconds = _process_cpu_seconds()
    cores = os.cpu_count() or 1
    cpu_percent = None
    cpu_cores = None
    if _last_cpu["wall"] is not None:
        wall = max(now - _last_cpu["wall"], 1e-9)
        delta = max(cpu_seconds - _last_cpu["cpu"], 0.0)
        cpu_percent = round(100 * delta / (wall * cores), 1)
        cpu_cores = round(delta / wall, 2)
    _last_cpu["wall"] = now
    _last_cpu["cpu"] = cpu_seconds
    load1 = load_per_core = None
    try:
        load1 = round(os.getloadavg()[0], 2)
        load_per_core = round(load1 / cores, 2)
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        pass
    system_memory_percent = None
    system_memory_gb = None
    try:  # optional dependency, best signal when present
        import psutil

        memory = psutil.virtual_memory()
        system_memory_percent = round(memory.percent, 1)
        system_memory_gb = round((memory.total - memory.available) / 1e9, 2)
    except ImportError:
        try:
            total = os.sysconf("SC_PHYS_PAGES")
            available = os.sysconf("SC_AVPHYS_PAGES")
            if total and available:
                system_memory_percent = round(100 * (1 - available / total), 1)
        except (ValueError, OSError, AttributeError):  # pragma: no cover
            pass
    return {
        "cpu_cores": cores,
        "process_cpu_percent": cpu_percent,
        "process_cpu_cores": cpu_cores,
        "load1": load1,
        "load1_per_core": load_per_core,
        "system_memory_percent": system_memory_percent,
        "system_memory_gb": system_memory_gb,
    }


def memory_snapshot() -> dict:
    """Current memory use: MPS driver/allocated and process peak RSS (GB)."""
    snapshot = {
        "cpu_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 3),
        "mps_allocated_gb": None,
        "mps_driver_gb": None,
    }
    if torch.backends.mps.is_available():
        try:
            snapshot["mps_allocated_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 3)
            snapshot["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 3)
        except RuntimeError:  # pragma: no cover - driver not initialized yet
            pass
    return snapshot


def release_memory() -> None:
    """Drop cached allocations so the next phase starts from a clean slate."""
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except RuntimeError:  # pragma: no cover
            pass


@dataclass
class MemoryBudget:
    max_gb: float | None = None

    def used_gb(self, snapshot: dict) -> float:
        values = [snapshot.get("cpu_peak_rss_gb") or 0.0, snapshot.get("mps_driver_gb") or 0.0]
        return max(values)

    def check(self, context: str = "") -> dict:
        snapshot = memory_snapshot()
        if self.max_gb is not None and self.used_gb(snapshot) > self.max_gb:
            raise MemoryError(
                f"Memory budget exceeded at {context}: "
                f"used {self.used_gb(snapshot):.2f} GB > limit {self.max_gb:.2f} GB "
                f"(mps_allocated={snapshot['mps_allocated_gb']}, mps_driver={snapshot['mps_driver_gb']}, "
                f"cpu_peak_rss={snapshot['cpu_peak_rss_gb']}). "
                "Lower --cache-batch-requests / --batch-requests or raise --max-memory-gb."
            )
        return snapshot


class TrainingMonitor:
    def __init__(
        self,
        output_dir: str | Path,
        total_steps: int,
        log_every: int = 10,
        max_memory_gb: float | None = None,
        label: str = "train",
        stream=None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total_steps = max(int(total_steps), 1)
        self.log_every = max(int(log_every), 1)
        self.label = label
        self.stream = stream or sys.stderr
        self.budget = MemoryBudget(max_gb=max_memory_gb)
        self.status_path = self.output_dir / "status.json"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.started = time.perf_counter()
        self.last_tokens = 0
        self.last_elapsed = 0.0

    def log(
        self,
        step: int,
        phase: str,
        loss: float | None = None,
        lr: float | None = None,
        grad_norm: float | None = None,
        tokens: int = 0,
        extra: dict | None = None,
        force: bool = False,
    ) -> dict | None:
        if not force and step % self.log_every:
            return None
        snapshot = self.budget.check(f"{self.label} step {step}")
        elapsed = time.perf_counter() - self.started
        eta = elapsed / max(step, 1) * max(self.total_steps - step, 0)
        window_tokens = max(tokens - self.last_tokens, 0)
        window_seconds = max(elapsed - self.last_elapsed, 1e-9)
        entry = {
            "label": self.label,
            "step": step,
            "total_steps": self.total_steps,
            "phase": phase,
            "loss": round(loss, 6) if loss is not None else None,
            "lr": lr,
            "grad_norm": round(grad_norm, 4) if grad_norm is not None else None,
            "tokens_per_second": round(window_tokens / window_seconds, 1) if window_tokens else None,
            "elapsed_seconds": round(elapsed, 2),
            "eta_seconds": round(eta, 1),
            "memory": snapshot,
            "pressure": pressure_snapshot(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra:
            entry.update(extra)
        self.last_tokens = tokens
        self.last_elapsed = elapsed
        self.status_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")
        with self.metrics_path.open("a") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        memory = snapshot.get("mps_driver_gb") or snapshot.get("cpu_peak_rss_gb")
        loss_text = f"loss={entry['loss']}" if entry["loss"] is not None else "loss=?"
        print(
            f"[{self.label}] step {step}/{self.total_steps} {phase} {loss_text} "
            f"mem={memory:.2f}GB eta={eta / 60:.1f}min",
            file=self.stream,
            flush=True,
        )
        return entry

    def note(self, message: str) -> None:
        entry = {
            "label": self.label,
            "note": message,
            "memory": memory_snapshot(),
            "pressure": pressure_snapshot(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.status_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")
        with self.metrics_path.open("a") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[{self.label}] {message}", file=self.stream, flush=True)

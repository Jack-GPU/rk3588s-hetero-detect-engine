#!/usr/bin/env python3
"""板上长测：边跑多路推理边采样 CPU / NPU / 温度（功耗表需外接时手填）。

默认驱动已有的四路队列 bench；也可只采样空闲基线。

用法：
  python3 board_soak_power.py --duration 120 --cmd './cpp/mpp_multi_queue_bench ...'
  python3 board_soak_power.py --duration 60 --idle
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None


def sample_once() -> dict:
    row: dict = {"t": time.time()}
    # CPU：/proc/stat 差分在外层算；这里记瞬时 loadavg
    try:
        row["loadavg_1"] = float(os.getloadavg()[0])
    except OSError:
        row["loadavg_1"] = None

    npu = read_text(Path("/sys/kernel/debug/rknpu/load"))
    row["rknpu_load"] = npu

    temps = {}
    thermal = Path("/sys/class/thermal")
    if thermal.is_dir():
        for z in sorted(thermal.glob("thermal_zone*")):
            t = read_text(z / "temp")
            typ = read_text(z / "type") or z.name
            if t and t.isdigit():
                temps[typ] = int(t) / 1000.0
    row["temp_c"] = temps

    # 常见供电监控节点（有则记，无则 null）
    power_uw = None
    for p in Path("/sys/class/hwmon").glob("hwmon*/power*_input"):
        v = read_text(p)
        if v and v.isdigit():
            power_uw = int(v)
            break
    row["power_uw"] = power_uw
    return row


def cpu_times() -> tuple[int, int] | None:
    try:
        line = Path("/proc/stat").read_text().splitlines()[0]
        parts = [int(x) for x in line.split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        total = sum(parts)
        return idle, total
    except (OSError, ValueError, IndexError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=int, default=120, help="采样秒数")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", default="bench/logs/soak_power.json")
    ap.add_argument("--idle", action="store_true", help="只采样，不跑 workload")
    ap.add_argument(
        "--cmd",
        default="./cpp/mpp_multi_queue_bench yolov8n_int8.rknn bus_1080p30.h264 90 auto 4 30",
        help="workload 命令（可循环）",
    )
    ap.add_argument("--loop-workload", action="store_true", default=True)
    args = ap.parse_args()

    proc = None
    if not args.idle:
        proc = subprocess.Popen(
            args.cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

    samples = []
    t0 = time.time()
    prev = cpu_times()
    try:
        while time.time() - t0 < args.duration:
            if proc is not None and proc.poll() is not None and args.loop_workload:
                proc = subprocess.Popen(
                    args.cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                )
            row = sample_once()
            cur = cpu_times()
            if prev and cur:
                di, dt = cur[0] - prev[0], cur[1] - prev[1]
                row["cpu_util_pct"] = (
                    None if dt <= 0 else round(100.0 * (1.0 - di / dt), 2)
                )
            else:
                row["cpu_util_pct"] = None
            prev = cur
            samples.append(row)
            time.sleep(args.interval)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

    cpus = [s["cpu_util_pct"] for s in samples if s.get("cpu_util_pct") is not None]
    temps_all = []
    for s in samples:
        for v in (s.get("temp_c") or {}).values():
            temps_all.append(v)
    powers = [s["power_uw"] for s in samples if s.get("power_uw") is not None]

    summary = {
        "duration_s": args.duration,
        "interval_s": args.interval,
        "idle": args.idle,
        "cmd": None if args.idle else args.cmd,
        "n_samples": len(samples),
        "cpu_util_pct": {
            "mean": round(sum(cpus) / len(cpus), 2) if cpus else None,
            "max": max(cpus) if cpus else None,
        },
        "temp_c": {
            "max": max(temps_all) if temps_all else None,
            "mean": round(sum(temps_all) / len(temps_all), 2) if temps_all else None,
        },
        "power_w": {
            "mean": round(sum(powers) / len(powers) / 1e6, 3) if powers else None,
            "note": "无 hwmon power*_input 时为 null，需 USB 功率计手测",
        },
        "samples": samples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "samples"}, indent=2, ensure_ascii=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time
from typing import Any, Callable

import requests


ToolFunction = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _run(command: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _gib(value: int | float) -> float:
    return round(float(value) / (1024.0 ** 3), 2)


def _number(value: str) -> int | float | str:
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return round(number, 2)


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return values


def _system_status(_: dict[str, Any]) -> dict[str, Any]:
    mem = _read_meminfo()
    total = mem.get("MemTotal", 0)
    available = mem.get("MemAvailable", 0)
    used = max(0, total - available)

    disk = shutil.disk_usage("/")
    try:
        uptime_seconds = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        uptime_seconds = 0.0

    try:
        load = [round(value, 2) for value in os.getloadavg()]
    except (OSError, AttributeError):
        load = []

    return {
        "hostname": socket.gethostname(),
        "local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "load_average_1_5_15": load,
        "uptime_hours": round(uptime_seconds / 3600.0, 2),
        "memory": {
            "used_gib": _gib(used),
            "available_gib": _gib(available),
            "total_gib": _gib(total),
        },
        "root_disk": {
            "used_gib": _gib(disk.used),
            "free_gib": _gib(disk.free),
            "total_gib": _gib(disk.total),
            "percent_used": round((disk.used / disk.total) * 100.0, 1)
            if disk.total
            else 0.0,
        },
    }


def _gpu_status(_: dict[str, Any]) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,pstate",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = _run(command)
    except FileNotFoundError:
        return {
            "available": False,
            "error": "nvidia-smi is not installed or not on PATH",
        }
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "nvidia-smi timed out"}

    if proc.returncode != 0:
        return {
            "available": False,
            "error": (proc.stderr or proc.stdout or "nvidia-smi failed").strip()[:500],
        }

    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 6:
            continue

        name, util, used, total, temp, pstate = fields[:6]
        try:
            used_mib = float(used)
            total_mib = float(total)
        except ValueError:
            used_mib = 0.0
            total_mib = 0.0

        gpus.append(
            {
                "name": name,
                "utilization_percent": _number(util),
                "memory_used_mib": _number(used),
                "memory_total_mib": _number(total),
                "memory_free_mib": round(max(0.0, total_mib - used_mib), 1),
                "temperature_c": _number(temp),
                "performance_state": pstate,
            }
        )

    return {"available": bool(gpus), "gpus": gpus}


def _ollama_status_factory(base_url: str) -> ToolFunction:
    base = base_url.rstrip("/")

    def ollama_status(_: dict[str, Any]) -> dict[str, Any]:
        version = None
        version_error = None

        try:
            response = requests.get(f"{base}/api/version", timeout=5)
            response.raise_for_status()
            version = (response.json() or {}).get("version")
        except Exception as exc:
            version_error = f"{type(exc).__name__}: {exc}"

        try:
            response = requests.get(f"{base}/api/ps", timeout=5)
            response.raise_for_status()
            raw_models = (response.json() or {}).get("models") or []
        except Exception as exc:
            return {
                "available": False,
                "version": version,
                "version_error": version_error,
                "error": f"{type(exc).__name__}: {exc}",
            }

        models: list[dict[str, Any]] = []
        for model in raw_models[:20]:
            details = model.get("details") or {}
            models.append(
                {
                    "name": model.get("name") or model.get("model"),
                    "size_bytes": model.get("size"),
                    "size_vram_bytes": model.get("size_vram"),
                    "expires_at": model.get("expires_at"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization_level": details.get("quantization_level"),
                    "family": details.get("family"),
                }
            )

        result = {
            "available": True,
            "version": version,
            "loaded_model_count": len(raw_models),
            "loaded_models": models,
        }
        if version_error:
            result["version_error"] = version_error
        return result

    return ollama_status


def _docker_status(args: dict[str, Any]) -> dict[str, Any]:
    include_stopped = bool(args.get("include_stopped", False))
    command = ["docker", "ps"]
    if include_stopped:
        command.append("-a")
    command.extend(["--format", "{{json .}}"])

    try:
        proc = _run(command)
    except FileNotFoundError:
        return {"available": False, "error": "docker is not installed or not on PATH"}
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "docker ps timed out"}

    if proc.returncode != 0:
        return {
            "available": False,
            "error": (proc.stderr or proc.stdout or "docker ps failed").strip()[:800],
        }

    containers: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        containers.append(
            {
                "name": item.get("Names"),
                "image": item.get("Image"),
                "state": item.get("State"),
                "status": item.get("Status"),
                "ports": item.get("Ports"),
            }
        )

    return {
        "available": True,
        "include_stopped": include_stopped,
        "container_count": len(containers),
        "containers": containers[:50],
    }


class ToolRegistry:
    def __init__(self, base_url: str):
        empty_object = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

        self._tools: dict[str, ToolDefinition] = {
            "system_status": ToolDefinition(
                name="system_status",
                description=(
                    "Read the current local computer status, including hostname, "
                    "local time, uptime, CPU load, RAM usage, and root disk usage."
                ),
                parameters=empty_object,
                function=_system_status,
            ),
            "gpu_status": ToolDefinition(
                name="gpu_status",
                description=(
                    "Read current NVIDIA GPU state using nvidia-smi, including "
                    "utilization, VRAM used/free/total, temperature, and p-state."
                ),
                parameters=empty_object,
                function=_gpu_status,
            ),
            "ollama_status": ToolDefinition(
                name="ollama_status",
                description=(
                    "Read the local Ollama version and models currently loaded "
                    "in memory. Use this for live Ollama or model-memory questions."
                ),
                parameters=empty_object,
                function=_ollama_status_factory(base_url),
            ),
            "docker_status": ToolDefinition(
                name="docker_status",
                description=(
                    "Read Docker container status on the local computer. "
                    "This tool cannot start, stop, or modify containers."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "include_stopped": {
                            "type": "boolean",
                            "description": (
                                "If true, include stopped containers. "
                                "If false, show only running containers."
                            ),
                        }
                    },
                    "additionalProperties": False,
                },
                function=_docker_status,
            ),
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def schemas(self, enabled_names: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            self._tools[name].schema()
            for name in enabled_names
            if name in self._tools
        ]

    def execute(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool: {name}"}

        try:
            result = tool.function(arguments or {})
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        return {"ok": True, "tool": name, "result": result}

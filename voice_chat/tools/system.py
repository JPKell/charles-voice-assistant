from __future__ import annotations

import configparser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


MAX_OUTPUT_CHARS = 20_000
SERVICE_RE = re.compile(r"^[A-Za-z0-9@_.:-]{1,128}$")
SINCE_RE = re.compile(r"^[A-Za-z0-9_ :+.,/-]{1,64}$")


def _clip(value: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _run(command: list[str], *, timeout: float = 10.0) -> dict[str, Any]:
    executable = command[0]
    if shutil.which(executable) is None:
        return {"available": False, "error": f"{executable} is not installed or not on PATH"}
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "error": f"{executable} timed out"}

    output = _clip(proc.stdout or proc.stderr)
    return {
        "available": True,
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "output": output,
    }


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def search_files(args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        return {"available": False, "error": "query is required"}

    root = Path(str(args.get("path") or Path.home())).expanduser()
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        return {"available": False, "error": f"Cannot access search path: {exc}"}
    if not root.is_dir():
        return {"available": False, "error": "search path must be a directory"}

    mode = str(args.get("mode") or "name").lower()
    limit = _bounded_int(args.get("limit"), 50, 1, 200)
    if mode == "content":
        command = [
            "rg",
            "--files-with-matches",
            "--fixed-strings",
            "--hidden",
            "--max-filesize",
            "10M",
            "--",
            query,
            str(root),
        ]
    elif mode == "name":
        depth = _bounded_int(args.get("max_depth"), 6, 1, 20)
        command = [
            "find",
            str(root),
            "-maxdepth",
            str(depth),
            "-iname",
            f"*{query}*",
            "-print",
        ]
    else:
        return {"available": False, "error": "mode must be 'name' or 'content'"}

    result = _run(command, timeout=15.0)
    if not result.get("available"):
        return result
    matches = [line for line in str(result.get("output") or "").splitlines() if line]
    return {
        "available": True,
        "ok": result.get("ok", False),
        "path": str(root),
        "mode": mode,
        "match_count_returned": min(len(matches), limit),
        "truncated": len(matches) > limit,
        "matches": matches[:limit],
    }


def _desktop_entries() -> list[dict[str, str]]:
    directories = [
        Path.home() / ".local/share/applications",
        Path("/usr/local/share/applications"),
        Path("/usr/share/applications"),
    ]
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.desktop"):
            desktop_id = path.name.removesuffix(".desktop")
            if desktop_id in seen:
                continue
            parser = configparser.ConfigParser(interpolation=None, strict=False)
            try:
                parser.read(path, encoding="utf-8")
                section = parser["Desktop Entry"]
            except (OSError, KeyError, configparser.Error):
                continue
            if section.getboolean("Hidden", fallback=False) or section.getboolean(
                "NoDisplay", fallback=False
            ):
                continue
            name = section.get("Name", "").strip()
            if not name or section.get("Type", "Application") != "Application":
                continue
            seen.add(desktop_id)
            entries.append({"name": name, "desktop_id": desktop_id})
    return sorted(entries, key=lambda item: item["name"].casefold())


def list_applications(args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query") or "").strip().casefold()
    limit = _bounded_int(args.get("limit"), 50, 1, 200)
    entries = _desktop_entries()
    if query:
        entries = [
            item
            for item in entries
            if query in item["name"].casefold() or query in item["desktop_id"].casefold()
        ]
    return {
        "available": True,
        "application_count_returned": min(len(entries), limit),
        "truncated": len(entries) > limit,
        "applications": entries[:limit],
    }


def launch_application(args: dict[str, Any]) -> dict[str, Any]:
    requested = str(args.get("application") or "").strip()
    if not requested:
        return {"available": False, "error": "application is required"}
    folded = requested.casefold().removesuffix(".desktop")
    entries = _desktop_entries()
    exact = [
        item
        for item in entries
        if folded in {item["name"].casefold(), item["desktop_id"].casefold()}
    ]
    if not exact:
        suggestions = [
            item for item in entries if folded in item["name"].casefold()
        ][:10]
        return {
            "available": False,
            "error": "No exact installed application match",
            "suggestions": suggestions,
        }
    if len(exact) > 1:
        return {
            "available": False,
            "error": "Application name is ambiguous; use a desktop_id",
            "matches": exact,
        }
    launcher = shutil.which("gtk-launch")
    if launcher is None:
        return {"available": False, "error": "gtk-launch is not installed or not on PATH"}

    selected = exact[0]
    try:
        subprocess.Popen(
            [launcher, selected["desktop_id"]],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return {"available": False, "error": f"Could not launch application: {exc}"}
    return {"available": True, "launched": selected}


def list_devices(args: dict[str, Any]) -> dict[str, Any]:
    category = str(args.get("category") or "all").lower()
    commands = {
        "storage": [
            "lsblk", "--json", "--output",
            "NAME,PATH,TYPE,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINTS,MODEL,TRAN,RM,RO,HOTPLUG",
        ],
        "usb": ["lsusb"],
        "pci": ["lspci"],
        "audio_playback": ["aplay", "--list-devices"],
        "audio_capture": ["arecord", "--list-devices"],
        "network": ["ip", "--brief", "link"],
    }
    if category != "all" and category not in commands:
        return {"available": False, "error": f"Unknown device category: {category}"}

    selected = commands if category == "all" else {category: commands[category]}
    devices: dict[str, Any] = {}
    for name, command in selected.items():
        result = _run(command)
        if name == "storage" and result.get("ok"):
            try:
                result["devices"] = json.loads(result.pop("output")).get("blockdevices", [])
            except (json.JSONDecodeError, AttributeError):
                pass
        devices[name] = result
    return {"available": True, "categories": devices}


def storage_permissions(args: dict[str, Any]) -> dict[str, Any]:
    requested = str(args.get("path") or "/").strip()
    path = Path(requested).expanduser()
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        return {"available": False, "error": f"Cannot inspect path: {exc}"}

    mount = _run(
        [
            "findmnt", "--json", "--target", str(resolved), "--output",
            "TARGET,SOURCE,FSTYPE,OPTIONS,SIZE,USED,AVAIL,USE%",
        ]
    )
    mount_data: Any = mount
    if mount.get("ok"):
        try:
            filesystems = json.loads(mount.get("output") or "{}").get("filesystems", [])
            mount_data = filesystems[0] if filesystems else {}
        except json.JSONDecodeError:
            pass

    return {
        "available": True,
        "path": str(resolved),
        "owner_uid": stat.st_uid,
        "owner_gid": stat.st_gid,
        "mode": oct(stat.st_mode & 0o7777),
        "current_user_access": {
            "read": os.access(resolved, os.R_OK),
            "write": os.access(resolved, os.W_OK),
            "execute": os.access(resolved, os.X_OK),
        },
        "mount": mount_data,
    }


def process_monitor(args: dict[str, Any]) -> dict[str, Any]:
    sort = str(args.get("sort") or "cpu").lower()
    if sort not in {"cpu", "memory"}:
        return {"available": False, "error": "sort must be 'cpu' or 'memory'"}
    limit = _bounded_int(args.get("limit"), 20, 1, 100)
    sort_field = "-%cpu" if sort == "cpu" else "-%mem"
    result = _run(
        [
            # Do not expose full command lines: they sometimes contain tokens.
            "ps", "-eo", "pid,ppid,user,stat,%cpu,%mem,etimes,comm",
            f"--sort={sort_field}",
        ]
    )
    if not result.get("ok"):
        return result
    lines = str(result.get("output") or "").splitlines()
    return {
        "available": True,
        "sort": sort,
        "processes": "\n".join(lines[: limit + 1]),
    }


def _service_command(scope: str, command: list[str]) -> list[str]:
    return ["systemctl", "--user", *command] if scope == "user" else ["systemctl", *command]


def service_status(args: dict[str, Any]) -> dict[str, Any]:
    service = str(args.get("service") or "").strip()
    scope = str(args.get("scope") or "system").lower()
    if not SERVICE_RE.fullmatch(service):
        return {"available": False, "error": "Invalid or missing service name"}
    if scope not in {"system", "user"}:
        return {"available": False, "error": "scope must be 'system' or 'user'"}
    command = _service_command(
        scope,
        [
            "show", service, "--no-pager",
            "--property=Id,Description,LoadState,ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus,MemoryCurrent,CPUUsageNSec",
        ],
    )
    result = _run(command)
    result.update({"service": service, "scope": scope})
    return result


def service_logs(args: dict[str, Any]) -> dict[str, Any]:
    service = str(args.get("service") or "").strip()
    scope = str(args.get("scope") or "system").lower()
    since = str(args.get("since") or "today").strip()
    lines = _bounded_int(args.get("lines"), 100, 1, 500)
    if not SERVICE_RE.fullmatch(service):
        return {"available": False, "error": "Invalid or missing service name"}
    if scope not in {"system", "user"}:
        return {"available": False, "error": "scope must be 'system' or 'user'"}
    if not SINCE_RE.fullmatch(since):
        return {"available": False, "error": "Invalid since value"}
    command = ["journalctl"]
    if scope == "user":
        command.append("--user")
    command.extend(
        ["--unit", service, "--lines", str(lines), "--since", since, "--no-pager", "--output=short-iso"]
    )
    result = _run(command, timeout=15.0)
    result.update({"service": service, "scope": scope, "since": since, "lines": lines})
    return result


def system_command(args: dict[str, Any]) -> dict[str, Any]:
    action = str(args.get("action") or "").lower()
    commands = {
        "identity": ["id"],
        "kernel": ["uname", "-a"],
        "uptime": ["uptime"],
        "memory": ["free", "-h"],
        "filesystems": ["df", "-hT"],
        "network": ["ip", "--brief", "address"],
        "temperatures": ["sensors"],
    }
    command = commands.get(action)
    if command is None:
        return {"available": False, "error": f"Unknown system action: {action}"}
    result = _run(command)
    result["action"] = action
    return result

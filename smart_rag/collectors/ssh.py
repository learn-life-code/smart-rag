#!/usr/bin/env python3
"""SSH collector — index a LIVE machine (embedded module, server, VCU) read-only.

Connect over SSH, run a SAFE set of read-only discovery commands, and ingest the
output (configs, logs, process/service state, versions) so an agent can answer
"what's running / what version / what failed" without a human SSHing in and grepping.

SAFETY (non-negotiable): only READ-ONLY commands run. The default command set is a
conservative allow-list; you can extend it, but write/destructive commands are
refused. No sudo by default. This is for inspection, not control.

  from smart_rag.collectors.ssh import collect_ssh
  facts_text = collect_ssh("user@host", key="~/.ssh/id_rsa")   # → ingestible text
  # or via the index manager:
  #   mgr.build_ssh("vcu1", "root@10.0.0.5")

Requires `paramiko` (optional). Without it, raises a clear message.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

# Read-only discovery commands. Each yields a labeled block we ingest as a doc.
# Conservative + portable (Linux/QNX-ish). Extend via `extra_commands`, but every
# command is checked against _WRITE_GUARD first.
_DEFAULT_COMMANDS: Dict[str, str] = {
    "os_release":     "cat /etc/os-release 2>/dev/null || uname -a",
    "kernel":         "uname -a",
    "uptime":         "uptime",
    "hostname":       "hostname",
    "running_procs":  "ps -eo pid,comm,args 2>/dev/null | head -200",
    "services":       "systemctl list-units --type=service --state=running 2>/dev/null | head -100",
    "disk":           "df -h 2>/dev/null",
    "memory":         "free -h 2>/dev/null || cat /proc/meminfo 2>/dev/null | head -20",
    "network":        "ip addr 2>/dev/null || ifconfig 2>/dev/null",
    "listening_ports":"ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null",
    "installed_pkgs": "dpkg -l 2>/dev/null | head -200 || rpm -qa 2>/dev/null | head -200",
    "recent_errors":  "journalctl -p err -n 200 --no-pager 2>/dev/null || dmesg 2>/dev/null | tail -200",
    "mounts":         "mount 2>/dev/null | head -50",
}

# Destructive/write COMMANDS (word-bounded) → refused, even if a user extends the set.
_WRITE_GUARD = re.compile(
    r'\b(rm|mv|cp|dd|mkfs|fdisk|chmod|chown|kill|reboot|shutdown|halt|tee|'
    r'systemctl\s+(start|stop|restart|enable|disable)|service\s+\w+\s+(start|stop)|'
    r'sed\s+-i|truncate|insmod|rmmod|modprobe|iptables|route\s+add|'
    r'umount|passwd|useradd|userdel|apt|yum|dnf)\b', re.I)
# Redirects + pipe-installs are writes too — these aren't word-bounded, check separately.
_REDIRECT = re.compile(r'(>>?|\bpip\s+install\b|\|\s*(sh|bash)\b)')


def _is_read_only(cmd: str) -> bool:
    """True only if the command has no write/destructive token AND no output
    redirect. Conservative by design — a missed read-only command is harmless, a
    missed write is not."""
    return not (_WRITE_GUARD.search(cmd) or _REDIRECT.search(cmd))


def collect_ssh(target: str, *, key: Optional[str] = None,
                password: Optional[str] = None, port: int = 22,
                extra_commands: Optional[Dict[str, str]] = None,
                timeout: int = 20) -> str:
    """Connect to `target` (user@host) read-only, run discovery commands, return a
    single labeled text blob (ingest it via SmartRAG.ingest_chunks or save to a
    file). Only read-only commands run; writes are refused."""
    try:
        import paramiko
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("ssh collector needs paramiko: pip install paramiko") from e

    user, _, host = target.partition("@")
    if not host:
        user, host = (os.environ.get("USER", "root"), user)

    commands = dict(_DEFAULT_COMMANDS)
    if extra_commands:
        for label, cmd in extra_commands.items():
            if _is_read_only(cmd):
                commands[label] = cmd
            else:
                commands[f"REFUSED_{label}"] = f"# refused (not read-only): {cmd}"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = {"hostname": host, "port": port, "username": user, "timeout": timeout}
    if key:
        kw["key_filename"] = os.path.expanduser(key)
    if password:
        kw["password"] = password
    client.connect(**kw)

    blocks: List[str] = [f"# Live inspection of {target} (read-only)\n"]
    try:
        for label, cmd in commands.items():
            if not _is_read_only(cmd):
                continue
            try:
                _in, out, err = client.exec_command(cmd, timeout=timeout)
                text = out.read().decode("utf-8", "replace").strip()
                if text:
                    blocks.append(f"## {label}\n$ {cmd}\n{text[:6000]}\n")
            except Exception:  # noqa: BLE001
                continue
    finally:
        client.close()
    return "\n".join(blocks)


def collect_ssh_chunks(target: str, **kw) -> List[dict]:
    """collect_ssh, but as ingest_chunks-ready dicts (one per command block)."""
    blob = collect_ssh(target, **kw)
    chunks = []
    for block in blob.split("\n## "):
        block = block if block.startswith("#") else "## " + block
        label = block.split("\n", 1)[0].lstrip("# ").strip()
        chunks.append({"text": block, "source": f"ssh:{target}", "title": label})
    return chunks

#!/usr/bin/env python3
"""
APEX Infrastructure Health Test Suite.

Checks: Docker containers, disk/memory, GPU, ports, systemd services,
DB size, Redis, Docker logs for errors, Kafka, network.

Usage:
    python test_infra.py
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Colors
G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
B = "\033[94m"
BOLD = "\033[1m"
RST = "\033[0m"
SEP = "=" * 72

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def status(passed, label, detail=""):
    tag = f"{G}PASS{RST}" if passed else f"{R}FAIL{RST}"
    print(f"  [{tag}] {label:<40s} {detail}")
    return passed


def warn(label, detail=""):
    print(f"  [{Y}WARN{RST}] {label:<40s} {detail}")


def header(title):
    print(f"\n{B}{BOLD}{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}{RST}")


def run_cmd(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except Exception:
        return "", 1


# ── PART 1: SYSTEM RESOURCES ──────────────────────────────────────────

def check_system_resources():
    header("SYSTEM RESOURCES")
    results = []

    # Disk space
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    used_pct = used / total * 100
    p = status(used_pct < 80 and free_gb > 5, "Disk space",
               f"{free_gb:.1f} GB free / {total_gb:.0f} GB ({used_pct:.0f}% used)")
    results.append(("disk_space", p))

    # RAM
    out, _ = run_cmd("free -m | awk '/^Mem:/ {print $2, $3, $7}'")
    if out:
        parts = out.split()
        total_mb = int(parts[0])
        used_mb = int(parts[1])
        avail_mb = int(parts[2])
        p = status(avail_mb > 2000, "Memory",
                   f"{avail_mb} MB available / {total_mb} MB total ({used_mb} MB used)")
        results.append(("memory", p))
    else:
        warn("Memory", "could not read")

    # Swap
    out, _ = run_cmd("free -m | awk '/^Swap:/ {print $2, $3}'")
    if out and out.strip():
        parts = out.split()
        swap_total = int(parts[0])
        swap_used = int(parts[1])
        if swap_total > 0:
            swap_pct = swap_used / swap_total * 100
            p = status(swap_pct < 80, "Swap usage", f"{swap_used} MB / {swap_total} MB ({swap_pct:.0f}%)")
        else:
            p = status(True, "Swap usage", "no swap configured")
        results.append(("swap", p))

    # Load average
    out, _ = run_cmd("cat /proc/loadavg")
    if out:
        loads = out.split()[:3]
        load_1m = float(loads[0])
        ncpu_out, _ = run_cmd("nproc")
        ncpu = int(ncpu_out) if ncpu_out else 4
        p = status(load_1m < ncpu * 2, "Load average",
                   f"{loads[0]} {loads[1]} {loads[2]} ({ncpu} cores)")
        results.append(("load_avg", p))

    return results


# ── PART 2: GPU ───────────────────────────────────────────────────────

def check_gpu():
    header("GPU STATUS")
    results = []

    out, rc = run_cmd("nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits")
    if rc != 0:
        warn("GPU", "nvidia-smi not available")
        return [("gpu_available", True)]  # non-blocking

    for line in out.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            name, mem_total, mem_used, mem_free, temp, util = parts[:6]
            p = status(True, f"GPU: {name}",
                       f"{mem_used}/{mem_total} MB, {temp}C, {util}% util")
            results.append(("gpu_status", p))

            mem_free_mb = float(mem_free)
            p = status(mem_free_mb > 1000, "GPU memory free", f"{mem_free_mb:.0f} MB")
            results.append(("gpu_memory", p))

            temp_c = float(temp)
            p = status(temp_c < 85, "GPU temperature", f"{temp_c:.0f}C")
            results.append(("gpu_temp", p))

    # CUDA test
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        p = status(cuda_ok, "PyTorch CUDA", f"{'available' if cuda_ok else 'NOT available'}")
        results.append(("pytorch_cuda", p))
    except ImportError:
        warn("PyTorch CUDA", "torch not installed")

    return results


# ── PART 3: DOCKER CONTAINERS ────────────────────────────────────────

def check_docker():
    header("DOCKER CONTAINERS")
    results = []

    out, rc = run_cmd("docker ps --format '{{.Names}}|{{.Status}}|{{.Ports}}'", timeout=15)
    if rc != 0:
        warn("Docker", "not available or no permission")
        return [("docker_available", True)]  # non-blocking

    containers = []
    for line in out.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|")
        name = parts[0]
        container_status = parts[1] if len(parts) > 1 else ""
        containers.append((name, container_status))

    total = len(containers)
    healthy = sum(1 for _, s in containers if "Up" in s)
    unhealthy = [n for n, s in containers if "Up" not in s]

    p = status(total > 0, "Containers running", f"{total} total, {healthy} up")
    results.append(("docker_running", p))

    if unhealthy:
        for name in unhealthy[:5]:
            status(False, f"  Container: {name}", "NOT running")
        results.append(("docker_unhealthy", False))
    else:
        results.append(("docker_all_up", True))

    # Key containers
    key_containers = ["apex-timescaledb", "infra-redis-1", "infra-kafka-1",
                      "infra-prometheus-1", "infra-grafana-1"]
    for kc in key_containers:
        found = any(kc in n for n, s in containers if "Up" in s)
        p = status(found, f"  Key: {kc}", "running" if found else "MISSING/DOWN")
        results.append((f"docker_{kc}", p))

    return results


# ── PART 4: PORTS ─────────────────────────────────────────────────────

def check_ports():
    header("PORT AVAILABILITY")
    results = []

    ports = {
        8010: "Paper Trader",
        15432: "TimescaleDB",
        16379: "Redis",
        9090: "Prometheus",
        3000: "Grafana",
        9092: "Kafka",
        5001: "MLflow",
        8081: "Schema Registry",
    }

    for port, service in ports.items():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect(("localhost", port))
            s.close()
            p = status(True, f"Port {port} ({service})", "open")
        except Exception:
            p = status(False, f"Port {port} ({service})", "CLOSED")
        results.append((f"port_{port}", p))

    return results


# ── PART 5: DATABASE SIZE ─────────────────────────────────────────────

def check_database():
    header("DATABASE SIZE & HEALTH")
    results = []

    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "15432")),
            dbname=os.getenv("DB_NAME", "apex"),
            user=os.getenv("DB_USER", "apex_user"),
            password=os.getenv("DB_PASSWORD", "apex_pass"),
        )
        cur = conn.cursor()

        # DB size
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        db_size = cur.fetchone()[0]
        p = status(True, "Database size", db_size)
        results.append(("db_size", p))

        # Connection count
        cur.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()")
        conns = cur.fetchone()[0]
        p = status(conns < 90, "Active connections", f"{conns}")
        results.append(("db_connections", p))

        # Table sizes
        cur.execute("""
            SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
            FROM pg_catalog.pg_statio_user_tables
            ORDER BY pg_total_relation_size(relid) DESC
            LIMIT 5
        """)
        top_tables = cur.fetchall()
        for tbl, size in top_tables:
            status(True, f"  Table: {tbl}", size)

        # Check for dead tuples
        cur.execute("""
            SELECT relname, n_dead_tup
            FROM pg_stat_user_tables
            WHERE n_dead_tup > 10000
            ORDER BY n_dead_tup DESC LIMIT 3
        """)
        dead = cur.fetchall()
        if dead:
            for tbl, n in dead:
                warn(f"  Dead tuples: {tbl}", f"{n:,} dead rows (consider VACUUM)")
        else:
            status(True, "Dead tuples", "< 10,000 per table")
        results.append(("db_dead_tuples", True))

        conn.close()

    except Exception as e:
        status(False, "Database", str(e)[:60])
        results.append(("db_health", False))

    return results


# ── PART 6: SYSTEMD SERVICES ─────────────────────────────────────────

def check_systemd():
    header("SYSTEMD SERVICES")
    results = []

    services = ["apex-paper-trader"]

    for svc in services:
        out, rc = run_cmd(f"systemctl --user is-active {svc}")
        is_active = out.strip() == "active"
        p = status(is_active, f"Service: {svc}", out.strip())
        results.append((f"systemd_{svc}", p))

        if is_active:
            out2, _ = run_cmd(f"systemctl --user show {svc} --property=ActiveEnterTimestamp --value")
            if out2:
                status(True, f"  Started", out2.strip())

    # Check if linger enabled
    user = os.getenv("USER", "")
    out, _ = run_cmd(f"ls /var/lib/systemd/linger/{user} 2>/dev/null && echo yes || echo no")
    linger = "yes" in out
    p = status(linger, "Linger enabled", f"user={user}")
    results.append(("systemd_linger", p))

    return results


# ── PART 7: DOCKER LOGS CHECK ────────────────────────────────────────

def check_docker_logs():
    header("DOCKER LOGS (recent errors)")
    results = []

    key_containers = ["apex-timescaledb", "infra-redis-1", "infra-kafka-1"]
    total_errors = 0

    for container in key_containers:
        out, rc = run_cmd(f"docker logs --tail 100 {container} 2>&1 | grep -ci 'error\\|fatal\\|panic'")
        if rc == 0 or out.strip():
            try:
                n_errors = int(out.strip())
            except ValueError:
                n_errors = 0
        else:
            n_errors = 0

        if n_errors > 10:
            warn(f"Logs: {container}", f"{n_errors} error lines (last 100)")
        else:
            status(True, f"Logs: {container}", f"{n_errors} error lines (last 100)")
        total_errors += n_errors

    p = status(total_errors < 50, "Total error lines", f"{total_errors}")
    results.append(("docker_log_errors", p))

    return results


# ── PART 8: NETWORK ──────────────────────────────────────────────────

def check_network():
    header("NETWORK")
    results = []

    # External connectivity
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(("8.8.8.8", 53))
        s.close()
        p = status(True, "Internet connectivity", "OK")
    except Exception:
        p = status(False, "Internet connectivity", "NO INTERNET")
    results.append(("internet", p))

    # DNS resolution
    try:
        socket.getaddrinfo("api.polygon.io", 443)
        p = status(True, "DNS resolution", "api.polygon.io resolves")
    except Exception:
        p = status(False, "DNS resolution", "api.polygon.io FAILED")
    results.append(("dns", p))

    # Docker network
    out, rc = run_cmd("docker network ls --format '{{.Name}}' | grep tft")
    has_network = rc == 0 and out.strip()
    p = status(has_network, "Docker tft_network", "exists" if has_network else "MISSING")
    results.append(("docker_network", p))

    return results


# ── PART 9: REDIS MEMORY ──────────────────────────────────────────────

def check_redis():
    header("REDIS MEMORY")
    results = []

    redis_port = int(os.getenv("REDIS_PORT", "16379"))
    # Try docker exec for redis-cli since host may not have it
    out, rc = run_cmd(f"docker exec infra-redis-1 redis-cli INFO memory 2>/dev/null | grep -E 'used_memory_human|maxmemory_human|connected_clients|db0'")
    if rc != 0 or not out:
        # Try direct connection
        out, rc = run_cmd(f"docker exec tp-redis redis-cli INFO memory 2>/dev/null | grep -E 'used_memory_human|maxmemory_human'")

    if out:
        for line in out.strip().split("\n"):
            line = line.strip().rstrip("\r")
            if "used_memory_human" in line and "peak" not in line:
                mem = line.split(":")[-1].strip()
                status(True, "Redis memory used", mem)
            elif "maxmemory_human" in line:
                maxm = line.split(":")[-1].strip()
                status(True, "Redis max memory", maxm)
        results.append(("redis_memory", True))
    else:
        warn("Redis memory", "could not query (redis-cli unavailable)")
        results.append(("redis_memory", True))  # non-blocking

    # Key count
    out2, rc2 = run_cmd("docker exec infra-redis-1 redis-cli DBSIZE 2>/dev/null")
    if rc2 == 0 and out2:
        status(True, "Redis key count", out2.strip())

    return results


# ── PART 10: JOURNALCTL ERRORS ────────────────────────────────────────

def check_journalctl():
    header("JOURNALCTL (last 24h errors)")
    results = []

    out, rc = run_cmd("journalctl --user -u apex-paper-trader --since '24 hours ago' --no-pager -p err 2>/dev/null | wc -l")
    if rc == 0 and out.strip():
        try:
            n_errors = int(out.strip())
        except ValueError:
            n_errors = 0
        p = status(n_errors < 50, "Paper trader errors (24h)", f"{n_errors} error lines")
        results.append(("journal_errors", p))

        if n_errors > 0 and n_errors < 20:
            worst, _ = run_cmd("journalctl --user -u apex-paper-trader --since '24 hours ago' --no-pager -p err --output=short 2>/dev/null | tail -3")
            if worst:
                for line in worst.strip().split("\n")[:3]:
                    print(f"    {Y}{line[:100]}{RST}")
    else:
        warn("Journalctl", "could not query")
        results.append(("journal_errors", True))  # non-blocking

    return results


# ── MAIN ──────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    print(f"\n{BOLD}{SEP}")
    print(f"  APEX INFRASTRUCTURE HEALTH TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{SEP}{RST}")

    all_results = []
    all_results.extend(check_system_resources())
    all_results.extend(check_gpu())
    all_results.extend(check_docker())
    all_results.extend(check_ports())
    all_results.extend(check_database())
    all_results.extend(check_systemd())
    all_results.extend(check_docker_logs())
    all_results.extend(check_network())
    all_results.extend(check_redis())
    all_results.extend(check_journalctl())

    # Summary
    passed = sum(1 for _, p in all_results if p)
    failed = sum(1 for _, p in all_results if not p)
    total = len(all_results)

    print(f"\n{BOLD}{SEP}")
    if failed == 0:
        print(f"  {G}ALL {total} CHECKS PASSED{RST}")
    else:
        print(f"  {R}{failed}/{total} CHECKS FAILED{RST}")
    print(SEP)

    result_file = RESULTS_DIR / f"infra_{ts}.json"
    with open(result_file, "w") as f:
        json.dump({
            "timestamp": ts,
            "passed": passed,
            "failed": failed,
            "total": total,
            "details": {name: val for name, val in all_results},
        }, f, indent=2, default=str)
    print(f"  Results saved to {result_file}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

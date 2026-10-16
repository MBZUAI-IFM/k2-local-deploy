#!/usr/bin/env python3
"""Run K2 Horizon 0.9B, 3.7B or 7B locally with llama.cpp. Standard library only (Python 3.8+), macOS / Linux / Windows.

Commands (run them in this order the first time):
  python3 k2local.py check                         # what fits on this computer, for each size
  python3 k2local.py install                       # download llama.cpp for this OS
  python3 k2local.py download --size 3.7B [--quant Q]  # download + verify one size (repeat for others)
  python3 k2local.py serve [--size 3.7B]           # start a model in the background, wait until ready
  python3 k2local.py ask "question"                # send a question, print the answer and speed
  python3 k2local.py status | stop

Everything lives in one folder (default ~/k2-horizon-local, change with --dir or K2_DIR).
"""
import argparse, hashlib, json, os, platform, re, shutil, signal, subprocess, sys, tarfile, threading, time, zipfile
import urllib.request, urllib.error

LLAMA_BUILD = "b11619"            # the build this skill was tested with
# Official GGUF repos from IFM. Sizes in GB. "notes" come from our own tests on an Apple M4 Pro (24 GB).
MODELS = {
    # Note: IFM names the 0.9B files "1B" and the 3.7B files "4B".
    "0.9B": {"repo": "IFM/K2-Horizon-0.9B-GGUF", "prefix": "K2-Horizon-1B", "temp": 0.6,   # IFM recommends 0.6 for 0.9B
             "gb": {"Q4_K_M": 0.67, "Q5_K_M": 0.77, "Q6_K": 0.89, "Q8_0": 1.15},
             "speed": "~165 tokens/s (Q8, M4 Pro)", "made_for": "watches, glasses, quick commands",
             "good": "instant replies, reminders, simple tool calls, short translations",
             "weak": "slips on reasoning: picked an outdated document, split 86 three ways as 28+28+30"},
    "3.7B": {"repo": "IFM/K2-Horizon-3.7B-GGUF", "prefix": "K2-Horizon-4B", "temp": 1.0,
             "gb": {"Q4_K_M": 3.16, "Q5_K_M": 3.64, "Q6_K": 4.16, "Q8_0": 5.39},
             "speed": "~46 tokens/s (Q8, M4 Pro)", "made_for": "phones and everyday laptop use",
             "good": "summaries, document questions, tool calls, code fixes, correct math in our tests",
             "weak": "Arabic phrasing and transliteration were partly wrong"},
    "7B":   {"repo": "IFM/K2-Horizon-7B-GGUF", "prefix": "K2-Horizon-7B", "temp": 1.0,
             "gb": {"Q4_K_M": 5.59, "Q5_K_M": 6.47, "Q6_K": 7.39, "Q8_0": 9.57},
             "speed": "~40 tokens/s (Q4, M4 Pro)", "made_for": "laptops and developer workstations",
             "good": "passed every check in our 12 tests: coding, agents, reviews, research",
             "weak": "slowest of the three; needs ~16 GB+ memory (on smaller machines it uses a more compressed file)"},
}
PORT = 8080
UA = {"User-Agent": "k2-horizon-local-skill"}


# ---------------------------------------------------------------- helpers
def home():
    d = os.path.expanduser(ARGS.dir or os.environ.get("K2_DIR") or "~/k2-horizon-local")  # --dir, then K2_DIR, then default
    os.makedirs(d, exist_ok=True)
    return d


def say(msg=""):
    print(msg, flush=True)


def fetch_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.loads(r.read())


def state_path():
    return os.path.join(home(), "state.json")


def load_state():
    try:
        return json.load(open(state_path()))
    except Exception:
        return {}


def save_state(**kw):
    s = load_state(); s.update(kw)
    json.dump(s, open(state_path(), "w"), indent=1)


def keep_awake():
    """Stop the computer from sleeping during long work. Sleep stalled our downloads to almost nothing."""
    system = platform.system()
    if system == "Darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    elif system == "Windows":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED
        except Exception:
            pass
    elif system == "Linux" and shutil.which("systemd-inhibit"):
        pass  # best effort only; most Linux desktops don't sleep mid-download


def total_ram_gb():
    system = platform.system()
    try:
        if system == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
        if system == "Linux":
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 2**20
        if system == "Windows":
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2**30
    except Exception:
        pass
    return None


def gpu_kind():
    if platform.system() == "Darwin":
        return "apple-metal" if platform.machine() == "arm64" else "none"
    if shutil.which("nvidia-smi"):
        try:
            subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, timeout=10)
            return "nvidia"
        except Exception:
            pass
    if shutil.which("vulkaninfo"):
        return "vulkan"
    return "none"


def fit(gb, ram):
    """How well a file fits. Real data point: the 9.6 GB 7B Q8 file swapped to ~3 tokens/s on a 24 GB Mac
    with a browser open, while 5.6 GB ran at ~40 tokens/s. So keep the file under ~25% of total memory."""
    if ram is None:
        return "unknown"
    if gb <= ram * 0.25:
        return "comfortable"
    if gb <= ram * 0.4:
        return "tight"
    return "too big"


def best_quant(size, ram):
    """Highest quality file that fits comfortably; otherwise the smallest file (which may be tight or too big)."""
    gbs = MODELS[size]["gb"]
    for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"):
        if fit(gbs[q], ram) == "comfortable":
            return q
    return "Q4_K_M"


def options(ram):
    out = []
    for size, m in MODELS.items():
        q = best_quant(size, ram)
        out.append({"size": size, "quant": q, "file_gb": m["gb"][q], "fit": fit(m["gb"][q], ram),
                    "speed_measured": m["speed"], "made_for": m["made_for"], "good_at": m["good"], "watch_out": m["weak"]})
    return out


def recommend(opts):
    """Largest size that fits comfortably: more capable, still fast. The user can pick a smaller one for speed."""
    ok = [o for o in opts if o["fit"] == "comfortable"]
    return ok[-1]["size"] if ok else ("0.9B" if opts[0]["fit"] != "too big" else None)


# ---------------------------------------------------------------- check
def cmd_check():
    system, arch = platform.system(), platform.machine().lower()
    ram = total_ram_gb()
    free = shutil.disk_usage(home()).free / 1e9
    gpu = gpu_kind()
    opts = options(ram)
    rec = recommend(opts)
    have = downloaded_sizes()
    for o in opts:
        o["already_downloaded"] = have.get(o["size"])
    info = {"os": system, "arch": arch, "ram_gb": round(ram, 1) if ram else None, "free_disk_gb": round(free, 1),
            "gpu": gpu, "options": opts, "recommended_size": rec, "dir": home()}
    say(json.dumps(info, indent=1, ensure_ascii=False))
    say()
    say(f"Memory: {ram:.0f} GB · free disk: {free:.0f} GB · GPU: {gpu}" if ram else f"Free disk: {free:.0f} GB · GPU: {gpu}")
    mark = {"comfortable": "✓ fits comfortably", "tight": "~ fits, but close other apps", "too big": "✗ too big", "unknown": "? unknown"}
    for n, o in enumerate(opts, 1):   # always smallest → largest, so "the middle one" / "number 2" are unambiguous
        dl = f"  [downloaded: {o['already_downloaded']}]" if o["already_downloaded"] else ""
        say(f"  {n}. {o['size']:>5}  {o['quant']:<7} {o['file_gb']:>5.2f} GB  {mark[o['fit']]:<28} {o['speed_measured']}{dl}")
    say(f"\nRecommended: {rec}" if rec else "\n⚠ Even 0.9B is a tight fit here.")
    for o in opts:
        if o["fit"] != "too big" and free < o["file_gb"] + 1:
            say(f"⚠ Not enough free disk for {o['size']} ({o['file_gb']:.1f} GB).")
    if gpu == "none" and system != "Darwin":
        say("ℹ No GPU found: models run on the CPU. That works, but expect them to be several times slower.")
    save_state(check=info)


# ---------------------------------------------------------------- install llama.cpp
def choose_asset(names, backend):
    system, arch = platform.system(), platform.machine().lower()
    arm = arch in ("arm64", "aarch64")
    b = LLAMA_BUILD
    if system == "Darwin":
        return [f"llama-{b}-bin-macos-{'arm64' if arm else 'x64'}.tar.gz"]
    if system == "Linux":
        if arm:
            return [f"llama-{b}-bin-ubuntu-{'vulkan-arm64' if backend == 'vulkan' else 'arm64'}.tar.gz"]
        return {"cuda": [f"llama-{b}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"cudart-llama-{b}-bin-ubuntu-cuda-12.8-x64.tar.gz"],
                "vulkan": [f"llama-{b}-bin-ubuntu-vulkan-x64.tar.gz"]}.get(backend, [f"llama-{b}-bin-ubuntu-x64.tar.gz"])
    if system == "Windows":
        if arm:
            return [f"llama-{b}-bin-win-{'vulkan' if backend == 'vulkan' else 'cpu'}-arm64.zip"]
        return {"cuda": [f"llama-{b}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"],
                "vulkan": [f"llama-{b}-bin-win-vulkan-x64.zip"]}.get(backend, [f"llama-{b}-bin-win-cpu-x64.zip"])
    raise SystemExit(f"Unsupported OS: {system}")


def find_server(root):
    exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    for d, _, files in os.walk(root):
        if exe in files:
            return os.path.join(d, exe)
    return None


def cmd_install():
    backend = ARGS.backend
    if backend == "auto":
        g = gpu_kind()
        backend = {"nvidia": "cuda", "vulkan": "vulkan", "apple-metal": "metal"}.get(g, "cpu")  # macOS builds always include Metal
    if platform.system() == "Darwin":
        backend = "metal" if platform.machine() == "arm64" else "cpu"   # macOS builds pick Metal themselves
    root = os.path.join(home(), "llama")
    if find_server(root) and not ARGS.force:
        say(f"✓ llama.cpp already installed: {find_server(root)}"); return
    rel = fetch_json(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_BUILD}")
    urls = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    wanted = choose_asset(urls, backend)
    os.makedirs(root, exist_ok=True)
    for name in wanted:
        if name not in urls:
            raise SystemExit(f"Release {LLAMA_BUILD} has no {name}. Available: {', '.join(sorted(urls))}")
        dest = os.path.join(home(), name)
        say(f"Downloading {name} …")
        urllib.request.urlretrieve(urls[name], dest)
        if name.endswith(".zip"):
            zipfile.ZipFile(dest).extractall(root)
        else:
            with tarfile.open(dest) as t:
                t.extractall(root)
        os.remove(dest)
    server = find_server(root)
    if not server:
        raise SystemExit("Unpacked llama.cpp but couldn't find llama-server. Look inside " + root)
    if platform.system() != "Windows":
        for d, _, files in os.walk(os.path.dirname(server)):
            for f in files:
                p = os.path.join(d, f)
                os.chmod(p, os.stat(p).st_mode | 0o111)
    if platform.system() == "Darwin":
        subprocess.run(["xattr", "-dr", "com.apple.quarantine", root], stderr=subprocess.DEVNULL)  # macOS blocks downloaded binaries otherwise
    out = subprocess.run([server, "--version"], capture_output=True, text=True)
    say((out.stdout + out.stderr).strip().splitlines()[0] if (out.stdout + out.stderr).strip() else "(no version output)")
    save_state(server=server, backend=backend)
    say(f"✓ llama.cpp {LLAMA_BUILD} ({backend}) installed.")


# ---------------------------------------------------------------- download the model
def filename_note(size):
    return {"0.9B": " (IFM names the 0.9B file \"1B\")", "3.7B": " (IFM names the 3.7B file \"4B\")"}.get(size, "")


def downloaded_sizes():
    out = {}
    for size, m in (load_state().get("models") or {}).items():
        if m.get("verified") and os.path.exists(m.get("path", "")):
            out[size] = m.get("quant")
    return out


def run_hint():
    """The exact command the user should type later, including --dir if it isn't the default."""
    script = os.path.abspath(__file__)
    py = "py" if platform.system() == "Windows" else "python3"
    d = home()
    extra = "" if d == os.path.expanduser("~/k2-horizon-local") else f' --dir "{d}"'
    port = "" if ARGS.port == PORT else f" --port {ARGS.port}"
    return f'{py} "{script}"{extra}{port}'


def record_model(size, **kw):
    st = load_state()
    models = st.get("models") or {}
    models[size] = {**models.get(size, {}), **kw}
    save_state(models=models, active=size)


def cmd_download():
    size = ARGS.size
    if not size:
        raise SystemExit("Choose a size: download --size 0.9B | 3.7B | 7B   (run check to see what fits)")
    m = MODELS[size]
    quant = ARGS.quant or best_quant(size, total_ram_gb())
    name = f"{m['prefix']}-{quant}.gguf"
    dest = os.path.join(home(), name)
    prev = (load_state().get("models") or {}).get(size, {})
    if prev.get("verified") == name and os.path.exists(dest) and os.path.getsize(dest) == prev.get("verified_size"):
        say(f"✓ {size} already downloaded and verified (sha256 checked when it was downloaded): {dest}" + filename_note(size))
        record_model(size, path=dest, quant=quant); return
    meta = next(f for f in fetch_json(f"https://huggingface.co/api/models/{m['repo']}/tree/main") if f["path"] == name)
    total, sha = meta["lfs"]["size"], meta["lfs"]["oid"]
    url = f"https://huggingface.co/{m['repo']}/resolve/main/{name}"
    keep_awake()
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have > total:
        os.remove(dest); have = 0
    say(f"Downloading {name}: {total / 1e9:.2f} GB ({have / 1e9:.2f} GB already here)")

    parts = 8                      # parallel ranges: a single connection kept stalling in our tests
    remaining = total - have
    bounds = [(have + remaining * i // parts, have + remaining * (i + 1) // parts - 1) for i in range(parts)] if remaining else []
    got = [0] * len(bounds)
    errors = []

    def worker(i, start, end):
        part = f"{dest}.part{i}"
        for attempt in range(40):
            size = os.path.getsize(part) if os.path.exists(part) else 0
            if start + size > end:
                return
            try:
                req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start + size}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk); got[i] = os.path.getsize(part)
            except Exception:
                time.sleep(min(30, 2 + attempt * 2))
        errors.append(i)

    threads = [threading.Thread(target=worker, args=(i, s, e), daemon=True) for i, (s, e) in enumerate(bounds)]
    for t in threads:
        t.start()
    last, t0 = sum(got), time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        now = sum(got)
        say(f"  {(have + now) / 1e9:5.2f} / {total / 1e9:.2f} GB · {(now - last) / 15 / 1e6:5.1f} MB/s")
        last = now
    if errors:
        raise SystemExit("Some parts failed after many retries. Run the same command again: it resumes.")
    with open(dest, "ab") as out:
        for i in range(len(bounds)):
            part = f"{dest}.part{i}"
            with open(part, "rb") as p:
                shutil.copyfileobj(p, out, 1 << 24)
            os.remove(part)
    say("Verifying checksum against Hugging Face …")
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if os.path.getsize(dest) != total or h.hexdigest() != sha:
        os.rename(dest, dest + ".bad")
        raise SystemExit("✗ Checksum mismatch. The damaged file was renamed to .bad. Run download again.")
    record_model(size, path=dest, quant=quant, verified=name, verified_size=total)
    say(f"✓ Verified {name} (sha256 matches Hugging Face).{filename_note(size)} Start it with: serve --size {size}")


# ---------------------------------------------------------------- serve / stop / status
def pid_alive(pid):
    if not pid:
        return False
    if platform.system() == "Windows":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def port_in_use():
    import socket
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", ARGS.port)) == 0


def healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{ARGS.port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def cmd_serve():
    st = load_state()
    server = st.get("server")
    models = st.get("models") or {}
    size = ARGS.size or st.get("active")
    if not server or not os.path.exists(server):
        raise SystemExit("llama.cpp isn't installed yet. Run: install")
    if not size or size not in models or not os.path.exists(models[size].get("path", "")):
        have = ", ".join(models) or "none"
        raise SystemExit(f"{size or 'That size'} isn't downloaded yet (downloaded: {have}). Run: download --size {size or '3.7B'}")
    model = models[size]["path"]
    if port_in_use():
        if pid_alive(st.get("pid")) and healthy():
            if st.get("serving") == size:
                say(f"✓ {size} is already running at http://127.0.0.1:{ARGS.port}"); return
            raise SystemExit(f"{st.get('serving')} is running. Run stop first, then serve --size {size}.")
        raise SystemExit(f"✗ Port {ARGS.port} is used by another program. Stop it, or pick another port: --port 8081 (use it with ask too).")
    log = open(os.path.join(home(), "server.log"), "w")
    # --reasoning-format none: llama.cpp b11619 doesn't know K2 Horizon's <ifm|think> tags and would hide the answer.
    cmd = [server, "-m", model, "--host", "127.0.0.1", "--port", str(ARGS.port), "--jinja",
           "--reasoning-format", "none", "-c", str(ARGS.ctx), "-ngl", "99"]
    kw = {"stdout": log, "stderr": log, "cwd": os.path.dirname(server)}
    if platform.system() == "Windows":
        kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kw)
    save_state(pid=proc.pid, port=ARGS.port, serving=size, active=size)
    say(f"Starting K2-Horizon-{size} (pid {proc.pid}) …")
    t0 = time.time()
    while time.time() - t0 < 900:   # first load of a big file can be slow
        if proc.poll() is not None:
            tail = open(os.path.join(home(), "server.log")).read()[-1500:]
            raise SystemExit("✗ The server stopped while loading. Last log lines:\n" + tail)
        if healthy():
            say(f"✓ {size} ready in {time.time() - t0:.0f}s at http://127.0.0.1:{ARGS.port}  (OpenAI-compatible API at /v1)")
            h = run_hint()
            say(f"  Ask:  {h} ask \"your question\"\n  Stop: {h} stop")
            return
        time.sleep(1)
    raise SystemExit("✗ Still not ready after 15 minutes. Check server.log; the computer may be out of memory.")


def cmd_stop():
    pid = load_state().get("pid")
    if not pid or not pid_alive(pid):
        save_state(pid=None); say("Nothing running."); return
    if platform.system() == "Windows":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not pid_alive(pid):
                break
            time.sleep(0.5)
        else:
            os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)
    if pid_alive(pid):
        raise SystemExit(f"✗ Couldn't stop pid {pid}. Close it manually.")
    save_state(pid=None, serving=None)
    say(f"✓ Stopped (pid {pid}); port {ARGS.port} is free." if not port_in_use() else f"✓ Stopped pid {pid}, but something else still uses port {ARGS.port}.")


def cmd_status():
    st = load_state()
    say(json.dumps({k: st.get(k) for k in ("server", "backend", "models", "active", "serving", "pid", "port")}, indent=1))
    say(f"running ✓ ({st.get('serving')})" if healthy() else "not running")


# ---------------------------------------------------------------- ask
THINK_END = re.compile(r"</ifm\|think\w*>")


def split_output(raw):
    """Return (thinking, answer, tool_calls). Handles <ifm|think>, <ifm|think_fast>, … and <ifm|tool_call> blocks."""
    parts = THINK_END.split(raw, maxsplit=1)
    thinking, answer = (parts[0], parts[1]) if len(parts) == 2 else ("", raw)
    thinking = re.sub(r"<ifm\|think\w*>", "", thinking)
    calls = []
    for name, body in re.findall(r"<ifm\|tool_call>\s*([\w.\-]+)(.*?)</ifm\|tool_call>", answer, re.S):
        args = {k.strip(): v.strip() for k, v in re.findall(
            r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*<ifm\|arg_value>(.*?)</ifm\|arg_value>", body, re.S)}
        calls.append({"name": name, "arguments": args})
    answer = re.sub(r"<ifm\|tool_calls>.*?(</ifm\|tool_calls>|$)", "", answer, flags=re.S)
    answer = re.sub(r"</?ifm\|think\w*>", "", answer).strip()
    return thinking.strip(), answer, calls


def cmd_ask():
    if not healthy():
        raise SystemExit("The model isn't running. Run: serve")
    size = load_state().get("serving")
    temp = MODELS.get(size, {}).get("temp", 1.0)   # IFM: 0.6 for 0.9B, 1.0 for the others
    body = {"messages": [{"role": "user", "content": ARGS.prompt}], "temperature": temp, "top_p": 0.95,
            "max_tokens": ARGS.max_tokens, "chat_template_kwargs": {"reasoning_effort": ARGS.effort}}
    req = urllib.request.Request(f"http://127.0.0.1:{ARGS.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.loads(r.read())
    m = res["choices"][0]["message"]
    thinking, answer, calls = split_output((m.get("reasoning_content") or "") + (m.get("content") or ""))
    if ARGS.show_thinking and thinking:
        say("── thinking ──\n" + thinking + "\n")
    say("── answer ──\n" + (answer or "(empty)"))
    for c in calls:
        say(f"── tool call ── {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})")
    tps = res.get("timings", {}).get("predicted_per_second")
    say(f"\n{size or 'model'} · {time.time() - t0:.1f}s · {res.get('usage', {}).get('completion_tokens', '?')} tokens" + (f" · {tps:.0f} tokens/s" if tps else ""))


# ---------------------------------------------------------------- main
def main():
    global ARGS
    common = argparse.ArgumentParser(add_help=False)   # so --dir/--port work before or after the command
    common.add_argument("--dir", default=argparse.SUPPRESS, help="where everything is stored (default ~/k2-horizon-local, or $K2_DIR)")
    common.add_argument("--port", type=int, default=argparse.SUPPRESS, help=f"server port (default {PORT})")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[common])
    sub = ap.add_subparsers(dest="cmd", required=True)
    def sp(name):
        return sub.add_parser(name, parents=[common])
    sp("check")
    p = sp("install"); p.add_argument("--backend", default="auto", choices=["auto", "cpu", "cuda", "vulkan"]); p.add_argument("--force", action="store_true")
    p = sp("download"); p.add_argument("--size", choices=list(MODELS)); p.add_argument("--quant", choices=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"])
    p = sp("serve"); p.add_argument("--size", choices=list(MODELS)); p.add_argument("--ctx", type=int, default=16384)
    p = sp("ask"); p.add_argument("prompt"); p.add_argument("--effort", default="medium", choices=["low", "medium", "high"],
                   help="the model's own default is high; ask uses medium so quick checks are fast")
    p.add_argument("--max-tokens", type=int, default=4096); p.add_argument("--show-thinking", action="store_true")
    sp("status"); sp("stop")
    ARGS = ap.parse_args()
    ARGS.dir = getattr(ARGS, "dir", None)
    ARGS.port = getattr(ARGS, "port", PORT)
    {"check": cmd_check, "install": cmd_install, "download": cmd_download, "serve": cmd_serve,
     "ask": cmd_ask, "status": cmd_status, "stop": cmd_stop}[ARGS.cmd]()


if __name__ == "__main__":
    main()

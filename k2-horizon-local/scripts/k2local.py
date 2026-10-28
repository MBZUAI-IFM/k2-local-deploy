#!/usr/bin/env python3
"""Run K2 Horizon 0.9B, 3.7B or 7B locally. Standard library only (Python 3.8+), macOS / Linux / Windows.

Two formats, both run with llama.cpp: gguf (compressed, the default) and gguf-bf16 (full quality).

Commands (run them in this order the first time):
  python3 k2local.py check                         # what fits on this computer, for each size and format
  python3 k2local.py install                       # latest llama.cpp for this OS and GPU
  python3 k2local.py download --size 3.7B [--format F] [--quant Q]  # download + verify (repeat for others)
  python3 k2local.py serve [--size 3.7B] [--format F]  # start a model in the background, wait until ready
  python3 k2local.py ask "question"                # send a question, print the answer and speed
  python3 k2local.py chat [--window]               # chat in the terminal (--window: in a new terminal window)
  python3 k2local.py status | stop

Everything lives in one folder (default ~/k2-horizon-local, change with --dir or K2_DIR).
"""
import argparse, hashlib, json, os, platform, re, shutil, signal, subprocess, sys, tarfile, threading, time, zipfile
import urllib.request, urllib.error

TESTED_BUILD = "b11619"           # the llama.cpp build this skill was tested with; the fallback if the latest fails
LLAMA_RELEASES = "https://github.com/ggml-org/llama.cpp/releases"
# Official GGUF repos from IFM. Sizes in GB. "notes" come from our own tests on an Apple M4 Pro (24 GB).
MODELS = {
    # Note: IFM names the 0.9B files "1B" and the 3.7B files "4B".
    "0.9B": {"repo": "IFM/K2-Horizon-0.9B-GGUF", "prefix": "K2-Horizon-1B", "temp": 0.6,   # IFM recommends 0.6 for 0.9B
             "gb": {"Q4_K_M": 0.67, "Q5_K_M": 0.77, "Q6_K": 0.89, "Q8_0": 1.15, "BF16": 2.16},
             "speed": "~165 tokens/s (Q8, M4 Pro)", "made_for": "watches, glasses, quick commands",
             "good": "instant replies, reminders, simple tool calls, short translations",
             "weak": "slips on reasoning: picked an outdated document, split 86 three ways as 28+28+30"},
    "3.7B": {"repo": "IFM/K2-Horizon-3.7B-GGUF", "prefix": "K2-Horizon-4B", "temp": 1.0,
             "gb": {"Q4_K_M": 3.16, "Q5_K_M": 3.64, "Q6_K": 4.16, "Q8_0": 5.39, "BF16": 10.13},
             "speed": "~46 tokens/s (Q8, M4 Pro)", "made_for": "phones and everyday laptop use",
             "good": "summaries, document questions, tool calls, code fixes, correct math in our tests",
             "weak": "Arabic phrasing and transliteration were partly wrong"},
    "7B":   {"repo": "IFM/K2-Horizon-7B-GGUF", "prefix": "K2-Horizon-7B", "temp": 1.0,
             "gb": {"Q4_K_M": 5.59, "Q5_K_M": 6.47, "Q6_K": 7.39, "Q8_0": 9.57, "BF16": 18.01},
             "speed": "~40 tokens/s (Q4, M4 Pro)", "made_for": "laptops and developer workstations",
             "good": "passed every check in our 12 tests: coding, agents, reviews, research",
             "weak": "slowest of the three; needs ~16 GB+ memory (on smaller machines it uses a more compressed file)"},
}
# One sentence each, shown by check so the user can choose. Speed comparisons are from 0.9B on an M1 Pro (16 GB):
# Q8 GGUF ~105 tokens/s, BF16 GGUF ~74.
FORMATS = {
    "gguf": ("GGUF, compressed",
             "A compressed single file run by llama.cpp: the smallest download and the fastest replies on any computer, with a small loss in quality."),
    "gguf-bf16": ("GGUF, full quality (BF16)",
                  "The full-precision weights in one GGUF file for llama.cpp: original quality with nothing extra to install, but about twice the size of compressed and ~30% slower."),
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


def format_options(ram):
    """For each format, the file size and fit of every size. The same 25% rule applies to full-quality GGUF."""
    have = load_state().get("models") or {}
    out = []
    for fmt, (title, blurb) in FORMATS.items():
        sizes = []
        for size, m in MODELS.items():
            key = model_key(size, fmt)
            gb = m["gb"]["BF16"] if fmt == "gguf-bf16" else m["gb"][best_quant(size, ram)]
            sizes.append({"size": size, "gb": gb, "fit": fit(gb, ram), "downloaded": bool(have.get(key, {}).get("verified"))})
        out.append({"format": fmt, "title": title, "about": blurb, "sizes": sizes})
    return out


# ---------------------------------------------------------------- formats in the state file
def model_key(size, fmt):
    """State key for one downloaded model: "7B" (compressed GGUF) or "7B-bf16"."""
    return {"gguf": size, "gguf-bf16": f"{size}-bf16"}[fmt]


def key_size(key):
    return key.split("-")[0]


def key_format(key):
    return {"bf16": "gguf-bf16"}.get(key.partition("-")[2], "gguf")


def label(key):
    """"7B" or "7B BF16 GGUF", for messages."""
    extra = {"gguf": "", "gguf-bf16": " BF16 GGUF"}[key_format(key)]
    return key_size(key) + extra


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
    formats = format_options(ram)
    info = {"os": system, "arch": arch, "ram_gb": round(ram, 1) if ram else None, "free_disk_gb": round(free, 1),
            "gpu": gpu, "options": opts, "recommended_size": rec, "formats": formats, "default_format": "gguf", "dir": home()}
    say(json.dumps(info, indent=1, ensure_ascii=False))
    say()
    say(f"Memory: {ram:.0f} GB · free disk: {free:.0f} GB · GPU: {gpu}" if ram else f"Free disk: {free:.0f} GB · GPU: {gpu}")
    mark = {"comfortable": "✓ fits comfortably", "tight": "~ fits, but close other apps", "too big": "✗ too big", "unknown": "? unknown"}
    for n, o in enumerate(opts, 1):   # always smallest → largest, so "the middle one" / "number 2" are unambiguous
        dl = f"  [downloaded: {o['already_downloaded']}]" if o["already_downloaded"] else ""
        say(f"  {n}. {o['size']:>5}  {o['quant']:<7} {o['file_gb']:>5.2f} GB  {mark[o['fit']]:<28} {o['speed_measured']}{dl}")
    say(f"\nRecommended: {rec}" if rec else "\n⚠ Even 0.9B is a tight fit here.")
    say("\nFormats (the list above uses a. GGUF, compressed, the default):")
    sym = {"comfortable": "✓", "tight": "~", "too big": "✗", "unknown": "?"}
    for letter, f in zip("ab", formats):
        say(f"  {letter}. {f['title']}: {f['about']}")
        if f["format"] != "gguf":
            say("       " + " · ".join(f"{s['size']} {s['gb']:.1f} GB {sym[s['fit']]}" + (" [downloaded]" if s["downloaded"] else "")
                                     for s in f["sizes"]))
    for o in opts:
        if o["fit"] != "too big" and free < o["file_gb"] + 1:
            say(f"⚠ Not enough free disk for {o['size']} ({o['file_gb']:.1f} GB).")
    if gpu == "none" and system != "Darwin":
        say("ℹ No GPU found: models run on the CPU. That works, but expect them to be several times slower.")
    save_state(check=info)


# ---------------------------------------------------------------- install llama.cpp
def latest_build():
    """The newest llama.cpp release, from GitHub's /releases/latest redirect rather than the API: the API allows only
    60 anonymous calls an hour per network, which a shared office network used up. Stable releases (v0.x) name their
    build in nightly-tag.txt. None if it can't be found."""
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{LLAMA_RELEASES}/latest", headers=UA), timeout=30) as r:
            tag = r.geturl().rstrip("/").rsplit("/", 1)[-1]
        if not re.fullmatch(r"b\d+", tag):
            with urllib.request.urlopen(urllib.request.Request(f"{LLAMA_RELEASES}/download/{tag}/nightly-tag.txt",
                                                               headers=UA), timeout=30) as r:
                tag = r.read().decode().strip()
        return tag if re.fullmatch(r"b\d+", tag) else None
    except Exception:
        return None


def fetch_llama(build, backend):
    """Download this build's archives into the storage folder. The names follow a fixed pattern, so no API call is
    needed. Returns the archive paths, or None if the build has no such file (404)."""
    got = []
    for name in choose_asset(build, backend):
        url = f"{LLAMA_RELEASES}/download/{build}/{name}"
        dest = os.path.join(home(), name)
        say(f"Downloading {name} …")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.HTTPError as e:
            for p in got:
                os.remove(p)
            if e.code == 404:
                return None
            raise SystemExit(f"✗ Couldn't download {url}: {e}")
        except OSError as e:
            raise SystemExit(f"✗ Couldn't download {url}: {e}. Check the internet connection and run install again.")
        got.append(dest)
    return got


def choose_asset(build, backend):
    system, arch = platform.system(), platform.machine().lower()
    arm = arch in ("arm64", "aarch64")
    b = build
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
    st = load_state()
    if find_server(root) and not ARGS.force:
        say(f"✓ llama.cpp {st.get('build') or ''} already installed: {find_server(root)}  (to update: install --force)"); return
    if find_server(root) and pid_alive(st.get("pid")):
        raise SystemExit("The model is running from the installed llama.cpp. Run stop first, then install --force.")
    if ARGS.build == "latest":
        latest = latest_build()
        if not latest:
            say(f"Couldn't look up the latest llama.cpp release, so using {TESTED_BUILD}, the build tested with this skill.")
        builds = [b for b in (latest, TESTED_BUILD) if b]
    else:
        builds = [TESTED_BUILD if ARGS.build == "tested" else ARGS.build]
    for build in builds:
        archives = fetch_llama(build, backend)
        if archives:
            break
        if build != builds[-1]:
            say(f"llama.cpp {build} has no download for this computer; using {TESTED_BUILD}, the build tested with this skill.")
    else:
        raise SystemExit(f"✗ llama.cpp {builds[-1]} has no {' + '.join(choose_asset(builds[-1], backend))}. "
                         f"See {LLAMA_RELEASES} or try --backend cpu.")
    shutil.rmtree(root, ignore_errors=True)   # an update must not leave the old llama-server next to the new one
    os.makedirs(root)
    for dest in archives:
        if dest.endswith(".zip"):
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
    save_state(server=server, backend=backend, build=build)
    say(f"✓ llama.cpp {build} ({backend}) installed" + (" (the build tested with this skill)." if build == TESTED_BUILD else "."))


# ---------------------------------------------------------------- download the model
def filename_note(size):
    return {"0.9B": " (IFM names the 0.9B file \"1B\")", "3.7B": " (IFM names the 3.7B file \"4B\")"}.get(size, "")


def downloaded_sizes():
    """Compressed GGUF downloads only, size -> quant (check's main list is about those)."""
    out = {}
    for key, m in (load_state().get("models") or {}).items():
        if key_format(key) == "gguf" and m.get("verified") and os.path.exists(m.get("path", "")):
            out[key] = m.get("quant")
    return out


def run_hint(py=None):
    """The exact command the user should type later, including --dir if it isn't the default."""
    script = os.path.abspath(__file__)
    py = py or ("py" if platform.system() == "Windows" else "python3")
    d = home()
    extra = "" if d == os.path.expanduser("~/k2-horizon-local") else f' --dir "{d}"'
    port = "" if ARGS.port == PORT else f" --port {ARGS.port}"
    return f'{py} "{script}"{extra}{port}'


def record_model(key, **kw):
    st = load_state()
    models = st.get("models") or {}
    models[key] = {**models.get(key, {}), **kw}
    save_state(models=models, active=key)


def fetch_file(url, dest, total, sha=None):
    """Download one file with 8 parallel ranges, resuming whatever is already there, then check its sha256."""
    keep_awake()
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have > total:
        os.remove(dest); have = 0
    say(f"Downloading {os.path.basename(dest)}: {total / 1e9:.2f} GB ({have / 1e9:.2f} GB already here)")

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
    last, t_last = sum(got), time.time()
    while any(t.is_alive() for t in threads):
        for _ in range(15):        # report every 15 s, but notice a finished small file right away
            time.sleep(1)
            if not any(t.is_alive() for t in threads):
                break
        now, t_now = sum(got), time.time()
        say(f"  {(have + now) / 1e9:5.2f} / {total / 1e9:.2f} GB · {(now - last) / (t_now - t_last) / 1e6:5.1f} MB/s")
        last, t_last = now, t_now
    if errors:
        raise SystemExit("Some parts failed after many retries. Run the same command again: it resumes.")
    with open(dest, "ab") as out:
        for i in range(len(bounds)):
            part = f"{dest}.part{i}"
            with open(part, "rb") as p:
                shutil.copyfileobj(p, out, 1 << 24)
            os.remove(part)
    if sha:
        say("Verifying checksum against Hugging Face …")
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        if os.path.getsize(dest) != total or h.hexdigest() != sha:
            os.rename(dest, dest + ".bad")
            raise SystemExit(f"✗ Checksum mismatch in {os.path.basename(dest)}. The damaged file was renamed to .bad. Run download again.")
    elif os.path.getsize(dest) != total:
        os.remove(dest)
        raise SystemExit(f"✗ {os.path.basename(dest)} came out the wrong size. Run download again.")


def cmd_download():
    size = ARGS.size
    if not size:
        raise SystemExit("Choose a size: download --size 0.9B | 3.7B | 7B   (run check to see what fits)")
    fmt = "gguf-bf16" if ARGS.quant == "BF16" else ARGS.format
    if fmt == "gguf-bf16" and ARGS.quant not in (None, "BF16"):
        raise SystemExit("--format gguf-bf16 is the BF16 file; leave out --quant (or use --format gguf with it).")
    m = MODELS[size]
    key = model_key(size, fmt)
    quant = "BF16" if fmt == "gguf-bf16" else ARGS.quant or best_quant(size, total_ram_gb())
    name = f"{m['prefix']}-{quant}.gguf"
    dest = os.path.join(home(), name)
    prev = (load_state().get("models") or {}).get(key, {})
    if prev.get("verified") == name and os.path.exists(dest) and os.path.getsize(dest) == prev.get("verified_size"):
        say(f"✓ {label(key)} already downloaded and verified (sha256 checked when it was downloaded): {dest}" + filename_note(size))
        record_model(key, path=dest, quant=quant); return
    meta = next(f for f in fetch_json(f"https://huggingface.co/api/models/{m['repo']}/tree/main") if f["path"] == name)
    total, sha = meta["lfs"]["size"], meta["lfs"]["oid"]
    fetch_file(f"https://huggingface.co/{m['repo']}/resolve/main/{name}", dest, total, sha)
    record_model(key, path=dest, quant=quant, verified=name, verified_size=total)
    fmt_flag = "" if fmt == "gguf" else f" --format {fmt}"
    say(f"✓ Verified {name} (sha256 matches Hugging Face).{filename_note(size)} Start it with: serve --size {size}{fmt_flag}")


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


def gguf_models(st):
    """Downloaded models, leaving out "-original" entries that older versions of this script could record."""
    return {k: v for k, v in (st.get("models") or {}).items() if not k.endswith("-original")}


def pick_key(st):
    """Which downloaded model to serve. --format wins; otherwise the last one used for that size,
    then compressed GGUF, then BF16 GGUF."""
    models = gguf_models(st)
    active = st.get("active") if st.get("active") in models else None
    size = ARGS.size or (key_size(active) if active else None)
    if not size:
        return None
    if ARGS.format:
        return model_key(size, ARGS.format)
    if active and key_size(active) == size:
        return active
    return next((k for k in (size, f"{size}-bf16") if k in models), size)


def cmd_serve():
    st = load_state()
    models = gguf_models(st)
    key = pick_key(st)
    if not key or key not in models or not os.path.exists(models[key].get("path", "")):
        have = ", ".join(label(k) for k in models) or "none"
        size = key_size(key) if key else "3.7B"
        fmt_flag = f" --format {key_format(key)}" if key and key_format(key) != "gguf" else ""
        raise SystemExit(f"{label(key) if key else 'That size'} isn't downloaded yet (downloaded: {have}). "
                         f"Run: download --size {size}{fmt_flag}")
    server = st.get("server")
    if not server or not os.path.exists(server):
        raise SystemExit("llama.cpp isn't installed yet. Run: install")
    model = models[key]["path"]
    if port_in_use():
        if pid_alive(st.get("pid")) and healthy():
            if st.get("serving") == key:
                say(f"✓ {label(key)} is already running at http://127.0.0.1:{ARGS.port}")
                return say_how_to_use()
            raise SystemExit(f"{label(st.get('serving') or '?')} is running. Run stop first, then serve again.")
        raise SystemExit(f"✗ Port {ARGS.port} is used by another program. Stop it, or pick another port: --port 8081 (use it with ask too).")
    log = open(os.path.join(home(), "server.log"), "w")
    # llama.cpp recognizes K2 Horizon's tags: replies come back as reasoning_content, content and tool_calls.
    cmd = [server, "-m", model, "--host", "127.0.0.1", "--port", str(ARGS.port), "--jinja",
           "-c", str(ARGS.ctx), "-ngl", "99"]
    kw = {"stdout": log, "stderr": log, "cwd": os.path.dirname(server)}
    if platform.system() == "Windows":
        kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kw)
    save_state(pid=proc.pid, port=ARGS.port, serving=key, active=key)
    say(f"Starting K2-Horizon-{label(key)} (pid {proc.pid}) …")
    t0 = time.time()
    while time.time() - t0 < 900:   # first load of a big file can be slow
        if proc.poll() is not None:
            tail = open(os.path.join(home(), "server.log")).read()[-1500:]
            raise SystemExit("✗ The server stopped while loading. Last log lines:\n" + tail)
        if healthy():
            say(f"✓ {label(key)} ready in {time.time() - t0:.0f}s at http://127.0.0.1:{ARGS.port}  (OpenAI-compatible API at /v1)")
            return say_how_to_use()
        time.sleep(1)
    raise SystemExit("✗ Still not ready after 15 minutes. Check server.log; the computer may be out of memory.")


def say_how_to_use():
    """The ways to talk to the running model. llama-server also serves a chat page at its address."""
    h = run_hint()
    say(f"  Web chat: http://127.0.0.1:{ARGS.port}  (open it in a browser)")
    say(f"  Chat: {h} chat\n  Ask:  {h} ask \"your question\"\n  Stop: {h} stop")


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
    say(json.dumps({k: st.get(k) for k in ("server", "build", "backend", "models", "active", "serving", "pid", "port")}, indent=1))
    say(f"running ✓ ({st.get('serving')})" if healthy() else "not running")


# ---------------------------------------------------------------- ask
THINK_END = re.compile(r"</ifm\|think\w*>")


def read_message(m):
    """(thinking, answer, tool_calls) from a reply message. The server already splits K2 Horizon's reasoning,
    answer and tool calls. One exception, seen with 3.7B at medium effort: it opens with <ifm|think_fast> but closes
    with </ifm|think>, so the server never finds the end and returns everything as reasoning, with an empty answer.
    The wrong tag stays in the text, so split it here."""
    reasoning, content = m.get("reasoning_content") or "", m.get("content") or ""
    if THINK_END.search(reasoning) or "<ifm|" in content:
        return split_output(reasoning + content)
    calls = []
    for c in m.get("tool_calls") or []:
        fn = c.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = fn.get("arguments")
        calls.append({"name": fn.get("name"), "arguments": args})
    return reasoning.strip(), content.strip(), calls


def split_output(raw):
    """Split raw K2 Horizon text into (thinking, answer, tool_calls). Accepts any closing tag (<ifm|think>,
    <ifm|think_fast>, …), which also covers 3.7B's mismatched medium tag, and parses <ifm|tool_call> blocks."""
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
    key = load_state().get("serving")
    size = label(key) if key else None
    temp = MODELS.get(key_size(key) if key else "", {}).get("temp", 1.0)   # IFM: 0.6 for 0.9B, 1.0 for the others
    body = {"messages": [{"role": "user", "content": ARGS.prompt}], "temperature": temp, "top_p": 0.95,
            "max_tokens": ARGS.max_tokens, "chat_template_kwargs": {"reasoning_effort": ARGS.effort}}
    req = urllib.request.Request(f"http://127.0.0.1:{ARGS.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.loads(r.read())
    m = res["choices"][0]["message"]
    thinking, answer, calls = read_message(m)
    if ARGS.show_thinking and thinking:
        say("── thinking ──\n" + thinking + "\n")
    say("── answer ──\n" + (answer or "(empty)"))
    for c in calls:
        say(f"── tool call ── {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})")
    tps = res.get("timings", {}).get("predicted_per_second")
    say(f"\n{size or 'model'} · {time.time() - t0:.1f}s · {res.get('usage', {}).get('completion_tokens', '?')} tokens" + (f" · {tps:.0f} tokens/s" if tps else ""))


# ---------------------------------------------------------------- chat
def cmd_chat():
    if not healthy():
        raise SystemExit("The model isn't running. Run: serve")
    if ARGS.window:
        return open_chat_window()
    try:
        import readline   # arrow keys and history while typing; not available on Windows
    except ImportError:
        pass
    key = load_state().get("serving")
    temp = MODELS.get(key_size(key) if key else "", {}).get("temp", 1.0)
    effort = ARGS.effort
    say(f"K2 Horizon {label(key) if key else ''} · {effort} thinking · runs privately on this computer")
    say("Type a message and press Enter. /new starts over, /effort low|medium|high changes the thinking, /exit quits.\n")
    history = []
    while True:
        try:
            text = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            say(); break
        if not text:
            continue
        if text in ("/exit", "/quit", "exit", "quit"):
            break
        if text == "/new":
            history = []; say("(started over)\n"); continue
        if text.startswith("/effort"):
            v = text.split()[1:]
            if v and v[0] in ("low", "medium", "high"):
                effort = v[0]; say(f"(thinking: {effort})\n")
            else:
                say("Use /effort low, /effort medium or /effort high\n")
            continue
        history.append({"role": "user", "content": text})
        try:
            history.append({"role": "assistant", "content": stream_reply(history, temp, effort)})
        except KeyboardInterrupt:
            history.pop(); say("\n(stopped)\n")


def stream_reply(messages, temp, effort):
    """Print the answer as it arrives and return it. While the model thinks, show a 'thinking…' placeholder."""
    body = {"messages": messages, "temperature": temp, "top_p": 0.95, "max_tokens": ARGS.max_tokens, "stream": True,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    req = urllib.request.Request(f"http://127.0.0.1:{ARGS.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    reasoning, content, thinking, started, tps = "", "", False, False, None
    out = sys.stdout
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            tps = (chunk.get("timings") or {}).get("predicted_per_second") or tps
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning += delta["reasoning_content"]
                if not thinking and not started:
                    out.write("k2 › thinking…"); out.flush(); thinking = True
            if delta.get("content"):
                if not started:
                    out.write(("\r" + " " * 20 + "\r" if thinking else "") + "k2 › "); started = True
                content += delta["content"]; out.write(delta["content"]); out.flush()
    _, answer, _ = read_message({"reasoning_content": reasoning, "content": content})
    if not started:   # 3.7B at medium effort: the whole reply arrives as reasoning (see read_message)
        out.write(("\r" + " " * 20 + "\r" if thinking else "") + "k2 › " + (answer or "(empty)"))
    say(f"\n   ({tps:.0f} tokens/s)\n" if tps else "\n")
    return answer


def open_chat_window():
    """Start the chat in a new terminal window, so an assistant that runs commands in the background can hand the
    user a chat they can type into."""
    py = f'"{sys.executable}"'   # the Python running now, in case python3 isn't on the new window's PATH
    cmd = f"{run_hint(py)} chat --effort {ARGS.effort}"
    system, ok = platform.system(), False
    if system == "Darwin":
        path = os.path.join(home(), "chat.command")   # Terminal opens .command files; no extra permission needed
        with open(path, "w") as f:
            f.write(f"#!/bin/sh\nclear\n{cmd}\n")
        os.chmod(path, 0o755)
        ok = subprocess.run(["open", path]).returncode == 0
    elif system == "Windows":
        path = os.path.join(home(), "chat.bat")
        with open(path, "w") as f:
            f.write(f"@echo off\r\ntitle K2 Horizon\r\n{cmd}\r\n")
        os.startfile(path); ok = True
    else:
        for term in (["x-terminal-emulator", "-e"], ["gnome-terminal", "--"], ["konsole", "-e"], ["xterm", "-e"]):
            if shutil.which(term[0]):
                subprocess.Popen(term + ["sh", "-c", cmd], start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                ok = True; break
    say("✓ Opened the chat in a new terminal window." if ok else "✗ Couldn't open a terminal window on this computer.")
    say(f"  To start it yourself, run in a terminal: {run_hint()} chat")


# ---------------------------------------------------------------- main
def build_arg(s):
    if re.fullmatch(r"latest|tested|b\d+", s):
        return s
    raise argparse.ArgumentTypeError(f"use latest, tested or a build like {TESTED_BUILD}")


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
    fmts = list(FORMATS)
    p = sp("install"); p.add_argument("--backend", default="auto", choices=["auto", "cpu", "cuda", "vulkan"]); p.add_argument("--force", action="store_true")
    p.add_argument("--build", default="latest", type=build_arg, help=f"llama.cpp build: latest (default), tested ({TESTED_BUILD}), or a specific one like {TESTED_BUILD}")
    p = sp("download"); p.add_argument("--size", choices=list(MODELS)); p.add_argument("--format", default="gguf", choices=fmts)
    p.add_argument("--quant", choices=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "BF16"], help="GGUF file to use (BF16 = --format gguf-bf16)")
    p = sp("serve"); p.add_argument("--size", choices=list(MODELS)); p.add_argument("--format", choices=fmts, help="default: the last one used for that size")
    p.add_argument("--ctx", type=int, default=16384)
    p = sp("ask"); p.add_argument("prompt"); p.add_argument("--effort", default="medium", choices=["low", "medium", "high"],
                   help="the model's own default is high; ask uses medium so quick checks are fast")
    p.add_argument("--max-tokens", type=int, default=4096); p.add_argument("--show-thinking", action="store_true")
    p = sp("chat"); p.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--max-tokens", type=int, default=8192); p.add_argument("--window", action="store_true", help="open the chat in a new terminal window")
    sp("status"); sp("stop")
    ARGS = ap.parse_args()
    ARGS.dir = getattr(ARGS, "dir", None)
    ARGS.port = getattr(ARGS, "port", PORT)
    {"check": cmd_check, "install": cmd_install, "download": cmd_download, "serve": cmd_serve,
     "ask": cmd_ask, "chat": cmd_chat, "status": cmd_status, "stop": cmd_stop}[ARGS.cmd]()


if __name__ == "__main__":
    main()

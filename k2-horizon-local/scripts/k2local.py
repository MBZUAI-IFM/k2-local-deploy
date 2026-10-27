#!/usr/bin/env python3
"""Run K2 Horizon 0.9B, 3.7B or 7B locally. Standard library only (Python 3.8+), macOS / Linux / Windows.

Three formats: gguf (compressed, the default) and gguf-bf16 (full quality) run with llama.cpp;
original runs IFM's safetensors files with PyTorch + Transformers in a private Python environment.

Commands (run them in this order the first time):
  python3 k2local.py check                         # what fits on this computer, for each size and format
  python3 k2local.py install [--format original]   # llama.cpp for this OS (or the Python environment for original)
  python3 k2local.py download --size 3.7B [--format F] [--quant Q]  # download + verify (repeat for others)
  python3 k2local.py serve [--size 3.7B] [--format F]  # start a model in the background, wait until ready
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
             "original": "IFM/K2-Horizon-0.9B", "original_gb": 2.16,
             "gb": {"Q4_K_M": 0.67, "Q5_K_M": 0.77, "Q6_K": 0.89, "Q8_0": 1.15, "BF16": 2.16},
             "speed": "~165 tokens/s (Q8, M4 Pro)", "made_for": "watches, glasses, quick commands",
             "good": "instant replies, reminders, simple tool calls, short translations",
             "weak": "slips on reasoning: picked an outdated document, split 86 three ways as 28+28+30"},
    "3.7B": {"repo": "IFM/K2-Horizon-3.7B-GGUF", "prefix": "K2-Horizon-4B", "temp": 1.0,
             "original": "IFM/K2-Horizon-3.7B", "original_gb": 10.14,
             "gb": {"Q4_K_M": 3.16, "Q5_K_M": 3.64, "Q6_K": 4.16, "Q8_0": 5.39, "BF16": 10.13},
             "speed": "~46 tokens/s (Q8, M4 Pro)", "made_for": "phones and everyday laptop use",
             "good": "summaries, document questions, tool calls, code fixes, correct math in our tests",
             "weak": "Arabic phrasing and transliteration were partly wrong"},
    "7B":   {"repo": "IFM/K2-Horizon-7B-GGUF", "prefix": "K2-Horizon-7B", "temp": 1.0,
             "original": "IFM/K2-Horizon-7B", "original_gb": 18.02,
             "gb": {"Q4_K_M": 5.59, "Q5_K_M": 6.47, "Q6_K": 7.39, "Q8_0": 9.57, "BF16": 18.01},
             "speed": "~40 tokens/s (Q4, M4 Pro)", "made_for": "laptops and developer workstations",
             "good": "passed every check in our 12 tests: coding, agents, reviews, research",
             "weak": "slowest of the three; needs ~16 GB+ memory (on smaller machines it uses a more compressed file)"},
}
# One sentence each, shown by check so the user can choose. Speed comparisons are from 0.9B on an M1 Pro (16 GB):
# Q8 GGUF ~105 tokens/s, BF16 GGUF ~74, original (PyTorch on the GPU) ~29.
FORMATS = {
    "gguf": ("GGUF, compressed",
             "A compressed single file run by llama.cpp: the smallest download and the fastest replies on any computer, with a small loss in quality."),
    "gguf-bf16": ("GGUF, full quality (BF16)",
                  "The full-precision weights in one GGUF file for llama.cpp: original quality with nothing extra to install, but about twice the size of compressed and ~30% slower."),
    "original": ("Original (Transformers)",
                 "IFM's original files exactly as published, run with PyTorch and Transformers: the reference version for research, fine-tuning and Python tools, but it needs a one-time ~1 GB install and is ~3-4x slower than compressed."),
}
# The versions IFM validated on the model pages. Installed only for the original format, in a private environment.
ORIGINAL_PACKAGES = ["torch==2.13.0", "transformers==5.15.0", "safetensors==0.8.0"]
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
    """For each format, the file size and fit of every size. Full-quality GGUF and original are the same weights,
    so they need the same memory; the same 25% rule applies (both load the whole model into memory)."""
    have = load_state().get("models") or {}
    out = []
    for fmt, (title, blurb) in FORMATS.items():
        sizes = []
        for size, m in MODELS.items():
            key = model_key(size, fmt)
            gb = m["original_gb"] if fmt == "original" else m["gb"]["BF16"] if fmt == "gguf-bf16" else m["gb"][best_quant(size, ram)]
            sizes.append({"size": size, "gb": gb, "fit": fit(gb, ram), "downloaded": bool(have.get(key, {}).get("verified"))})
        out.append({"format": fmt, "title": title, "about": blurb, "sizes": sizes})
    return out


# ---------------------------------------------------------------- formats in the state file
def model_key(size, fmt):
    """State key for one downloaded model: "7B" (compressed GGUF), "7B-bf16" or "7B-original"."""
    return {"gguf": size, "gguf-bf16": f"{size}-bf16", "original": f"{size}-original"}[fmt]


def key_size(key):
    return key.split("-")[0]


def key_format(key):
    return {"bf16": "gguf-bf16", "original": "original"}.get(key.partition("-")[2], "gguf")


def label(key):
    """"7B", "7B BF16 GGUF" or "7B original", for messages."""
    extra = {"gguf": "", "gguf-bf16": " BF16 GGUF", "original": " original"}[key_format(key)]
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
    for letter, f in zip("abc", formats):
        say(f"  {letter}. {f['title']}: {f['about']}")
        if f["format"] != "gguf":
            say("       " + " · ".join(f"{s['size']} {s['gb']:.1f} GB {sym[s['fit']]}" + (" [downloaded]" if s["downloaded"] else "")
                                     for s in f["sizes"]))
    if not original_python_ok():
        say("     (c needs Python 3.10+ and a one-time install: install --format original)")
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
    if ARGS.format == "original":
        return install_original()
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


# ---------------------------------------------------------------- install the Python environment (original format)
def original_python():
    env = os.path.join(home(), "python")
    return os.path.join(env, "Scripts", "python.exe") if platform.system() == "Windows" else os.path.join(env, "bin", "python")


def original_python_ok():
    py = original_python()
    return os.path.exists(py) and subprocess.run([py, "-c", "import torch, transformers, safetensors"],
                                                 capture_output=True).returncode == 0


def python_310():
    """A Python 3.10+ to build the environment from. PyTorch and Transformers no longer support older versions."""
    if sys.version_info >= (3, 10):
        return [sys.executable]
    for name in ("python3.13", "python3.12", "python3.11", "python3.10"):
        if shutil.which(name):
            return [shutil.which(name)]
    if platform.system() == "Windows" and shutil.which("py"):
        for v in ("3.13", "3.12", "3.11", "3.10"):
            if subprocess.run(["py", f"-{v}", "--version"], capture_output=True).returncode == 0:
                return ["py", f"-{v}"]
    raise SystemExit("The original format needs Python 3.10 or newer, for PyTorch. Install it from python.org, "
                     "then run install --format original again. (The GGUF formats work with your current Python.)")


def install_original():
    py = original_python()
    if original_python_ok() and not ARGS.force:
        say(f"✓ Python environment for the original format already installed: {os.path.dirname(os.path.dirname(py))}")
        return report_torch(py)
    env = os.path.join(home(), "python")
    say(f"Creating a private Python environment in {env} (nothing is installed system-wide) …")
    subprocess.run(python_310() + ["-m", "venv", "--clear", env], check=True)
    say(f"Installing {', '.join(ORIGINAL_PACKAGES)}, the versions IFM validated. This downloads 0.5-3 GB, depending on your system …")
    for step in (["install", "--upgrade", "pip"], ["install", *ORIGINAL_PACKAGES]):
        if subprocess.run([py, "-m", "pip", "--disable-pip-version-check", *step]).returncode != 0:
            raise SystemExit("✗ pip couldn't install the packages (see above). Check your internet connection and run "
                             "install --format original --force to retry.")
    report_torch(py)


def report_torch(py):
    out = subprocess.run([py, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__, "
                          "'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("✗ The environment is broken:\n" + out.stderr[-1500:] + "\nRun install --format original --force.")
    torch_v, tf_v, device = out.stdout.split()
    save_state(original_python=py, original_device=device)
    say(f"✓ PyTorch {torch_v} + Transformers {tf_v} ready. The original format will run on: {device}"
        + ("  (CPU only: it works, but slowly)" if device == "cpu" else ""))


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


def run_hint():
    """The exact command the user should type later, including --dir if it isn't the default."""
    script = os.path.abspath(__file__)
    py = "py" if platform.system() == "Windows" else "python3"
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
    if fmt == "original":
        return download_original(size)
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


def download_original(size):
    """Every file of IFM's original repo (weights, config, tokenizer, IFM's model code) into one folder.
    Weight files are checked against their sha256; the small text files against their size."""
    if ARGS.quant:
        raise SystemExit("--quant is for the GGUF formats. The original format has one version per size.")
    repo = MODELS[size]["original"]
    key = model_key(size, "original")
    folder = os.path.join(home(), repo.split("/")[1])
    os.makedirs(folder, exist_ok=True)
    files = [f for f in fetch_json(f"https://huggingface.co/api/models/{repo}/tree/main")
             if f["type"] == "file" and not f["path"].endswith((".md", ".png", ".jpg"))]   # skip the model card and images
    total = sum(f["size"] for f in files)
    marker = os.path.join(folder, ".verified.json")   # files already checked, so a resumed download doesn't rehash them
    try:
        done = json.load(open(marker))
    except Exception:
        done = {}
    if all(done.get(f["path"]) == f["size"] and os.path.exists(os.path.join(folder, f["path"])) for f in files):
        say(f"✓ {label(key)} already downloaded and verified: {folder}")
        record_model(key, path=folder, verified=True, verified_size=total); return
    say(f"Downloading {repo} (original format): {len(files)} files, {total / 1e9:.2f} GB in total")
    for f in files:
        dest = os.path.join(folder, f["path"])
        if done.get(f["path"]) == f["size"] and os.path.exists(dest):
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{f['path']}"
        if "lfs" in f:
            fetch_file(url, dest, f["size"], f["lfs"]["oid"])
        else:
            urllib.request.urlretrieve(url, dest)
            if os.path.getsize(dest) != f["size"]:
                raise SystemExit(f"✗ {f['path']} came out the wrong size. Run download again.")
        done[f["path"]] = f["size"]
        json.dump(done, open(marker, "w"), indent=1)
    record_model(key, path=folder, verified=True, verified_size=total)
    say(f"✓ Verified all {len(files)} files (weights by sha256). Start it with: serve --size {size} --format original")
    if not original_python_ok():
        say("  First run the one-time Python setup: install --format original")


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


def pick_key(st):
    """Which downloaded model to serve. --format wins; otherwise the last one used for that size,
    then compressed GGUF, then BF16 GGUF, then original."""
    models = st.get("models") or {}
    active = st.get("active")
    size = ARGS.size or (key_size(active) if active else None)
    if not size:
        return None
    if ARGS.format:
        return model_key(size, ARGS.format)
    if active and key_size(active) == size and active in models:
        return active
    return next((k for k in (size, f"{size}-bf16", f"{size}-original") if k in models), size)


def cmd_serve():
    st = load_state()
    models = st.get("models") or {}
    key = pick_key(st)
    if not key or key not in models or not os.path.exists(models[key].get("path", "")):
        have = ", ".join(label(k) for k in models) or "none"
        size = key_size(key) if key else "3.7B"
        fmt_flag = f" --format {key_format(key)}" if key and key_format(key) != "gguf" else ""
        raise SystemExit(f"{label(key) if key else 'That size'} isn't downloaded yet (downloaded: {have}). "
                         f"Run: download --size {size}{fmt_flag}")
    fmt = key_format(key)
    if fmt == "original":
        py = original_python()
        if not original_python_ok():
            raise SystemExit("The Python environment for the original format isn't installed yet. Run: install --format original")
    else:
        server = st.get("server")
        if not server or not os.path.exists(server):
            raise SystemExit("llama.cpp isn't installed yet. Run: install")
    model = models[key]["path"]
    if port_in_use():
        if pid_alive(st.get("pid")) and healthy():
            if st.get("serving") == key:
                say(f"✓ {label(key)} is already running at http://127.0.0.1:{ARGS.port}"); return
            raise SystemExit(f"{label(st.get('serving') or '?')} is running. Run stop first, then serve again.")
        raise SystemExit(f"✗ Port {ARGS.port} is used by another program. Stop it, or pick another port: --port 8081 (use it with ask too).")
    log = open(os.path.join(home(), "server.log"), "w")
    if fmt == "original":
        cmd = [py, os.path.join(os.path.dirname(os.path.abspath(__file__)), "original_server.py"),
               "--model", model, "--port", str(ARGS.port)]
        kw = {"stdout": log, "stderr": log, "cwd": home(), "env": {**os.environ, "PYTHONUNBUFFERED": "1"}}
    else:
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
    fmts = list(FORMATS)
    p = sp("install"); p.add_argument("--backend", default="auto", choices=["auto", "cpu", "cuda", "vulkan"]); p.add_argument("--force", action="store_true")
    p.add_argument("--format", default="gguf", choices=fmts, help="original: set up PyTorch + Transformers instead of llama.cpp")
    p = sp("download"); p.add_argument("--size", choices=list(MODELS)); p.add_argument("--format", default="gguf", choices=fmts)
    p.add_argument("--quant", choices=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "BF16"], help="GGUF file to use (BF16 = --format gguf-bf16)")
    p = sp("serve"); p.add_argument("--size", choices=list(MODELS)); p.add_argument("--format", choices=fmts, help="default: the last one used for that size")
    p.add_argument("--ctx", type=int, default=16384)
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

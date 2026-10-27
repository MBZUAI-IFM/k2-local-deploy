---
name: k2-horizon-local
description: Install and run K2 Horizon (MBZUAI / IFM) locally on the user's own Mac, Linux or Windows computer, in the size that fits their machine (0.9B, 3.7B or 7B) and the format they want (compressed GGUF, full-quality BF16 GGUF, or IFM's original safetensors with Transformers). Checks the computer, lets the user pick the size and format, then downloads, starts and tests it. Use this whenever someone wants to run, deploy, install, self-host, try offline, or set up K2 Horizon (or "K2 0.9B / 3.7B / 7B", "IFM model") on their own laptop, desktop or device, or asks which K2 Horizon size their computer can handle, even if they don't mention llama.cpp or GGUF. Also use it to switch sizes, restart, test, stop or troubleshoot an existing local K2 Horizon setup.
---

# Run K2 Horizon locally, in the size that fits

This skill takes someone from nothing to a working, private copy of K2 Horizon on their own computer. It checks what the machine can handle, lets the **user** pick the size and format, then installs, downloads, starts and tests it. Everything goes through one helper script, `k2local.py`, which step 0 fetches from GitHub if it isn't next to this file, so this file is all a user needs to share. The script uses only Python's standard library, so it works on macOS, Linux and Windows with any AI assistant that can run commands. (The original format is the exception: it installs PyTorch and Transformers into a private environment, see step 3.)

The guidance comes from real runs on an Apple M4 Pro with 24 GB of memory: all three sizes were tested with the same 12 tasks. Where a step exists because something went wrong in testing, the reason is given so you can adapt.

## The three sizes

| Size | Made for | File (Q4 → Q8) | Measured speed* | Good at | Watch out |
|---|---|---|---|---|---|
| **0.9B** | Watches, glasses, quick commands | 0.7 → 1.2 GB | ~165 tokens/s | Instant replies, reminders, simple tool calls | Slips on reasoning (picked an outdated document; split 86 as 28+28+30) |
| **3.7B** | Phones, everyday laptop use | 3.2 → 5.4 GB | ~46 tokens/s | Summaries, document Q&A, tool calls, code fixes | Arabic phrasing was partly wrong |
| **7B** | Laptops, developer workstations | 5.6 → 9.6 GB | ~40 tokens/s | Passed every check in our tests | Slowest of the three; needs ~16 GB+ memory |

\*On an M4 Pro Mac. A token is about ¾ of a word. Machines without a GPU are several times slower.

IFM names the 0.9B files "1B" and the 3.7B files "4B" (for example `K2-Horizon-4B-Q8_0.gguf`). If a user sees those names, explain that they're the same models.

## The three formats

All three are IFM's official files. The table above is for the default, compressed GGUF.

| Format | `--format` | Download (0.9B / 3.7B / 7B) | Runs with | Speed vs compressed* |
|---|---|---|---|---|
| **GGUF, compressed** (default) | `gguf` | 0.7–1.2 / 3.2–5.4 / 5.6–9.6 GB | llama.cpp | 1× (fastest) |
| **GGUF, full quality (BF16)** | `gguf-bf16` | 2.2 / 10.1 / 18.0 GB | llama.cpp | ~0.7× |
| **Original** | `original` | 2.2 / 10.1 / 18.0 GB | PyTorch + Transformers | ~0.3× |

\*0.9B on an M1 Pro (16 GB): Q8 GGUF ~105 tokens/s, BF16 GGUF ~74, original ~29.

One sentence each, to tell the user (`check` prints the same):
- **GGUF, compressed:** a compressed single file run by llama.cpp: the smallest download and the fastest replies on any computer, with a small loss in quality.
- **GGUF, full quality (BF16):** the full-precision weights in one GGUF file for llama.cpp: original quality with nothing extra to install, but about twice the size and ~30% slower.
- **Original:** IFM's original files exactly as published, run with PyTorch and Transformers: the reference version for research, fine-tuning and Python tools, but it needs a one-time ~1 GB install and is ~3–4× slower.

BF16 and original load the whole full-precision model into memory, so the same 25% rule applies and they're much bigger: on 16 GB only 0.9B fits. In testing, 3.7B original on a 16 GB Mac loaded, then swapped 12 GB to disk and barely produced output.

## Before you start

- You need Python 3.8+ (`python3 --version`, or `py --version` on Windows) and internet for the downloads.
- In the commands below, `k2local.py` stands for the script's full path from step 0. Files go in `~/k2-horizon-local`. To change that, use `--dir` (which wins) or the `K2_DIR` environment variable. `--dir` and `--port` work before or after the command name.
- If a previous run left files there, the script reuses them (`install` and `download` say "already"). Tell the user they were reused.

## Steps

### 0. Get the script
- If this file sits in a folder that has `scripts/k2local.py` next to it (the user downloaded the whole repository), use that path.
- Otherwise, download the two scripts from the project's GitHub repository into `~/k2-horizon-local/scripts`. This single command works on macOS, Linux and Windows (use `py` instead of `python3` on Windows) and prints the folder:
```
python3 -c "import os,urllib.request as u;d=os.path.join(os.path.expanduser('~'),'k2-horizon-local','scripts');os.makedirs(d,exist_ok=True);[u.urlretrieve('https://raw.githubusercontent.com/iayazedan/run-k2-horizon-locally/main/k2-horizon-local/scripts/'+f,os.path.join(d,f)) for f in ('k2local.py','original_server.py')];print(d)"
```
Tell the user in one line what you downloaded and from where: two small Python scripts from github.com/iayazedan/run-k2-horizon-locally. If they're already there from an earlier run, reuse them. Run the command again only to update them. Keep `original_server.py` in the same folder as `k2local.py`: the original format needs it.

### 1. Check the computer
```
python3 k2local.py check
```
This prints the memory, free disk and GPU, then one numbered line per size: which file it would use, how well it fits, the measured speed, and whether it's already downloaded. It also gives a recommendation. Below that, it lists the three formats (a, b, c) with one sentence each, and for BF16 and original, how well each size fits. Sizes that are already downloaded start in seconds, which is worth mentioning when the user chooses. "Fits comfortably" means the file is at most ~25% of the computer's memory. That rule comes from testing: on 24 GB, a 5.6 GB file ran at ~40 tokens/s, while a 9.6 GB file swapped to disk and dropped to ~3 tokens/s with a browser open.

### 2. Let the user choose the size
This is the user's decision, so present the options instead of picking silently. Show a short version of the `check` result in plain words, for example:

> Your Mac has 24 GB of memory, so all three sizes fit:
> 1. **0.9B** (1.2 GB): fastest, good for quick commands, but makes reasoning mistakes.
> 2. **3.7B** (5.4 GB): a good balance. It's a bit faster than 7B and got our summary, document and math tests right.
> 3. **7B** (5.6 GB): the most capable, and passed every test. *Recommended.*
>
> Which one would you like? You can add another size later.

Keep the options numbered, smallest to largest, so answers like "the middle one" or "2" are clear. Compare the actual file sizes on this machine instead of assuming bigger models mean bigger downloads. On memory-limited machines, 7B uses a more compressed file, so it can be about the same size as 3.7B. The real trade-off is speed versus capability.

Leave out sizes marked "too big", or mention them only to say they won't run well. If a size is "tight", say it will work if they close other apps. If the user says "you pick", or you can't ask (for example in an automated run), use the recommended size and say so. Users can download several sizes and switch between them later.

Then offer the format, in one short message with the three one-sentence explanations (from `check` or the table above). Compressed GGUF is the default and the right answer for most people: if the user doesn't care, or you can't ask, use it without making a fuss. Only list BF16 or original for sizes `check` marks as fitting. If the user mentions research, fine-tuning, comparing against the reference model, or using it from PyTorch, suggest original; if they want full quality without extra installs, suggest BF16.

### 3. Install
```
python3 k2local.py install                       # for the GGUF formats (compressed or BF16)
python3 k2local.py install --format original     # for the original format
```
The first downloads llama.cpp build b11619, the one tested with this skill, choosing the right build for the OS and GPU: Metal on Apple Silicon, CUDA for NVIDIA, Vulkan or CPU otherwise. On macOS it also removes the "downloaded from the internet" quarantine flag that would otherwise block the program. To override the choice, use `--backend cpu|cuda|vulkan`, for example if the CUDA build fails to start.

`--format original` instead creates a private Python environment in `~/k2-horizon-local/python` and installs the versions IFM validated: PyTorch 2.13.0, Transformers 5.15.0 and Safetensors 0.8.0 (about 850 MB on a Mac; more on Linux, where PyTorch includes CUDA). Nothing is installed system-wide. It needs Python 3.10 or newer, and finds one if the current Python is older. At the end it prints where the model will run: `cuda`, `mps` (Apple GPU) or `cpu`. Install both if the user wants to try both kinds of format.

### 4. Download the chosen size and format
```
python3 k2local.py download --size 3.7B                        # compressed GGUF; optional: --quant Q4_K_M | Q5_K_M | Q6_K | Q8_0
python3 k2local.py download --size 0.9B --format gguf-bf16      # full-quality GGUF
python3 k2local.py download --size 0.9B --format original       # IFM's original files
```
This downloads the official files from IFM's Hugging Face page. For compressed GGUF it picks the best quality that fits comfortably unless you pass `--quant`. For original, it downloads the whole model folder (weights, config, tokenizer and IFM's model code; 3.7B and 7B are split into 36 weight files) and checks every weight file's SHA-256. A rerun skips files that are already verified. It uses 8 parallel connections, keeps the computer awake, and verifies the SHA-256 checksum. For the bigger sizes this is the longest step, so run it in the background if you can and report progress. If it stops, the same command resumes it. Single connections stalled and a sleeping Mac froze downloads in testing, which is why the script works this way.

### 5. Start it
```
python3 k2local.py serve --size 3.7B                  # optional: --format gguf | gguf-bf16 | original
```
This starts the model in the background at `http://127.0.0.1:8080` (local only) and waits until it's ready, usually a few seconds. Without `--format`, it uses the format last used for that size.

The GGUF formats run in llama-server. The original format runs in `original_server.py`, a small server using the private Python environment. It offers the same API (`/v1/chat/completions`, with streaming and tools, plus `/health`), so `ask` and everything in [using-from-code.md](https://raw.githubusercontent.com/iayazedan/run-k2-horizon-locally/main/k2-horizon-local/references/using-from-code.md) work the same way.

K2 Horizon wraps its thinking in tags that change with the effort: `<ifm|think>` (high), `<ifm|think_fast>` (medium) and `<ifm|think_faster>` (low). llama.cpp recognizes them, so replies come back split: the thinking in `reasoning_content`, the answer in `content`, and tool calls in `tool_calls`. `original_server.py` returns the same fields.

One known exception: at medium effort, 3.7B opens with `<ifm|think_fast>` but closes with `</ifm|think>`, so llama.cpp never finds the end of the thinking. The whole reply lands in `reasoning_content` and `content` is empty (tool calls too). `ask` detects this and recovers the answer; [using-from-code.md](https://raw.githubusercontent.com/iayazedan/run-k2-horizon-locally/main/k2-horizon-local/references/using-from-code.md) shows the same fix for the user's own code. This was reported to IFM.

Only one model runs at a time. To switch size or format, run `stop` and then `serve --size <size> --format <format>`.

### 6. Test it
```
python3 k2local.py ask "Split 86 equally between 3 people. How much does each get?"
```
Expect 28⅔ (about 28.67) and a speed line. Answers vary a little between runs: 7B was right every time in testing, and 3.7B was right most times (once it gave "28 each, with 2 left over"). 0.9B may get it wrong. That's a known limit of the smallest model, not a setup problem, so mention it if it happens. Keep the test question unambiguous: "Split 86 between 3" sometimes gets a whole-number split.

Show the user the answer and the speed. If they like, ask one question of their own.

### 7. Hand it over
Tell the user, in plain language:
- Which size is running, at `http://127.0.0.1:8080`, and that their data stays on their machine.
- To chat: use the exact `Ask:` command that `serve` printed. It includes the full script path, plus `--dir` or `--port` when they aren't the defaults, so the user can copy it as is. It uses `medium` thinking by default, which is fast. Add `--effort high` for hard questions (the model's own default, slower and more careful) or `--effort low` for quick ones.
- To use it from code or other apps: it speaks the OpenAI chat API at `/v1`. See [using-from-code.md](https://raw.githubusercontent.com/iayazedan/run-k2-horizon-locally/main/k2-horizon-local/references/using-from-code.md).
- To try another size or format: `download --size … [--format …]`, then `stop`, then `serve --size … [--format …]`.
- To stop: `stop`. It confirms the server exited and the port is free. To start again later, run `serve`; install and download only happen once.
- To remove everything: delete the `~/k2-horizon-local` folder.

## When something goes wrong
Read `references/troubleshooting.md` next to this file or, if it isn't there, [troubleshooting.md](https://raw.githubusercontent.com/iayazedan/run-k2-horizon-locally/main/k2-horizon-local/references/troubleshooting.md) on GitHub. It covers the problems actually seen: empty answers, slow output, stalled downloads, macOS blocking the program, the server quitting on load, and a busy port. `k2local.py status` shows what's downloaded, which size is running, and the port. `server.log` in the storage folder has the server's own messages.

## Recommended settings (from IFM's model pages)
- `top_p 0.95`. `temperature 1.0`, except **0.9B, which uses 0.6**. `ask` sets this automatically.
- `reasoning_effort`: `high` (default), `medium` or `low`, sent as `chat_template_kwargs: {"reasoning_effort": "…"}`.
- Give hard tasks plenty of output tokens. IFM suggests up to 32,768.

---
name: k2-horizon-local
description: Install and run K2 Horizon (MBZUAI / IFM) locally on the user's own Mac, Linux or Windows computer, in the size that fits their machine (0.9B, 3.7B or 7B) and the GGUF format they want (compressed, or full-quality BF16). Checks the computer, lets the user pick the size and format, then downloads, starts and tests it. Use this whenever someone wants to run, deploy, install, self-host, try offline, or set up K2 Horizon (or "K2 0.9B / 3.7B / 7B", "IFM model") on their own laptop, desktop or device, or asks which K2 Horizon size their computer can handle, even if they don't mention llama.cpp or GGUF. Also use it to switch sizes, restart, test, stop or troubleshoot an existing local K2 Horizon setup.
---

# Run K2 Horizon locally, in the size that fits

This skill takes someone from nothing to a working, private copy of K2 Horizon on their own computer. It checks what the machine can handle, lets the **user** pick the size and format, then installs, downloads, starts and tests it. Everything goes through one helper script, `k2local.py`, which step 0 fetches from GitHub if it isn't next to this file, so this file is all a user needs to share. The script uses only Python's standard library, so it works on macOS, Linux and Windows with any AI assistant that can run commands. Nothing else gets installed besides llama.cpp, which runs the model.

The guidance comes from real runs on an Apple M4 Pro with 24 GB of memory: all three sizes were tested with the same 12 tasks. Where a step exists because something went wrong in testing, the reason is given so you can adapt.

## The three sizes

| Size | Made for | File (Q4 → Q8) | Measured speed* | Good at | Watch out |
|---|---|---|---|---|---|
| **0.9B** | Watches, glasses, quick commands | 0.7 → 1.2 GB | ~165 tokens/s | Instant replies, reminders, simple tool calls | Slips on reasoning (picked an outdated document; split 86 as 28+28+30) |
| **3.7B** | Phones, everyday laptop use | 3.2 → 5.4 GB | ~46 tokens/s | Summaries, document Q&A, tool calls, code fixes | Arabic phrasing was partly wrong |
| **7B** | Laptops, developer workstations | 5.6 → 9.6 GB | ~40 tokens/s | Passed every check in our tests | Slowest of the three; needs ~16 GB+ memory |

\*On an M4 Pro Mac. A token is about ¾ of a word. Machines without a GPU are several times slower.

IFM names the 0.9B files "1B" and the 3.7B files "4B" (for example `K2-Horizon-4B-Q8_0.gguf`). If a user sees those names, explain that they're the same models.

## The two formats

Both are IFM's official GGUF files, run by llama.cpp. The table above is for the default, compressed GGUF.

| Format | `--format` | Download (0.9B / 3.7B / 7B) | Speed vs compressed* |
|---|---|---|---|
| **GGUF, compressed** (default) | `gguf` | 0.7–1.2 / 3.2–5.4 / 5.6–9.6 GB | 1× (fastest) |
| **GGUF, full quality (BF16)** | `gguf-bf16` | 2.2 / 10.1 / 18.0 GB | ~0.7× |

\*0.9B on an M1 Pro (16 GB): Q8 GGUF ~105 tokens/s, BF16 GGUF ~74.

One sentence each, to tell the user (`check` prints the same):
- **GGUF, compressed:** a compressed single file run by llama.cpp: the smallest download and the fastest replies on any computer, with a small loss in quality.
- **GGUF, full quality (BF16):** the full-precision weights in one GGUF file for llama.cpp: original quality with nothing extra to install, but about twice the size and ~30% slower.

BF16 loads the whole full-precision model into memory, so the same 25% rule applies and it's much bigger: on 16 GB only 0.9B fits. In testing, a 10 GB full-precision 3.7B on a 16 GB Mac loaded, then swapped 12 GB to disk and barely produced output.

## Before you start

- You need Python 3.8+ (`python3 --version`, or `py --version` on Windows) and internet for the downloads.
- In the commands below, `k2local.py` stands for the script's full path from step 0. Files go in `~/k2-horizon-local`. To change that, use `--dir` (which wins) or the `K2_DIR` environment variable. `--dir` and `--port` work before or after the command name.
- If a previous run left files there, the script reuses them (`install` and `download` say "already"). Tell the user they were reused.

## Steps

### 0. Get the script
- If this file sits in a folder that has `scripts/k2local.py` next to it (the user downloaded the whole repository), use that path.
- Otherwise, download the script from the project's GitHub repository into `~/k2-horizon-local/scripts`. This single command works on macOS, Linux and Windows (use `py` instead of `python3` on Windows) and prints the folder:
```
python3 -c "import os,urllib.request as u;d=os.path.join(os.path.expanduser('~'),'k2-horizon-local','scripts');os.makedirs(d,exist_ok=True);u.urlretrieve('https://raw.githubusercontent.com/MBZUAI-IFM/k2-local-deploy/main/k2-horizon-local/scripts/k2local.py',os.path.join(d,'k2local.py'));print(d)"
```
Tell the user in one line what you downloaded and from where: one small Python script from github.com/MBZUAI-IFM/k2-local-deploy. If it's already there from an earlier run, reuse it. Run the command again only to update it.

### 1. Check the computer
```
python3 k2local.py check
```
This prints the memory, free disk and GPU, then one numbered line per size: which file it would use, how well it fits, the measured speed, and whether it's already downloaded. It also gives a recommendation. Below that, it lists the two formats (a, b) with one sentence each, and for BF16, how well each size fits. Sizes that are already downloaded start in seconds, which is worth mentioning when the user chooses. "Fits comfortably" means the file is at most ~25% of the computer's memory. That rule comes from testing: on 24 GB, a 5.6 GB file ran at ~40 tokens/s, while a 9.6 GB file swapped to disk and dropped to ~3 tokens/s with a browser open.

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

Then offer the format, in one short message with the two one-sentence explanations (from `check` or the table above). Compressed GGUF is the default and the right answer for most people: if the user doesn't care, or you can't ask, use it without making a fuss. Only offer BF16 for sizes `check` marks as fitting, and suggest it if the user wants full quality.

### 3. Install
```
python3 k2local.py install
```
This downloads llama.cpp build b11619, the one K2 Horizon was tested with, choosing the right file for the OS and GPU: Metal on Apple Silicon, CUDA for NVIDIA, Vulkan or CPU otherwise. It downloads straight from llama.cpp's GitHub releases page without using the GitHub API, which allows only 60 anonymous calls an hour per network and failed on a shared office network. It never updates by itself. Only if the user asks for a newer llama.cpp, run `stop`, then `install --build latest` (or a build number such as `--build b11669`), and tell them it hasn't been tested with K2 Horizon. `install --build tested` goes back to b11619. On macOS it also removes the "downloaded from the internet" quarantine flag that would otherwise block the program. To override the choice, use `--backend cpu|cuda|vulkan`, for example if the CUDA build fails to start.

### 4. Download the chosen size and format
```
python3 k2local.py download --size 3.7B                        # compressed GGUF; optional: --quant Q4_K_M | Q5_K_M | Q6_K | Q8_0
python3 k2local.py download --size 0.9B --format gguf-bf16      # full-quality GGUF
```
This downloads the official files from IFM's Hugging Face page. For compressed GGUF it picks the best quality that fits comfortably unless you pass `--quant`. It uses 8 parallel connections, keeps the computer awake, and verifies the SHA-256 checksum. For the bigger sizes this is the longest step, so run it in the background if you can and report progress. If it stops, the same command resumes it. Single connections stalled and a sleeping Mac froze downloads in testing, which is why the script works this way.

### 5. Start it
```
python3 k2local.py serve --size 3.7B                  # optional: --format gguf | gguf-bf16
```
This starts the model in the background at `http://127.0.0.1:8080` (local only) and waits until it's ready, usually a few seconds. It sets IFM's recommended settings as the defaults for everything that talks to it, including the web chat and other apps: high reasoning, `top_p` 0.95, and temperature 1.0 (0.6 for 0.9B). Without `--format`, it uses the format last used for that size. It then prints how to use it: the web chat link (llama-server has a chat page at its address), and the exact `chat`, `ask` and `stop` commands.

The model runs in llama-server, which offers an OpenAI-compatible API (`/v1/chat/completions`, with streaming and tools, plus `/health`), used by `ask`, `chat` and everything in [using-from-code.md](https://raw.githubusercontent.com/MBZUAI-IFM/k2-local-deploy/main/k2-horizon-local/references/using-from-code.md).

K2 Horizon wraps its thinking in tags that change with the effort: `<ifm|think>` (high), `<ifm|think_fast>` (medium) and `<ifm|think_faster>` (low). llama.cpp recognizes them, so replies come back split: the thinking in `reasoning_content`, the answer in `content`, and tool calls in `tool_calls`.

One known exception: at medium effort, 3.7B opens with `<ifm|think_fast>` but closes with `</ifm|think>`, so llama.cpp never finds the end of the thinking. The whole reply lands in `reasoning_content` and `content` is empty (tool calls too). `ask` detects this and recovers the answer; [using-from-code.md](https://raw.githubusercontent.com/MBZUAI-IFM/k2-local-deploy/main/k2-horizon-local/references/using-from-code.md) shows the same fix for the user's own code. This was reported to IFM.

Only one model runs at a time. To switch size or format, run `stop` and then `serve --size <size> --format <format>`.

### 6. Test it
```
python3 k2local.py ask "Split 86 equally between 3 people. How much does each get?"
```
Expect 28⅔ (about 28.67) and a speed line. Answers vary a little between runs: 7B was right every time in testing, and 3.7B was right most times (once it gave "28 each, with 2 left over"). 0.9B may get it wrong. That's a known limit of the smallest model, not a setup problem, so mention it if it happens. Keep the test question unambiguous: "Split 86 between 3" sometimes gets a whole-number split.

Show the user the answer and the speed. If they like, ask one question of their own.

### 7. Hand it over
First open a chat for the user in a new terminal window:
```
python3 k2local.py chat --window
```
You run commands in the background, so a chat started there can't take the user's typing. `--window` opens a separate terminal window where the user talks to the model directly (Terminal on macOS, a console window on Windows, the desktop's terminal on Linux). If it prints that it couldn't open a window, give the user the command it printed to run in their own terminal.

Then tell the user, in plain language:
- Which size is running, and that their data stays on their machine.
- **The chat window:** type a message and press Enter. `/new` starts a new conversation, `/effort low|medium|high` changes how much it thinks (`high` by default, the most careful; `low` is faster for quick questions), and `/exit` closes it. To open it again later, use the exact `Chat:` command that `serve` printed. It includes the full script path, plus `--dir` or `--port` when they aren't the defaults, so the user can copy it as is.
- **The web chat:** give the `Web chat:` link that `serve` printed, normally `http://127.0.0.1:8080`, to open in a browser. It only works on this computer, and only while the model is running.
- For a single question from the command line, the `Ask:` command that `serve` printed.
- To use it from code or other apps: it speaks the OpenAI chat API at `/v1`. See [using-from-code.md](https://raw.githubusercontent.com/MBZUAI-IFM/k2-local-deploy/main/k2-horizon-local/references/using-from-code.md).
- To try another size or format: `download --size … [--format …]`, then `stop`, then `serve --size … [--format …]`.
- To stop: `stop`. It confirms the server exited and the port is free. To start again later, run `serve`; install and download only happen once.
- To remove everything: delete the `~/k2-horizon-local` folder.

## When something goes wrong
Read `references/troubleshooting.md` next to this file or, if it isn't there, [troubleshooting.md](https://raw.githubusercontent.com/MBZUAI-IFM/k2-local-deploy/main/k2-horizon-local/references/troubleshooting.md) on GitHub. It covers the problems actually seen: empty answers, slow output, stalled downloads, GitHub's rate limit, macOS blocking the program, the server quitting on load, and a busy port. `k2local.py status` shows what's downloaded, which size is running, and the port. `server.log` in the storage folder has the server's own messages.

## Recommended settings (from IFM's model pages)
- `top_p 0.95`. `temperature 1.0`, except **0.9B, which uses 0.6**. `serve` makes these the server's defaults.
- `reasoning_effort`: `high` (default), `medium` or `low`, sent as `chat_template_kwargs: {"reasoning_effort": "…"}`.
- Give hard tasks plenty of output tokens. IFM suggests up to 32,768.

---
name: k2-horizon-local
description: Install and run K2 Horizon (MBZUAI / IFM) locally on the user's own Mac, Linux or Windows computer, in the size that fits their machine (0.9B, 3.7B or 7B), with llama.cpp and the official GGUF files. Checks the computer, lets the user pick the size, then downloads, starts and tests it. Use this whenever someone wants to run, deploy, install, self-host, try offline, or set up K2 Horizon (or "K2 0.9B / 3.7B / 7B", "IFM model") on their own laptop, desktop or device, or asks which K2 Horizon size their computer can handle, even if they don't mention llama.cpp or GGUF. Also use it to switch sizes, restart, test, stop or troubleshoot an existing local K2 Horizon setup.
---

# Run K2 Horizon locally, in the size that fits

This skill takes someone from nothing to a working, private copy of K2 Horizon on their own computer. It checks what the machine can handle, lets the **user** pick the size, then installs, downloads, starts and tests it. Everything goes through one helper script, `scripts/k2local.py`. It uses only Python's standard library, so it works on macOS, Linux and Windows with any AI assistant that can run commands.

The guidance comes from real runs on an Apple M4 Pro with 24 GB of memory: all three sizes were tested with the same 12 tasks. Where a step exists because something went wrong in testing, the reason is given so you can adapt.

## The three sizes

| Size | Made for | File (Q4 → Q8) | Measured speed* | Good at | Watch out |
|---|---|---|---|---|---|
| **0.9B** | Watches, glasses, quick commands | 0.7 → 1.2 GB | ~165 tokens/s | Instant replies, reminders, simple tool calls | Slips on reasoning (picked an outdated document; split 86 as 28+28+30) |
| **3.7B** | Phones, everyday laptop use | 3.2 → 5.4 GB | ~46 tokens/s | Summaries, document Q&A, tool calls, code fixes | Arabic phrasing was partly wrong |
| **7B** | Laptops, developer workstations | 5.6 → 9.6 GB | ~40 tokens/s | Passed every check in our tests | Slowest of the three; needs ~16 GB+ memory |

\*On an M4 Pro Mac. A token is about ¾ of a word. Machines without a GPU are several times slower.

IFM names the 0.9B files "1B" and the 3.7B files "4B" (for example `K2-Horizon-4B-Q8_0.gguf`). If a user sees those names, explain that they're the same models.

## Before you start

- You need Python 3.8+ (`python3 --version`, or `py --version` on Windows) and internet for the downloads.
- Run the script by its path, for example `python3 <skill-dir>/scripts/k2local.py check`. Files go in `~/k2-horizon-local`. To change that, use `--dir` (which wins) or the `K2_DIR` environment variable. `--dir` and `--port` work before or after the command name.
- If a previous run left files there, the script reuses them (`install` and `download` say "already"). Tell the user they were reused.

## Steps

### 1. Check the computer
```
python3 scripts/k2local.py check
```
This prints the memory, free disk and GPU, then one numbered line per size: which file it would use, how well it fits, the measured speed, and whether it's already downloaded. It also gives a recommendation. Sizes that are already downloaded start in seconds, which is worth mentioning when the user chooses. "Fits comfortably" means the file is at most ~25% of the computer's memory. That rule comes from testing: on 24 GB, a 5.6 GB file ran at ~40 tokens/s, while a 9.6 GB file swapped to disk and dropped to ~3 tokens/s with a browser open.

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

### 3. Install llama.cpp
```
python3 scripts/k2local.py install
```
This downloads llama.cpp build b11619, the one tested with this skill, choosing the right build for the OS and GPU: Metal on Apple Silicon, CUDA for NVIDIA, Vulkan or CPU otherwise. On macOS it also removes the "downloaded from the internet" quarantine flag that would otherwise block the program. To override the choice, use `--backend cpu|cuda|vulkan`, for example if the CUDA build fails to start.

### 4. Download the chosen size
```
python3 scripts/k2local.py download --size 3.7B        # optional: --quant Q4_K_M | Q5_K_M | Q6_K | Q8_0
```
This downloads the official file from IFM's Hugging Face page, picking the best quality that fits comfortably unless you pass `--quant`. It uses 8 parallel connections, keeps the computer awake, and verifies the SHA-256 checksum. For the bigger sizes this is the longest step, so run it in the background if you can and report progress. If it stops, the same command resumes it. Single connections stalled and a sleeping Mac froze downloads in testing, which is why the script works this way.

### 5. Start it
```
python3 scripts/k2local.py serve --size 3.7B
```
This starts llama-server in the background at `http://127.0.0.1:8080` (local only) and waits until it's ready, usually a few seconds.

The server runs with `--reasoning-format none` on purpose. K2 Horizon wraps its thinking in `<ifm|think>…</ifm|think>` tags, and llama.cpp b11619 doesn't recognize them: without this flag the answer ends up hidden in the "thinking" field and looks empty. `ask`, and the code in `references/using-from-code.md`, split the text correctly.

Only one size runs at a time. To switch, run `stop` and then `serve --size <other>`.

### 6. Test it
```
python3 scripts/k2local.py ask "Split 86 equally between 3 people. How much does each get?"
```
Expect 28⅔ (about 28.67) and a speed line. Answers vary a little between runs: 7B was right every time in testing, and 3.7B was right most times (once it gave "28 each, with 2 left over"). 0.9B may get it wrong. That's a known limit of the smallest model, not a setup problem, so mention it if it happens. Keep the test question unambiguous: "Split 86 between 3" sometimes gets a whole-number split.

Show the user the answer and the speed. If they like, ask one question of their own.

### 7. Hand it over
Tell the user, in plain language:
- Which size is running, at `http://127.0.0.1:8080`, and that their data stays on their machine.
- To chat: use the exact `Ask:` command that `serve` printed. It includes the full script path, plus `--dir` or `--port` when they aren't the defaults, so the user can copy it as is. It uses `medium` thinking by default, which is fast. Add `--effort high` for hard questions (the model's own default, slower and more careful) or `--effort low` for quick ones.
- To use it from code or other apps: it speaks the OpenAI chat API at `/v1`. See `references/using-from-code.md`.
- To try another size: `download --size …`, then `stop`, then `serve --size …`.
- To stop: `stop`. It confirms the server exited and the port is free. To start again later, run `serve`; install and download only happen once.
- To remove everything: delete the `~/k2-horizon-local` folder.

## When something goes wrong
Read `references/troubleshooting.md`. It covers the problems actually seen: empty answers, slow output, stalled downloads, macOS blocking the program, the server quitting on load, and a busy port. `k2local.py status` shows what's downloaded, which size is running, and the port. `server.log` in the storage folder has the server's own messages.

## Recommended settings (from IFM's model pages)
- `top_p 0.95`. `temperature 1.0`, except **0.9B, which uses 0.6**. `ask` sets this automatically.
- `reasoning_effort`: `high` (default), `medium` or `low`, sent as `chat_template_kwargs: {"reasoning_effort": "…"}`.
- Give hard tasks plenty of output tokens. IFM suggests up to 32,768.

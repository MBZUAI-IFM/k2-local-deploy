# Troubleshooting K2 Horizon (0.9B / 3.7B / 7B) locally

Every problem here happened during real testing. Start with `python3 scripts/k2local.py status`, then look at `~/k2-horizon-local/server.log`.

## Empty answer with 3.7B at medium effort
**Cause:** the model, not the setup. llama.cpp recognizes K2 Horizon's thinking tags and splits replies into `reasoning_content` (thinking), `content` (answer) and `tool_calls`. But at medium effort, 3.7B opens its thinking with `<ifm|think_fast>` and closes it with `</ifm|think>` instead of `</ifm|think_fast>`. llama.cpp keeps waiting for the matching tag, so the whole reply, answer and any tool call included, ends up in `reasoning_content`, and `content` is empty. Seen in every 3.7B medium run in testing (Q4_K_M). 0.9B closes its tags correctly. Reported to IFM.
**Fix:** `ask` already handles it. In your own code, if `reasoning_content` contains a `</ifm|think…>` tag, split it yourself with `read_message()` from `using-from-code.md`. Or use `high` or `low` effort with 3.7B.

## Very slow (a few tokens per second on a Mac or GPU machine)
**Cause:** memory pressure. The model plus other apps don't fit, so the OS swaps the model to disk. Seen with the 9.6 GB Q8 file on a 24 GB Mac with Chrome open: ~3 tokens/s instead of ~40.
**Fix:** close heavy apps, or use a smaller file or size: `stop`, then `download --size 7B --quant Q4_K_M` (or `--size 3.7B`), then `serve --size …`. `check` shows what fits comfortably. On macOS, check swap with `sysctl vm.swapusage`.
On a computer with no GPU, a few tokens per second is normal.

## Download stalls or crawls
**Cause:** the computer went to sleep (the test Mac slept after 1 idle minute, which froze the download), or a single connection stalled.
**Fix:** rerun `download`. It resumes from where it stopped, uses 8 parallel connections, and keeps the computer awake (`caffeinate` on macOS, `SetThreadExecutionState` on Windows). On Linux, keep the session active or use `systemd-inhibit`.

## `install` fails with "HTTP Error 403: rate limit exceeded"
**Cause:** older versions of the script asked the GitHub API for the llama.cpp download link. GitHub allows only 60 anonymous API calls an hour per network, and on a shared office network everyone's apps count toward the same 60. Seen on an office network, while the model download from Hugging Face worked fine.
**Fix:** get the current script (rerun step 0 in `SKILL.md`). It downloads llama.cpp straight from the releases page and doesn't use the API. With an old script, wait an hour, or run `install` once on another network such as a phone hotspot.

## Problems after llama.cpp updated itself
`serve` moves to the newest llama.cpp release when one comes out. If answers look wrong or empty, or the server won't start after an update, go back to the build that worked, for example b11669: `stop`, `install --force --build b11669`, then `serve`. A build installed by number stays until you run `install --force` again. `status` shows the installed build. To start without checking for updates once, use `serve --no-update`.

## "Checksum mismatch"
The file was damaged in transit. The script renamed it to `.bad`. Delete that file and run `download` again.

## Server stops while loading, or `serve` times out
- **Incomplete model file.** This happened when a download broke at 6.1 of 9.6 GB. Run `download` again: it checks the size and checksum.
- **Out of memory.** Use `--quant Q4_K_M` or a smaller size, or lower the context size: `serve --ctx 8192`.
- **Wrong GPU build.** CUDA or Vulkan builds fail without matching drivers. Run `install --backend cpu --force` to fall back to the CPU build.
- **The first load was slow.** The first start after a fresh download once took longer than 5 minutes. A retry loaded in 7 seconds. `serve` waits up to 15 minutes.

## macOS: "llama-server cannot be opened because the developer cannot be verified"
`install` normally removes the quarantine flag. If the message still appears, run `xattr -dr com.apple.quarantine ~/k2-horizon-local/llama`.

## "Address already in use"
Port 8080 is taken. `serve` now checks this and says so. Run `stop` if it's an earlier K2 server, or use another port everywhere: `serve --port 8081` and `ask "…" --port 8081`.

## Windows notes
- Use `py` instead of `python3` if `python3` isn't found.
- With an NVIDIA GPU, `install` downloads both the CUDA build and the matching CUDA runtime zip into the same folder. llama.cpp needs both.
- The Windows and Linux paths of this skill weren't tested on real hardware. Only macOS on Apple Silicon was. If the auto-picked build fails, retry with `install --backend cpu --force`.

## "X isn't downloaded yet" or "Y is running. Run stop first"
Each size and format is downloaded separately (`download --size … --format …`), and only one runs at a time. `status` lists what's downloaded and what's running. In the state, `7B` is compressed GGUF and `7B-bf16` is BF16 GGUF.

## 0.9B gets simple reasoning wrong
That's the model, not the setup. In testing it answered "28, 28, 30" to an equal split, and it picked an outdated document over the current one. Use 3.7B or 7B when correctness matters. 0.9B is meant for quick commands on tiny devices.

# Troubleshooting K2 Horizon (0.9B / 3.7B / 7B) on llama.cpp

Every problem here happened during real testing. Start with `python3 scripts/k2local.py status`, then look at `~/k2-horizon-local/server.log`.

## Answers are empty, or full of `<ifm|…>` tags
**Cause:** K2 Horizon wraps its reasoning in `<ifm|think>…</ifm|think>`. At other effort levels the tag names vary, such as `<ifm|think_fast>` and `</ifm|think_faster>`. Tool calls come as `<ifm|tool_calls><ifm|tool_call>name<ifm|arg_key>k</ifm|arg_key><ifm|arg_value>v</ifm|arg_value></ifm|tool_call></ifm|tool_calls>`. llama.cpp b11619 doesn't recognize these. With its default reasoning parser, it puts the entire reply into `reasoning_content` and `content` comes back empty.
**Fix:** start the server with `--reasoning-format none` (`serve` already does) and split the text yourself. Use `split_output()` in `k2local.py`, or the copy in `using-from-code.md`. Everything after the first `</ifm|think…>` tag is the answer.

## Very slow (a few tokens per second on a Mac or GPU machine)
**Cause:** memory pressure. The model plus other apps don't fit, so the OS swaps the model to disk. Seen with the 9.6 GB Q8 file on a 24 GB Mac with Chrome open: ~3 tokens/s instead of ~40.
**Fix:** close heavy apps, or use a smaller file or size: `stop`, then `download --size 7B --quant Q4_K_M` (or `--size 3.7B`), then `serve --size …`. `check` shows what fits comfortably. On macOS, check swap with `sysctl vm.swapusage`.
On a computer with no GPU, a few tokens per second is normal.

## Download stalls or crawls
**Cause:** the computer went to sleep (the test Mac slept after 1 idle minute, which froze the download), or a single connection stalled.
**Fix:** rerun `download`. It resumes from where it stopped, uses 8 parallel connections, and keeps the computer awake (`caffeinate` on macOS, `SetThreadExecutionState` on Windows). On Linux, keep the session active or use `systemd-inhibit`.

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
Each size is downloaded separately (`download --size …`), and only one runs at a time. `status` lists what's downloaded and what's running.

## 0.9B gets simple reasoning wrong
That's the model, not the setup. In testing it answered "28, 28, 30" to an equal split, and it picked an outdated document over the current one. Use 3.7B or 7B when correctness matters. 0.9B is meant for quick commands on tiny devices.

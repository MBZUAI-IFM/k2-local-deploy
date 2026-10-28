# Run K2 Horizon locally

![Run K2 Horizon on your own computer: 0.9B, 3.7B and 7B](docs/images/banner.svg)

Run [K2 Horizon](https://huggingface.co/IFM), the open models from MBZUAI's Institute of Foundation Models, privately on your own computer. One script checks your computer, picks the right model file, then installs, downloads and starts it. After the download, nothing leaves your machine.

Works on macOS, Linux and Windows. It only needs Python 3.8+.

## Quick start

**With an AI agent:** give it [`SKILL.md`](k2-horizon-local/SKILL.md) and ask it to *"set up K2 Horizon on my computer."* It does every step and lets you choose the size.

**By hand:** download this repository (**Code → Download ZIP**), open a terminal in its folder, and run:

```bash
python3 k2-horizon-local/scripts/k2local.py check                  # what fits your computer
python3 k2-horizon-local/scripts/k2local.py install                # get llama.cpp
python3 k2-horizon-local/scripts/k2local.py download --size 3.7B   # 0.9B, 3.7B or 7B
python3 k2-horizon-local/scripts/k2local.py serve --size 3.7B      # start it
python3 k2-horizon-local/scripts/k2local.py chat                   # chat in the terminal
```

On Windows, use `py` instead of `python3`.

While it's running, you can also chat in your browser at **http://127.0.0.1:8080**. Stop it with `stop`. Next time, just run `serve`.

## Which size?

| Size | Best for | Download | Speed* |
|---|---|---|---|
| **0.9B** | Quick commands; makes reasoning mistakes | 0.7–1.2 GB | ~165 tokens/s |
| **3.7B** | Everyday use: summaries, documents, code fixes | 3.2–5.4 GB | ~46 tokens/s |
| **7B** | Best results overall | 5.6–9.6 GB | ~40 tokens/s |

\*On an Apple M4 Pro Mac with 24 GB of memory. Computers without a GPU are several times slower.

As a rule of thumb: 8 GB of memory → 0.9B, 16 GB → 3.7B, 24 GB or more → 7B. `check` tells you for your computer.

These are compressed files. For full quality, add `--format gguf-bf16` to `download` and `serve`. Those files are about twice the size and need much more memory.

## Commands

| Command | What it does |
|---|---|
| `check` | Shows what fits your computer and recommends a size |
| `install` | Downloads the latest llama.cpp for your system |
| `download --size …` | Downloads and verifies a model |
| `serve --size …` | Updates llama.cpp if a newer release is out, then starts the model at `http://127.0.0.1:8080` |
| `chat` | Chats in the terminal (`--window` opens a new window) |
| `ask "…"` | Asks one question |
| `status` / `stop` | Shows what's running / stops it |

Everything is stored in `~/k2-horizon-local`. To remove it all, delete that folder.

## Use it from code

The model speaks the OpenAI API at `http://127.0.0.1:8080/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
reply = client.chat.completions.create(model="k2-horizon", messages=[{"role": "user", "content": "Hello!"}])
print(reply.choices[0].message.content)
```

More in [`using-from-code.md`](k2-horizon-local/references/using-from-code.md), including tool calling.

## Something went wrong?

See [`troubleshooting.md`](k2-horizon-local/references/troubleshooting.md).

Tested on Apple Silicon Macs. Linux and Windows are supported but not yet tested on real hardware.

## Community and credits

Questions or feedback? Join the IFM Discord: [ifm.ai/discord](https://ifm.ai/discord)

- **K2 Horizon** by the [Institute of Foundation Models (IFM), MBZUAI](https://huggingface.co/IFM), Apache 2.0.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** runs the models, MIT license.

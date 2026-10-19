# Using a local K2 Horizon model from code

The server speaks the OpenAI chat-completions API at `http://127.0.0.1:8080/v1`. No API key is needed, and it only listens on the local machine. This is the same for all three formats: the GGUF formats run in llama-server, and the original format runs in `scripts/original_server.py`, which offers the same API (normal and streaming replies, tools, and the same raw `<ifm|think>` text).

## Settings (from IFM's model page)
- `temperature: 1.0` (use **0.6 for 0.9B**), `top_p: 0.95`
- `chat_template_kwargs: {"reasoning_effort": "low" | "medium" | "high"}`. `high` is the default and the most careful.
- Give hard tasks plenty of `max_tokens`. The thinking counts toward the limit.

## Python (standard library)
```python
import json, re, urllib.request

def split_output(raw):
    """(thinking, answer) from K2 Horizon's raw text. Tag names vary by effort: <ifm|think>, <ifm|think_fast>, …"""
    parts = re.split(r"</ifm\|think\w*>", raw, maxsplit=1)
    thinking, answer = (parts[0], parts[1]) if len(parts) == 2 else ("", raw)
    thinking = re.sub(r"<ifm\|think\w*>", "", thinking)
    answer = re.sub(r"<ifm\|tool_calls>.*?(</ifm\|tool_calls>|$)", "", answer, flags=re.S)
    return thinking.strip(), re.sub(r"</?ifm\|think\w*>", "", answer).strip()

def ask(prompt, effort="medium"):   # the model's own default is "high"; medium is faster for everyday use
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 1.0, "top_p": 0.95,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(req))["choices"][0]["message"]
    return split_output((m.get("reasoning_content") or "") + (m.get("content") or ""))[1]

print(ask("Summarize in one line: dinner moved to 8 PM, stand-up cancelled."))
```

## OpenAI SDK
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="k2-horizon",  # any name works with llama-server
    messages=[{"role": "user", "content": "Hello"}],
    temperature=1.0, top_p=0.95,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)
raw = (getattr(r.choices[0].message, "reasoning_content", None) or "") + (r.choices[0].message.content or "")
# then use split_output(raw) from above
```

## Tool calling
Pass OpenAI-style `tools` in the request. Because the server runs with `--reasoning-format none`, tool calls arrive inside the text as:
```
<ifm|tool_calls>
<ifm|tool_call>send_message
<ifm|arg_key>to</ifm|arg_key>
<ifm|arg_value>Sara</ifm|arg_value>
</ifm|tool_call>
</ifm|tool_calls>
```
`split_output()` in `scripts/k2local.py` also returns these as `[{"name": ..., "arguments": {...}}]`. To send the tool result back, add an assistant message with `tool_calls`, then a `{"role": "tool", "tool_call_id": ..., "content": ...}` message. In testing, all three sizes called simple tools correctly. 3.7B and 7B also used the results well.

## Original format straight from Transformers
If you downloaded the original format and want the model inside your own Python code (for research or fine-tuning), load the folder with the private environment's Python (`~/k2-horizon-local/python/bin/python`):
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "~/k2-horizon-local/K2-Horizon-0.9B"   # expand ~ first, e.g. with os.path.expanduser
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).to("mps")  # or "cuda" / "cpu"
enc = tok.apply_chat_template([{"role": "user", "content": "Hello"}], add_generation_prompt=True,
                              return_dict=True, return_tensors="pt", reasoning_effort="low").to(model.device)
stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<|ifm|im_end|>")]   # IFM's config only lists the first
out = model.generate(**enc, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.95, eos_token_id=stop)
raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False).replace("<|ifm|im_end|>", "")
print(split_output(raw)[1])   # split_output() from above
```

## Alternative: vLLM or SGLang on a GPU server
IFM's official commands use `--reasoning-parser k2_horizon --tool-call-parser k2_horizon`, which do this splitting on the server. For example:
```
vllm serve IFM/K2-Horizon-7B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 \
  --reasoning-parser k2_horizon --enable-auto-tool-choice --tool-call-parser k2_horizon
```

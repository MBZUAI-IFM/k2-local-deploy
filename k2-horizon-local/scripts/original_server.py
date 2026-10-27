#!/usr/bin/env python3
"""Serve K2 Horizon's original (safetensors) weights with Transformers, using the same API as llama-server.

k2local.py starts this with the private Python environment that `install --format original` creates.
It answers at /health, /v1/models and /v1/chat/completions (normal and streaming), so `ask`, the OpenAI SDK
and the examples in references/using-from-code.md work the same as with the GGUF formats.

Like llama-server, replies come back split: the reasoning in reasoning_content, the answer in content and
tool calls in tool_calls. The split uses split_output() from k2local.py, which accepts any closing think tag,
so 3.7B's mismatched medium-effort tag (<ifm|think_fast> … </ifm|think>) doesn't hide the answer here.
"""
import argparse, json, os, re, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k2local import THINK_END, split_output   # standard library only, safe to import

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="folder with the downloaded original files")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--name", default="k2-horizon")
ARGS = ap.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading {ARGS.model} on {DEVICE} …", flush=True)
TOK = AutoTokenizer.from_pretrained(ARGS.model, trust_remote_code=True)   # IFM ships its own model code
MODEL = AutoModelForCausalLM.from_pretrained(ARGS.model, dtype=torch.bfloat16, trust_remote_code=True,
                                             low_cpu_mem_usage=True).to(DEVICE).eval()
LOCK = threading.Lock()   # one generation at a time; /health still answers while it runs
# Stop at the end of the assistant's turn. IFM's generation_config only lists <|endoftext|>, but the model ends
# each turn with <|ifm|im_end|>; without it, generation runs on and the tag leaks into the answer.
_eos = MODEL.generation_config.eos_token_id
STOP = [i for i in (_eos if isinstance(_eos, list) else [_eos]) if i is not None]
_im_end = TOK.convert_tokens_to_ids("<|ifm|im_end|>")
if isinstance(_im_end, int) and _im_end != TOK.unk_token_id and _im_end not in STOP:
    STOP.append(_im_end)
# Tokens to drop from the text. The <ifm|think> and tool-call tags stay until the reply is split.
DROP = {t for t in [TOK.eos_token, TOK.pad_token, TOK.bos_token, *TOK.convert_ids_to_tokens(STOP)] if t}
print(f"Ready on http://127.0.0.1:{ARGS.port}", flush=True)


OPEN = re.compile(r"<ifm\|think\w*>")


def clean(text):
    for t in DROP:
        text = text.replace(t, "")
    return text


def tool_calls(calls):
    """OpenAI-style tool_calls from split_output()'s [{"name", "arguments"}]. Arrays and objects arrive as JSON."""
    out = []
    for c in calls:
        args = {}
        for k, v in c["arguments"].items():
            try:
                args[k] = json.loads(v) if v[:1] in "[{" else v
            except ValueError:
                args[k] = v
        out.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(args, ensure_ascii=False)}})
    return out


def split_partial(text):
    """(reasoning, answer) of a reply that's still being generated. Holds back a tag that's only half written,
    and any tool-call block, which is sent whole at the end."""
    i = text.rfind("<")
    tail = text[i:] if i != -1 else ""
    if tail and ">" not in tail and any(t.startswith(tail[:len(t)]) for t in ("<ifm|", "</ifm|")):
        text = text[:i]
    m = THINK_END.search(text)
    if not m:
        return OPEN.sub("", text).lstrip(), ""
    answer = text[m.end():].lstrip()
    t = answer.find("<ifm|tool_calls>")
    return OPEN.sub("", text[:m.start()]).lstrip(), answer if t == -1 else answer[:t]


def prepare(body):
    kw = dict(body.get("chat_template_kwargs") or {})
    if body.get("tools"):
        kw["tools"] = body["tools"]
    enc = TOK.apply_chat_template(body["messages"], add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt", **kw)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    temp = float(body.get("temperature", 1.0))
    gen = {"max_new_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096),
           "do_sample": temp > 0, "eos_token_id": STOP, "pad_token_id": TOK.pad_token_id or TOK.eos_token_id}
    if temp > 0:
        gen.update(temperature=temp, top_p=float(body.get("top_p", 0.95)))
    return enc, gen


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)

    def send_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self.send_json(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self.send_json(200, {"object": "list", "data": [{"id": ARGS.name, "object": "model", "owned_by": "IFM"}]})
        self.send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self.send_json(404, {"error": {"message": "not found"}})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            enc, gen = prepare(body)
        except Exception as e:
            return self.send_json(400, {"error": {"message": f"bad request: {e}"}})
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        n_in = enc["input_ids"].shape[1]
        with LOCK:
            if body.get("stream"):
                return self.stream(enc, gen, rid, created)
            t0 = time.time()
            with torch.inference_mode():
                out = MODEL.generate(**enc, **gen)
            dt = time.time() - t0
        new = out[0][n_in:]
        thinking, answer, calls = split_output(clean(TOK.decode(new, skip_special_tokens=False)))
        n_out = len(new)
        message = {"role": "assistant", "content": answer, "reasoning_content": thinking}
        if calls:
            message["tool_calls"] = tool_calls(calls)
        finish = "tool_calls" if calls else "length" if n_out >= gen["max_new_tokens"] else "stop"
        self.send_json(200, {
            "id": rid, "object": "chat.completion", "created": created, "model": ARGS.name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
            "timings": {"predicted_n": n_out, "predicted_ms": dt * 1000, "predicted_per_second": n_out / dt if dt else None},
        })

    def stream(self, enc, gen, rid, created):
        streamer = TextIteratorStreamer(TOK, skip_prompt=True, skip_special_tokens=False)
        worker = threading.Thread(target=lambda: MODEL.generate(**enc, **gen, streamer=streamer), daemon=True)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def chunk(delta, finish=None):
            obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": ARGS.name,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        worker.start()
        chunk({"role": "assistant", "content": ""})
        text, sent_r, sent_a = "", "", ""
        for piece in streamer:
            text += clean(piece)
            r, a = split_partial(text)
            if len(r) > len(sent_r):
                chunk({"reasoning_content": r[len(sent_r):]}); sent_r = r
            if len(a) > len(sent_a):
                chunk({"content": a[len(sent_a):]}); sent_a = a
        worker.join()
        thinking, answer, calls = split_output(text)   # flush whatever was held back
        if thinking.startswith(sent_r) and len(thinking) > len(sent_r):
            chunk({"reasoning_content": thinking[len(sent_r):]})
        if answer.startswith(sent_a) and len(answer) > len(sent_a):
            chunk({"content": answer[len(sent_a):]})
        if calls:
            chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(tool_calls(calls))]})
        chunk({}, "tool_calls" if calls else "stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()

"""Local novelty mouth for the existing Voynich lab checkpoint; no training.

Uses the lab's own architecture/tokenizer on CPU. OpenAI wire compatibility
does not confer conversation, instruction-following, or decoding ability.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "voynich-experimental"
DEFAULT_LAB = Path(__file__).resolve().parent
CHECKPOINT = "weights/model.pt"
MAX_OUTPUT = 256


def request_options(body):
    if not isinstance(body, dict):
        raise ValueError("expected a JSON object")
    if body.get("model") != MODEL_ID:
        raise ValueError(f"model must be {MODEL_ID}")
    if body.get("tools") or body.get("response_format") or body.get("n", 1) != 1:
        raise ValueError("only single plain-text completions are supported")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    if any(not isinstance(m, dict) or not isinstance(m.get("content"), str)
           for m in messages):
        raise ValueError("messages must contain plain text only")
    # This is a text continuation model. Roles supply no learned chat template.
    prompt = "\n".join(m["content"] for m in messages)
    limit = body.get("max_completion_tokens", body.get("max_tokens", 128))
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_tokens must be a positive integer")
    temperature = float(body.get("temperature", 0.8))
    top_p = float(body.get("top_p", 1.0))
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    stops = body.get("stop") or []
    if isinstance(stops, str):
        stops = [stops]
    if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
        raise ValueError("stop must be nonempty strings")
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)
                             or not 0 <= seed < 2**63):
        raise ValueError("seed must be an integer in [0, 2**63)")
    return prompt, min(limit, MAX_OUTPUT), temperature, top_p, stops, seed


class Voynich:
    def __init__(self, lab):
        lab = Path(lab).resolve()
        sys.path.insert(0, str(lab))
        sys.path.insert(0, str(lab / ".runtime"))
        import torch
        from src.train import LanguageModel
        from src.tokenize import tokenize_line
        self.torch = torch
        self.tokenize = tokenize_line
        torch.set_num_threads(2)
        path = lab / CHECKPOINT
        # Only local tensor/primitive data; never execute checkpoint pickle code.
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.spec = saved["spec"]
        self.vocab = self.spec["vocabulary"]
        self.lookup = {t: i for i, t in enumerate(self.vocab)}
        self.width = self.spec["training"]["context_length"]
        self.clusters = json.loads((lab / "configs/phase3.json").read_text())["clusters"]
        self.model = LanguageModel(len(self.vocab), self.spec["architecture"],
                                   self.spec["with_position"], self.spec["training"])
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.lock = threading.Lock()
        self.info = dict(model=MODEL_ID, device="cpu", checkpoint=str(path),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         parameters=self.model.count, context_tokens=self.width,
                         max_output_tokens=MAX_OUTPUT,
                         note="EVA pattern continuation; no translation or chat training")

    def complete(self, options):
        prompt, limit, temperature, top_p, stops, seed = options
        torch = self.torch
        tokens, events = [], []
        for li, line in enumerate(prompt.splitlines()):
            pieces = self.tokenize(line, self.spec["tokenization"], self.clusters,
                                   self.spec["bpe_merges"])
            if li and pieces:
                tokens.append(" ")
                events.append(0)
            tokens.extend(pieces)
            events.extend(([5 if li == 0 else 3] + [0] * (len(pieces) - 1)) if pieces else [])
        ids = [2] + [self.lookup.get(t, 1) for t in tokens]
        flags = events + [0 if tokens else 5]
        # Retain the actual suffix, including its known pre-target line events.
        ids, flags = ids[-self.width:], flags[-self.width:]
        prompt_count = len(ids) - (1 if ids[0] == 2 else 0)
        output, generated, reason = "", 0, "length"
        with self.lock, torch.inference_mode():
            rng = torch.Generator(device="cpu")
            rng.manual_seed(seed) if seed is not None else rng.seed()
            for _ in range(limit):
                logits = self.model(torch.tensor([ids]), torch.tensor([flags]))[0, -1].float()
                logits[:3] = -float("inf")  # PAD/UNK/BOS are not visible EVA.
                if temperature == 0:
                    nxt = int(logits.argmax())
                else:
                    probs = (logits / temperature).softmax(-1)
                    if top_p < 1:
                        ordered, indices = probs.sort(descending=True)
                        ordered[(ordered.cumsum(0) - ordered) >= top_p] = 0
                        probs = torch.zeros_like(probs).scatter(0, indices, ordered)
                        probs /= probs.sum()
                    nxt = int(torch.multinomial(probs, 1, generator=rng))
                generated += 1
                output += self.vocab[nxt]
                cuts = [output.index(s) for s in stops if s in output]
                if cuts:
                    output, reason = output[:min(cuts)], "stop"
                    break
                ids = (ids + [nxt])[-self.width:]
                flags = (flags + [0])[-self.width:]
        usage = dict(prompt_tokens=prompt_count, completion_tokens=generated,
                     total_tokens=prompt_count + generated)
        return output, reason, usage


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        # Never log prompt or resident text.
        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, engine.info)
            elif self.path == "/v1/models":
                self.reply(200, dict(object="list", data=[dict(id=MODEL_ID,
                           object="model", owned_by="local-voynich-lab")]))
            else:
                self.reply(404, dict(error="unknown route"))

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.reply(404, dict(error="unknown route"))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_048_576:
                    raise ValueError("request body must be 1 byte to 1 MiB")
                body = json.loads(self.rfile.read(length))
                options = request_options(body)
            except (ValueError, TypeError) as exc:
                return self.reply(400, dict(error=dict(message=str(exc))))
            try:
                text, reason, usage = engine.complete(options)
            except Exception:
                return self.reply(500, dict(error=dict(message="Voynich inference failed")))
            common = dict(id="voynich-" + str(time.time_ns()), created=int(time.time()), model=MODEL_ID)
            if not body.get("stream"):
                return self.reply(200, dict(**common, object="chat.completion", usage=usage,
                    choices=[dict(index=0, message=dict(role="assistant", content=text), finish_reason=reason)]))
            # Buffered SSE: finish stop matching before emitting visible text.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            def emit(delta, finish=None, counts=None):
                payload = dict(**common, object="chat.completion.chunk",
                    choices=[dict(index=0, delta=delta, finish_reason=finish)])
                if counts is not None:
                    payload["usage"] = counts
                self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
            try:
                emit(dict(role="assistant"))
                for start in range(0, len(text), 16):
                    emit(dict(content=text[start:start+16]))
                emit({}, reason, usage)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", default=os.environ.get("JNAIQ_VOYNICH_LAB", str(DEFAULT_LAB)))
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    engine = Voynich(args.lab)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(engine))
    print(f"{MODEL_ID}: CPU, http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

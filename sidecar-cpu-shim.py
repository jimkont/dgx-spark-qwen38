#!/usr/bin/env python3
"""CPU arm of the System One sidecar: POST /v1/systemone, answered by llama-server.

Reproduces the SGLang /v1/systemone protocol (PROMPT_FORMAT_VERSION 1, read from
the image's serving_systemone + serving_decisions in the GPU arm):

  state -> text: None -> "", str -> str, else json.dumps(ensure_ascii=False, separators)
  per question:  content = "\n".join([text, "", *lines]) with the server wording
  prompt:        the model's own chat template over [{"role":"user","content":content}],
                 add_generation_prompt=True, enable_thinking=False (thinking forced off)
  labels:        encode(prompt + label) must be exactly one new token, per question kind
  scoring:       raw full-softmax mass of each label token at the answer position,
                 read from llama-server's top-500 logprobs of the first sampled token
                 (one generated token, prefill only)

The answer shapes, confidence formulas, and usage field match the engine's so
bench-systemone.py can score this target exactly as it scored the GPU lanes.
Run: ./sidecar-cpu-shim.py [--port 8098] [--engine http://127.0.0.1:8099]
     [--template /tmp/qwen35-chat-template.jinja]
"""

import argparse
import json
import math
import string
import threading
import time
import urllib.error
import urllib.request

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jinja2

SERVER = "http://127.0.0.1:8099"
N_PROBS = 500
MODEL_NAME = "qwen3.5-2b-cpu"

_lock = threading.Lock()           # one prefill at a time; the CPU is the point
_label_ids = {}                    # kind -> (label, id) list, verified per rendered prompt


_JINJA_ENV = jinja2.Environment(
    undefined=jinja2.ChainableUndefined,
    keep_trailing_newline=True,
    autoescape=False,
)


def engine(path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        SERVER + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
def render_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tokenize(s: str) -> list:
    return engine("/tokenize", {"content": s})["tokens"]


def _labels_for(kind: str, names: list) -> list:
    if kind == "choice":
        return list(string.ascii_uppercase[: len(names)])
    if kind == "yes_no":
        return ["yes", "no"]
    return [str(i) for i in range(len(names))]


def _qtext(question) -> str:
    """Rendered question instructions, empty when blank (matches is_blank_decision_text)."""
    q = render_text(question)
    return "" if not q.strip() else q


def _lines(text: str, kind: str, question, names, details, labels) -> list:
    qtext = _qtext(question)
    if kind == "choice":
        lines = [f"Question: {qtext}"] if qtext else []
        for label, name, detail in zip(labels, names, details):
            d = render_text(detail)
            lines.append(f"{label}: {name} - {d}" if d else f"{label}: {name}")
        lines.append("Answer with the letter of one option only.")
    elif kind == "score":
        lines = [f"Question: {qtext}"] if qtext else []
        lines += [f"{label}: {render_text(level)}"
                  for label, level in zip(labels, details)]
        lines.append("Answer with the number of one level only.")
    else:
        lines = [f"Is the following true? {qtext}" if qtext else "Is the following true?"]
        for label, detail in zip(labels, details):
            d = render_text(detail)
            if d:
                lines.append(f"{label}: {d}")
        lines.append("Answer with yes or no only.")
    return lines


def _view(question: dict):
    qtype = question["type"]
    if qtype == "choice":
        criteria = question["criteria"]
        return "choice", question.get("instructions"), list(criteria), list(criteria.values())
    if qtype == "noul":
        criteria = question.get("criteria") or {}
        return "yes_no", question.get("instructions"), ["yes", "no"], [
            criteria.get("true"), criteria.get("false")]
    criteria = question.get("criteria") or []
    return "score", question.get("instructions"), [str(i) for i in range(len(criteria))], list(criteria)


def _check_labels(prompt: str, prompt_ids: list, labels: list, kind: str):
    """Each label must add exactly one distinct token after the full rendered prompt.
    Cached per (kind, count): labels of a kind are always the A to Z or 0 to n prefix,
    so a longer check subsumes every shorter one."""
    ids = []
    for label in labels:
        got = tokenize(prompt + label)
        if len(got) != len(prompt_ids) + 1 or got[:-1] != prompt_ids:
            raise ValueError(f"label {label!r} is not one distinct token after the "
                             f"chat prompt ({kind}); tokenization mismatch")
        if got[-1] in ids:
            raise ValueError(f"label {label!r} collides with an earlier label token")
        ids.append(got[-1])
    _label_ids[(kind, len(ids))] = ids
    return ids


def _label_ids_for(kind: str, count: int, prompt: str, labels: list) -> list:
    """Cached label ids for a kind prefix; verified on the prompt only on a miss.
    The answer-position context is the fixed closing line plus the generation
    suffix, so ids are stable across prompts of one kind."""
    cached = [v for (k, n), v in _label_ids.items() if k == kind and n >= count]
    if cached:
        return max(cached, key=len)[:count]
    return _check_labels(prompt, tokenize(prompt), labels, kind)


_TEMPLATE = None


def render_prompt(template_src: str, content: str) -> str:
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _JINJA_ENV.from_string(template_src)
    return _TEMPLATE.render(
        messages=[{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _normalized(p: list) -> list:
    total = math.fsum(p)
    return [x / total for x in p]


def _choice_confidence(q: list) -> float:
    n = len(q)
    return min(1.0, max(0.0, (n * max(q) - 1) / (n - 1)))


def _score_confidence(q: list) -> float:
    n = len(q)
    top = max(q)
    spread = math.fsum(x for x in q if x != top)
    uniform_spread = 1.0 - 1.0 / n
    return max(0.0, 1 - spread / uniform_spread)


def answer(view, probabilities, mass, qid):
    kind, _, names, details = view
    if not all(math.isfinite(v) for v in [*probabilities, mass]):
        raise RuntimeError(f"question {qid!r} scored non-finite values")
    if kind == "yes_no":
        return {"type": "noul", "noul": probabilities[0], "x_label_mass": mass}
    by_name = dict(zip(names, probabilities))
    if kind == "choice":
        return {"type": "choice",
                "choice": names[probabilities.index(max(probabilities))],
                "confidence": _choice_confidence(_normalized(probabilities)),
                "probabilities": by_name, "x_label_mass": mass}
    return {"type": "score",
            "score": math.fsum(i * p for i, p in enumerate(probabilities)),
            "confidence": _score_confidence(_normalized(probabilities)),
            "legend": dict(zip(names, details)),
            "probabilities": by_name, "x_label_mass": mass}


def handle_systemone(body: dict, template_src: str) -> dict:
    text = render_text(body.get("state"))
    questions = body.get("questions") or {}
    answers = {}
    prompt_tokens = 0
    t0 = time.time()
    for qid, question in questions.items():
        kind, question_text, names, details = _view(question)
        labels = _labels_for(kind, names)
        content = "\n".join([text, "", *_lines(text, kind, question_text, names, details, labels)])
        prompt = render_prompt(template_src, content)
        label_ids = _label_ids_for(kind, len(labels), prompt, labels)
        resp = engine("/completions", {
            "prompt": prompt, "max_tokens": 1, "logprobs": True,
            "n_probs": N_PROBS, "temperature": 0,
        }, timeout=600.0)
        prompt_tokens += resp["tokens_evaluated"]
        top = resp["completion_probabilities"][-1]["top_logprobs"]
        mass_by_id = {e["id"]: e["logprob"] for e in top}
        probabilities = [math.exp(mass_by_id.get(i, float("-inf"))) for i in label_ids]
        mass = math.fsum(probabilities)
        answers[qid] = answer((kind, question_text, names, details), probabilities, mass, qid)
    return {"model": MODEL_NAME, "answers": answers,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
            "x": {"wall_s": round(time.time() - t0, 3), "n": len(questions)}}


class Handler(BaseHTTPRequestHandler):
    template_src = ""

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, payload: dict):
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if self.path != "/v1/systemone":
            self._send(404, {"error": "unknown route"})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        with _lock:
            try:
                out = handle_systemone(body, self.template_src)
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
                return
        self._send(200, out)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "unknown route"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--engine", default="http://127.0.0.1:8099")
    ap.add_argument("--template", default="/tmp/qwen35-chat-template.jinja")
    args = ap.parse_args()
    global SERVER
    SERVER = args.engine.rstrip("/")
    Handler.template_src = open(args.template).read()
    # Verify the label contract on all three kinds before serving anything.
    probes = {
        "choice": {"type": "choice", "instructions": "Pick a color.",
                   "criteria": {"red": "warm", "blue": "cold"}},
        "noul": {"type": "noul", "instructions": "Is the sky blue?",
                 "criteria": {"true": "clear day", "false": "cloudy"}},
        "score": {"type": "score", "instructions": "Rate it.",
                  "criteria": ["low", "mid", "high"]},
    }
    handle_systemone({"state": "probe", "questions": probes}, Handler.template_src)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"shim on 127.0.0.1:{args.port} -> engine {SERVER} "
          f"(labels verified: {', '.join(sorted({k for (k, n) in _label_ids}))})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

"""Optional live L1 interpreter backed by a local model served by Ollama.

It sends the Claude interpreter's system prompt and output schema (`l1_claude`), and its
replies pass through the same `parse_output` contract checks, so a second model family
is measured behind the same L1 contract. The served model tag and the sampling options
(temperature, seed) identify the interpreter: they are nu_I. Local calls cost nothing
and need no credentials, only a running Ollama server.

Not exercised by the test suite, which uses recorded replies.
"""

from __future__ import annotations

import json
import platform
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit

from dsoi.layers.l1_claude import SYSTEM_PROMPT, interpret_reply, output_schema, user_prompt
from dsoi.layers.l1_interpreter import InterpreterSchema, InterpretResult
from dsoi.types import Message

OLLAMA_URL = "http://localhost:11434/api/chat"


def model_metadata(model: str, url: str = OLLAMA_URL) -> dict:
    """Resolve a mutable model tag before a new run; never infer historical weights.

    Ollama's /api/tags exposes the digest, format, parameter count and quantization;
    /api/version exposes the serving runtime. Host details describe this client, which
    may differ from the inference host when a remote URL is used.
    """
    parts = urlsplit(url)

    def get(endpoint):
        target = urlunsplit((parts.scheme, parts.netloc, endpoint, "", ""))
        with urllib.request.urlopen(target, timeout=30) as response:
            return json.load(response)

    wanted = model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"
    matches = [m for m in get("/api/tags")["models"]
               if m.get("name") in (model, wanted) or m.get("model") in (model, wanted)]
    if len(matches) != 1 or not matches[0].get("digest"):
        raise ValueError(f"cannot resolve one immutable digest for {model!r}")
    info = matches[0]
    return {"requested_model": model, "model_digest": info["digest"],
            "model_details": info.get("details", {}),
            "ollama_version": get("/api/version")["version"],
            "client_host": {"platform": platform.platform(), "machine": platform.machine(),
                            "processor": platform.processor()}}


def local_schema(schema: InterpreterSchema, reason_chars: int = 300) -> dict:
    """The L1 output schema with the free-text `reason` bounded: small local models can
    otherwise fill the token budget with it and be cut off (seen in the pilot)."""
    out = output_schema(schema)
    out["properties"]["reason"] = {"type": "string", "maxLength": reason_chars}
    return out


def build_request(message: Message, schema: InterpreterSchema, model: str, snapshot: str,
                  temperature: float, seed: int, max_tokens: int = 1024,
                  system: str = SYSTEM_PROMPT) -> dict:
    """An /api/chat request constrained to the L1 output schema."""
    return {
        "model": model,
        "stream": False,
        "format": local_schema(schema),
        "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_prompt(message, schema, snapshot)}],
    }


def call(request: dict, url: str = OLLAMA_URL, timeout: float = 600.0) -> dict:
    """Send one request; return the fields the response store keeps."""
    t0 = time.monotonic()
    req = urllib.request.Request(url, data=json.dumps(request).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return {"text": out.get("message", {}).get("content"),
            "stop_reason": {"length": "max_tokens"}.get(out.get("done_reason"),
                                                         out.get("done_reason")),
            "served_model": out.get("model"),
            "usage": {"input_tokens": out.get("prompt_eval_count", 0),
                      "output_tokens": out.get("eval_count", 0)},
            "seconds": round(time.monotonic() - t0, 2)}


def interpret(reply: dict, nu_i: str, message: Message,
              schema: InterpreterSchema) -> InterpretResult:
    """A stored or live reply under the L1 contract; nu_i names model and options."""
    return interpret_reply(nu_i, reply.get("stop_reason"), reply.get("text"), message, schema)

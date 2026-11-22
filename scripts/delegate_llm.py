#!/usr/bin/env python3
"""Zero-dependency CLI router v2 for local Ollama and multi-provider cloud pools."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import re
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
HEAVY_MODEL_RE = re.compile(r"(?:^|[:\-_])(1[4-9]|[2-9]\d|\d{3,})b(?:$|[:\-_])", re.IGNORECASE)
OLLAMA_LOCK_PATH = Path.home() / ".llm-router-ollama.lock"
OLLAMA_LOCK_MAX_AGE_SEC = 600
RETRYABLE_HTTP_CODES = {401, 403, 408, 429, 500, 502, 503, 504}
COMMANDCODE_ALLOWED_MODEL = "deepseek/deepseek-v4.1-flash"


class OllamaBusyError(RuntimeError):
    """Raised when Ollama is locked by another process or holding a heavy model in VRAM."""


_SUMMARIZE_CASCADE: list[tuple[str, str]] = [
    ("groq", "qwen/qwen3.8-27b"),
    ("groq", "openai/gpt-oss-120b"),
    ("cloudflare", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
    ("openrouter", "openrouter/free"),
    ("openference", "DeepSeek-V4.1-Flash-1017-Free"),
    ("commandcode", COMMANDCODE_ALLOWED_MODEL),
    ("github", "openai/gpt-4.1-mini"),
    ("gemini", "gemini-3-flash-preview"),
    ("ollama", "qwen3.5:4b"),
    ("cerebras", "gpt-oss-120b"),
]

TASK_CASCADES: dict[str, list[tuple[str, str]]] = {
    "summarize": list(_SUMMARIZE_CASCADE),
    "general": list(_SUMMARIZE_CASCADE),
    "code": [
        ("commandcode", COMMANDCODE_ALLOWED_MODEL),
        ("groq", "qwen/qwen3.8-27b"),
        ("cloudflare", "@cf/qwen/qwen2.5-coder-32b-instruct"),
        ("github", "openai/gpt-4.1-mini"),
        ("openrouter", "openrouter/free"),
        ("openference", "DeepSeek-V4.1-Flash-1017-Free"),
        ("gemini", "gemini-3-flash-preview"),
        ("ollama", "qwen2.5-coder:7b"),
        ("cerebras", "qwen-3.8-27b"),
    ],
    "embed": [("ollama", "qwen3-embedding:0.6b")],
    "local-heavy": [
        ("ollama", "qwen3.6:35b-a3b"),
        ("gemini", "gemini-3-flash-preview"),
        ("openrouter", "openrouter/free"),
        ("cerebras", "gpt-oss-120b"),
    ],
    "fast-cloud": [
        ("groq", "qwen/qwen3.8-27b"),
        ("cloudflare", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
        ("openrouter", "openrouter/free"),
        ("openference", "DeepSeek-V4.1-Flash-1017-Free"),
        ("commandcode", COMMANDCODE_ALLOWED_MODEL),
        ("gemini", "gemini-3.1-flash-lite-preview"),
        ("ollama", "qwen3.5:4b"),
        ("cerebras", "gpt-oss-120b"),
    ],
    "batch-cloud": [
        ("groq", "openai/gpt-oss-120b"),
        ("cloudflare", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
        ("openrouter", "openrouter/free"),
        ("gemini", "gemini-3-flash-preview"),
        ("ollama", "qwen3.5:4b"),
        ("cerebras", "gpt-oss-120b"),
    ],
    "long-ctx": [
        ("gemini", "gemini-3-flash-preview"),
        ("nvidia", "moonshotai/kimi-k3.1-pro"),
        ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        ("ollama", "qwen3.5:4b"),
        ("cerebras", "gpt-oss-120b"),
    ],
    "opencode-go": [
        ("opencode", "big-pickle"),
        ("openrouter", "openrouter/free"),
        ("ollama", "qwen3.5:4b"),
        ("cerebras", "qwen-3.8-27b"),
    ],
    "reason": [
        ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
        ("nvidia", "mistralai/mistral-large-4-1t-a52b"),
        ("github", "deepseek/deepseek-r1"),
        ("commandcode", COMMANDCODE_ALLOWED_MODEL),
        ("openference", "GLM-5.3-Flash-Free"),
        ("venice", "venice-uncensored"),
        ("gemini", "gemini-3-flash-preview"),
        ("ollama", "qwen3.6:35b-a3b"),
        ("cerebras", "gpt-oss-120b"),
    ],
    "fallback-cloud": [
        ("openrouter", "openrouter/free"),
        ("openference", "DeepSeek-V4.1-Flash-1017-Free"),
        ("github", "openai/gpt-4.1"),
        ("cloudflare", "@cf/qwen/qwen2.5-coder-32b-instruct"),
        ("commandcode", COMMANDCODE_ALLOWED_MODEL),
        ("opencode", "big-pickle"),
        ("ollama", "qwen3.5:4b"),
        ("cerebras", "gpt-oss-120b"),
    ],
    "freebuff": [
        ("freebuff", "freebuff-tui"),
        ("ollama", "qwen3.5:4b"),
        ("cerebras", "qwen-3.8-27b"),
    ],
}

DEFAULT_PROVIDER_MODELS = {
    "ollama": "qwen3.5:4b",
    "groq": "qwen/qwen3.8-27b",
    "nvidia": "nvidia/nemotron-3-ultra-550b-a55b",
    "cerebras": "gpt-oss-120b",
    "gemini": "gemini-3-flash-preview",
    "cloudflare": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "github": "openai/gpt-4.1-mini",
    "openrouter": "openrouter/free",
    "opencode": "big-pickle",
    "openference": "DeepSeek-V4.1-Flash-1017-Free",
    "venice": "venice-uncensored",
    "commandcode": COMMANDCODE_ALLOWED_MODEL,
    "freebuff": "freebuff-tui",
}

ALL_PROVIDERS = [
    "auto",
    "ollama",
    "groq",
    "nvidia",
    "cerebras",
    "gemini",
    "cloudflare",
    "github",
    "openrouter",
    "opencode",
    "openference",
    "venice",
    "commandcode",
    "freebuff",
]

MODEL_FLEET_CATALOG: dict[str, dict] = {
    "groq": {
        "role": "Inferência ultrarrápida em nuvem (resumos, código, lotes)",
        "models": [
            {"id": "qwen/qwen3.8-27b", "use": "Código, refatoração e resumos rápidos (0.25s)"},
            {"id": "openai/gpt-oss-120b", "use": "Raciocínio rápido e lotes (0.41s)"},
            {"id": "openai/gpt-oss-20b", "use": "Classificação e extração leve (0.31s)"},
        ],
    },
    "cloudflare": {
        "role": "Workers AI na borda (resumos, código 32B, Llama 3.3 70B)",
        "models": [
            {"id": "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "use": "Resumos e texto geral 70B FP8 (0.41s)"},
            {"id": "@cf/qwen/qwen2.5-coder-32b-instruct", "use": "Código, diffs e testes 32B (0.52s)"},
            {"id": "@cf/openai/gpt-oss-120b", "use": "Raciocínio 120B (1.23s)"},
        ],
    },
    "commandcode": {
        "role": "Uso ESTRITO apenas para DeepSeek V4.1 Flash (código, resumos, raciocínio)",
        "models": [
            {"id": COMMANDCODE_ALLOWED_MODEL, "use": "Único modelo permitido neste provedor (1.61s)"},
        ],
    },
    "github": {
        "role": "GitHub Models (GPT-4.1, GPT-4o-mini, DeepSeek R1, Llama 3.3 70B)",
        "models": [
            {"id": "openai/gpt-4.1-mini", "use": "Código leve, instruções estruturadas e resumos (<1s)"},
            {"id": "openai/gpt-4.1", "use": "Raciocínio e código de alta precisão (<1s)"},
            {"id": "openai/gpt-4o-mini", "use": "Tarefas rápidas gerais (<1s)"},
            {"id": "deepseek/deepseek-r1", "use": "Raciocínio passo a passo (<1s)"},
            {"id": "meta/llama-3.3-70b-instruct", "use": "Síntese e análise geral 70B (<1s)"},
        ],
    },
    "openrouter": {
        "role": "Meta-roteador gratuito e modelos :free",
        "models": [
            {"id": "openrouter/free", "use": "Roteador automático de modelos gratuitos (2.19s)"},
            {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "use": "550B MoE gratuito (0.59s)"},
            {"id": "nvidia/nemotron-3.5-super-vl:free", "use": "120B MoE gratuito (0.75s)"},
            {"id": "baidu/ernie-5.5-flash-preview:free", "use": "Flash gratuito para texto/resumos (1.27s)"},
        ],
    },
    "openference": {
        "role": "Streaming SSE gratuito (DeepSeek V4.1 Flash / GLM-5.3 Flash)",
        "models": [
            {"id": "DeepSeek-V4.1-Flash-1017-Free", "use": "Código e resumos rápidos via SSE (2.05s)"},
            {"id": "GLM-5.3-Flash-Free", "use": "Raciocínio GLM via SSE (2.80s)"},
        ],
    },
    "nvidia": {
        "role": "Raciocínio pesado, arquitetura e segunda opinião (timeout 600s)",
        "models": [
            {"id": "nvidia/nemotron-3-ultra-550b-a55b", "use": "Flagship 550B MoE arquitetura/auditoria (1.31s)"},
            {"id": "mistralai/mistral-large-4-1t-a52b", "use": "1T MoE revisão profunda (2.73s)"},
            {"id": "poolside/laguna-xs-2.1", "use": "Engenharia de software e patches (0.86s)"},
            {"id": "nvidia/nemotron-3.5-super-vl-120b-a12b", "use": "120B MoE análise geral (0.88s)"},
            {"id": "nvidia/nemotron-3.5-lightning-30b-a3b", "use": "Análises médias rápidas (1.38s)"},
            {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "use": "Raciocínio estruturado (1.66s)"},
            {"id": "nvidia/nemotron-3.5-bolt-e4b", "use": "Formatação e triagem ultraleve (0.52s)"},
            {"id": "google/gemma-4-31b-it", "use": "Redação e documentação (29.56s)"},
            {"id": "moonshotai/kimi-k3.1-pro", "use": "Contexto longo e segunda opinião (42.98s)"},
            {"id": "stepfun-ai/step-5-preview", "use": "Raciocínio avançado (43.72s)"},
            {"id": "z-ai/glm-5.3-flash", "use": "GLM lento / cold-start (usar --timeout 600)"},
        ],
    },
    "gemini": {
        "role": "Contexto gigante (1M+ tokens) com pool de 5 chaves",
        "models": [
            {"id": "gemini-3-flash-preview", "use": "Padrão --task long-ctx (1.02s)"},
            {"id": "gemini-3.1-flash-lite-preview", "use": "Alternativa leve para contexto longo (1.00s)"},
        ],
    },
    "venice": {
        "role": "Zero Data Retention (requer créditos USD/Diem; HTTP 402 sem saldo)",
        "models": [
            {"id": "venice-uncensored", "use": "Privacidade Zero Data Retention (quando houver saldo)"},
        ],
    },
    "opencode": {
        "role": "OpenCode Zen/Go (fallback automático para openrouter se cota -free esgotar)",
        "models": [
            {"id": "big-pickle", "use": "Modelo gratuito OpenCode Zen"},
            {"id": "beam-free", "use": "Modelo gratuito OpenCode Zen"},
        ],
    },
    "ollama": {
        "role": "Fallback local offline / dados sensíveis / embeddings (8 GB VRAM, 1 por vez)",
        "models": [
            {"id": "qwen3.5:4b", "use": "Resumos offline / dados privados"},
            {"id": "qwen2.5-coder:7b", "use": "Código e testes offline"},
            {"id": "qwen3-embedding:0.6b", "use": "Padrão --task embed (1024 dims)"},
            {"id": "qwen3.6:35b-a3b", "use": "Padrão --task local-heavy e fallback de reason"},
        ],
    },
    "cerebras": {
        "role": "Último recurso absoluto pós-Ollama (ou --provider cerebras explícito)",
        "models": [
            {"id": "gpt-oss-120b", "use": "Último recurso geral (0.34s)"},
            {"id": "qwen-3.8-27b", "use": "Último recurso código/texto (0.25s)"},
        ],
    },
}


def load_env_file(env_path: Path) -> None:
    """Load key=value pairs from env_path without overriding existing os.environ."""
    if not env_path.is_file():
        return
    try:
        content = env_path.read_text(encoding="utf-8")
    except OSError:
        return

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and (
            (value[0] == '"' and value[-1] == '"')
            or (value[0] == "'" and value[-1] == "'")
        ):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def get_key_pool(prefix: str) -> list[str]:
    """Collect numbered keys (<PREFIX>_1..9) plus optional unnumbered fallback, shuffled."""
    keys: list[str] = []
    for idx in range(1, 10):
        val = os.environ.get(f"{prefix}_{idx}", "").strip()
        if val and val not in keys:
            keys.append(val)
    fallback = os.environ.get(prefix, "").strip()
    if fallback and fallback not in keys:
        keys.append(fallback)
    random.shuffle(keys)
    return keys


def _parse_sse_response(raw: str) -> dict:
    """Assemble OpenAI-compatible SSE stream chunks into a single response dict."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s.startswith("data:") or s == "data: [DONE]":
            continue
        try:
            obj = json.loads(s[5:].strip())
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or choices[0].get("message") or {}
        if delta.get("content"):
            content_parts.append(str(delta["content"]))
        if delta.get("reasoning_content"):
            reasoning_parts.append(str(delta["reasoning_content"]))
    return {
        "choices": [
            {
                "message": {
                    "content": "".join(content_parts).strip(),
                    "reasoning_content": "".join(reasoning_parts).strip(),
                }
            }
        ]
    }


def _curl_post_json(
    url: str,
    payload_bytes: bytes,
    headers: dict[str, str],
    timeout: int,
    stream: bool,
) -> dict:
    """Fallback HTTPS POST using system curl (native OS TLS store, e.g. SChannel)."""
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_bin:
        raise RuntimeError("curl não encontrado para fallback TLS do sistema.")
    cmd = [curl_bin, "-sS", "-m", str(timeout), "-w", "\n%{http_code}", "-X", "POST", url]
    if stream:
        cmd.append("-N")
    for k, v in headers.items():
        cmd.extend(["-H", f"{k}: {v}"])
    cmd.extend(["--data-binary", "@-"])
    proc = subprocess.run(
        cmd,
        input=payload_bytes.decode("utf-8"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout + 10,
    )
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f"curl falhou (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
    parts = proc.stdout.rsplit("\n", 1)
    body = parts[0] if len(parts) == 2 else proc.stdout
    code_str = parts[1].strip() if len(parts) == 2 else "0"
    status_code = int(code_str) if code_str.isdigit() else 0
    if status_code >= 400 or status_code == 0:
        raise urllib.error.HTTPError(url, status_code, body[:200], None, None)
    if stream or body.lstrip().startswith("data:"):
        return _parse_sse_response(body)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        stripped = body.strip()
        if stripped and not stripped.lstrip().startswith("<"):
            return {"choices": [{"message": {"content": stripped}}]}
        raise


def http_post_json(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: int = 600,
    stream: bool = False,
) -> dict:
    """POST JSON payload and return decoded JSON (or assembled SSE) response."""
    req_headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "User-Agent": "llm-worker-router/2.0",
    }
    if headers:
        req_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc.reason):
            return _curl_post_json(url, data, req_headers, timeout=timeout, stream=stream)
        raise

    if stream or raw.lstrip().startswith("data:"):
        return _parse_sse_response(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        stripped = raw.strip()
        if stripped and not stripped.lstrip().startswith("<"):
            return {"choices": [{"message": {"content": stripped}}]}
        raise


def is_pid_alive(pid: int) -> bool:
    """Return True if process with given PID is still running."""
    if pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return kernel32.GetLastError() == 5  # ERROR_ACCESS_DENIED => process exists
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _cleanup_stale_ollama_lock() -> dict | None:
    """Read lockfile; remove it if stale/orphaned and return active lock info or None."""
    if not OLLAMA_LOCK_PATH.exists():
        return None
    try:
        data = json.loads(OLLAMA_LOCK_PATH.read_text(encoding="utf-8-sig"))
        pid = int(data.get("pid", 0))
        ts = float(data.get("timestamp", 0.0))
    except Exception:
        # Unreadable/corrupted lock older than 5s or invalid
        try:
            age = time.time() - OLLAMA_LOCK_PATH.stat().st_mtime
            if age > 5:
                OLLAMA_LOCK_PATH.unlink(missing_ok=True)
                return None
        except OSError:
            return None
        return {"pid": 0, "timestamp": time.time(), "model": "unknown"}

    age = time.time() - ts
    if age > OLLAMA_LOCK_MAX_AGE_SEC or not is_pid_alive(pid):
        try:
            OLLAMA_LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return {"pid": pid, "timestamp": ts, "age_sec": round(age, 1), "model": data.get("model", "")}


def _query_ollama_ps(timeout: int = 5) -> tuple[bool, list[str]]:
    """Query Ollama /api/ps to list models currently loaded in VRAM."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    req = urllib.request.Request(
        f"{base_url}/api/ps",
        headers={"Accept": "application/json", "User-Agent": "llm-worker-router/2.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        models = []
        for item in data.get("models") or []:
            name = item.get("name") or item.get("model")
            if name:
                models.append(str(name))
        return True, models
    except Exception:
        return False, []


def check_ollama_status(target_model: str | None = None) -> dict:
    """Check whether Ollama is free or busy (lockfile active or heavy model in VRAM)."""
    lock_info = _cleanup_stale_ollama_lock()
    reachable, loaded_models = _query_ollama_ps()
    heavy_conflict = None
    if target_model:
        for loaded in loaded_models:
            if loaded != target_model and HEAVY_MODEL_RE.search(loaded):
                heavy_conflict = loaded
                break

    busy = bool(lock_info) or bool(heavy_conflict)
    reasons = []
    if lock_info:
        reasons.append(f"lock ativo (pid={lock_info.get('pid')}, model={lock_info.get('model')})")
    if heavy_conflict:
        reasons.append(f"modelo pesado na VRAM ({heavy_conflict})")

    return {
        "busy": busy,
        "reason": "; ".join(reasons) if reasons else "livre",
        "lock_active": bool(lock_info),
        "lock_info": lock_info,
        "ollama_reachable": reachable,
        "loaded_models": loaded_models,
        "heavy_conflict": heavy_conflict,
    }


@contextmanager
def ollama_execution_lock(target_model: str, wait: bool = False, wait_timeout: int = 180):
    """Acquire exclusive file lock for Ollama; fail fast if wait=False and busy."""
    deadline = time.monotonic() + wait_timeout
    acquired = False

    while True:
        status = check_ollama_status(target_model=None if wait else target_model)
        if not wait and status["busy"]:
            raise OllamaBusyError(status["reason"])

        if not status["lock_active"]:
            try:
                fd = os.open(str(OLLAMA_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    payload = json.dumps(
                        {"pid": os.getpid(), "timestamp": time.time(), "model": target_model}
                    )
                    os.write(fd, payload.encode("utf-8"))
                finally:
                    os.close(fd)
                acquired = True
                break
            except FileExistsError:
                if not wait:
                    raise OllamaBusyError("lock adquirido concorrentemente por outro processo")

        if not wait:
            raise OllamaBusyError(status["reason"])
        if time.monotonic() >= deadline:
            raise OllamaBusyError(f"timeout aguardando lock do Ollama ({status['reason']})")
        time.sleep(1.0)

    try:
        yield
    finally:
        if acquired:
            try:
                if OLLAMA_LOCK_PATH.exists():
                    data = json.loads(OLLAMA_LOCK_PATH.read_text(encoding="utf-8-sig"))
                    if int(data.get("pid", 0)) == os.getpid():
                        OLLAMA_LOCK_PATH.unlink(missing_ok=True)
            except Exception:
                OLLAMA_LOCK_PATH.unlink(missing_ok=True)


def call_ollama_chat(
    model: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
    timeout: int = 600,
) -> str:
    """Call local Ollama chat API."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    res = http_post_json(f"{base_url}/api/chat", payload, timeout=timeout)
    message = res.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content and message.get("thinking"):
        content = str(message.get("thinking")).strip()
    return content


def call_ollama_embed(model: str, text: str, timeout: int = 120) -> dict:
    """Call local Ollama embeddings endpoint."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    try:
        res = http_post_json(
            f"{base_url}/api/embed",
            {"model": model, "input": text},
            timeout=timeout,
        )
        embeddings = res.get("embeddings")
        if embeddings:
            return {"model": model, "dimensions": len(embeddings[0]), "embedding": embeddings[0]}
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    res = http_post_json(
        f"{base_url}/api/embeddings",
        {"model": model, "prompt": text},
        timeout=timeout,
    )
    emb = res.get("embedding") or []
    return {"model": model, "dimensions": len(emb), "embedding": emb}


def resolve_provider_spec(provider: str) -> tuple[list[str], str, dict[str, str]]:
    """Return (endpoints, key_prefix, extra_headers) for an OpenAI-compatible provider."""
    if provider == "groq":
        return (["https://api.groq.com/openai/v1/chat/completions"], "GROQ_API_KEY", {})
    if provider == "nvidia":
        return (["https://integrate.api.nvidia.com/v1/chat/completions"], "NVIDIA_API_KEY", {})
    if provider == "cerebras":
        return (["https://api.cerebras.ai/v1/chat/completions"], "CEREBRAS_API_KEY", {})
    if provider == "gemini":
        return (
            ["https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"],
            "GEMINI_API_KEY",
            {},
        )
    if provider == "cloudflare":
        acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
        if not acct:
            raise RuntimeError("CLOUDFLARE_ACCOUNT_ID ausente em ~/.llm-router.env")
        return (
            [f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions"],
            "CLOUDFLARE_API_KEY",
            {},
        )
    if provider == "github":
        return (
            [
                "https://models.github.ai/inference/chat/completions",
                "https://models.inference.ai.azure.com/chat/completions",
            ],
            "GITHUB_API_KEY",
            {},
        )
    if provider == "openrouter":
        return (
            ["https://openrouter.ai/api/v1/chat/completions"],
            "OPENROUTER_API_KEY",
            {"HTTP-Referer": "https://localhost", "X-Title": "llm-worker-router"},
        )
    if provider == "opencode":
        return (["https://opencode.ai/zen/v1/chat/completions"], "OPENCODE_API_KEY", {})
    if provider == "openference":
        return (["https://api.openference.com/v1/chat/completions"], "OPENFERENCE_API_KEY", {})
    if provider == "venice":
        return (["https://api.venice.ai/api/v1/chat/completions"], "VENICE_API_KEY", {})
    if provider == "commandcode":
        return (["https://api.commandcode.ai/provider/v1/chat/completions"], "COMMANDCODE_API_KEY", {})
    raise ValueError(f"Provedor não suportado: {provider}")


def call_openai_compatible(
    provider: str,
    model: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
    timeout: int = 600,
) -> str:
    """Call an OpenAI-compatible provider rotating across shuffled keys and fallback endpoints."""
    endpoints, key_prefix, extra_headers = resolve_provider_spec(provider)
    keys = get_key_pool(key_prefix)
    if not keys:
        raise RuntimeError(f"Nenhuma chave encontrada para {key_prefix}_1..9 em ~/.llm-router.env")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    use_stream = provider == "openference"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_stream:
        payload["stream"] = True

    errors: list[str] = []
    for endpoint in endpoints:
        for attempt, api_key in enumerate(keys, start=1):
            headers = {"Authorization": f"Bearer {api_key}", **extra_headers}
            try:
                res = http_post_json(endpoint, payload, headers=headers, timeout=timeout, stream=use_stream)
                choices = res.get("choices") or []
                if not choices:
                    raise RuntimeError("Resposta sem campo 'choices'.")
                msg = choices[0].get("message") or {}
                content = (msg.get("content") or "").strip()
                if not content and msg.get("reasoning_content"):
                    content = str(msg.get("reasoning_content")).strip()
                if not content:
                    raise RuntimeError("Resposta vazia do modelo.")
                return content
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                errors.append(f"Key #{attempt}: HTTP {exc.code} {err_body}".strip())
                if exc.code not in RETRYABLE_HTTP_CODES and exc.code < 500:
                    break
            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                OSError,
                json.JSONDecodeError,
                RuntimeError,
            ) as exc:
                errors.append(f"Key #{attempt}: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"Falha em {provider}/{model}: " + " | ".join(errors))


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from terminal output."""
    return ANSI_ESCAPE_RE.sub("", text).strip()


def call_freebuff_cli(prompt: str, timeout: int = 45) -> str:
    """Fail fast with clear diagnostic: freebuff CLI requires interactive TTY/TUI."""
    _ = (prompt, timeout, strip_ansi)
    raise RuntimeError(
        "freebuff CLI (AppData/Roaming/npm/freebuff.ps1) é exclusivamente interativo/TUI "
        "sem flag headless (--print/-p) e trava/encerra sem saída em subprocesso não-TTY. "
        "Execute 'freebuff' manualmente em um terminal interativo."
    )


def execute_provider(
    provider: str,
    model: str,
    task: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
    timeout: int,
    explicit_model: bool,
    wait_ollama: bool = False,
) -> str:
    """Execute a single provider/model target."""
    if provider == "commandcode" and model != COMMANDCODE_ALLOWED_MODEL:
        raise ValueError("Provider 'commandcode' is strictly restricted to DeepSeek V4.1 Flash")
    if provider == "ollama":
        with ollama_execution_lock(model, wait=wait_ollama, wait_timeout=min(timeout, 300)):
            if task == "embed" and not explicit_model:
                result = call_ollama_embed(model, prompt, timeout=min(timeout, 120))
                return json.dumps(result, ensure_ascii=False)
            return call_ollama_chat(
                model, prompt, system_prompt, temperature, max_tokens, timeout=timeout
            )
    if provider == "freebuff":
        return call_freebuff_cli(prompt, timeout=min(timeout, 45))
    return call_openai_compatible(
        provider, model, prompt, system_prompt, temperature, max_tokens, timeout=timeout
    )


def build_prompt(prompt: str | None, file_paths: list[str]) -> str:
    """Assemble prompt text including local file contents when --file is given."""
    parts: list[str] = []
    for raw_path in file_paths:
        p = Path(raw_path)
        if not p.is_file():
            raise FileNotFoundError(f"Arquivo não encontrado: {raw_path}")
        text = p.read_text(encoding="utf-8", errors="replace")
        parts.append(f"--- FILE: {p} ---\n{text}\n--- END FILE: {p} ---")

    if prompt:
        parts.append(prompt)
    elif not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            parts.append(stdin_text)

    combined = "\n\n".join(parts).strip()
    if not combined:
        raise ValueError("Informe --prompt, --file ou entrada via stdin.")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roteador v2 zero-dependências para Ollama local e pools multi-provedor."
    )
    parser.add_argument(
        "--provider",
        choices=ALL_PROVIDERS,
        default="auto",
        help="Provedor alvo ou 'auto' para cascata automática por --task.",
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_CASCADES.keys()),
        default="summarize",
        help="Tipo de tarefa usada para roteamento automático.",
    )
    parser.add_argument("--model", help="Sobrescreve o modelo padrão.")
    parser.add_argument("--prompt", help="Prompt para o worker.")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Arquivo local a ser lido e enviado ao worker (pode repetir).",
    )
    parser.add_argument("--system", help="Prompt de sistema opcional.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout em segundos por requisição HTTP (padrão: 600).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emite resultado estruturado em JSON.",
    )
    parser.add_argument(
        "--check-ollama",
        action="store_true",
        help="Inspeciona lockfile e /api/ps do Ollama e imprime status em JSON.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Lista todos os provedores configurados, modelos disponíveis e especialidades.",
    )
    return parser.parse_args()


def load_env_candidates() -> None:
    """Load env files in priority order without overriding already-set variables."""
    skill_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path.cwd() / ".env",
        skill_root / ".env",
        Path.home() / ".llm-router.env",
    ]
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        load_env_file(path)


def main() -> int:
    load_env_candidates()
    args = parse_args()

    if args.list_models:
        print(json.dumps(MODEL_FLEET_CATALOG, ensure_ascii=False, indent=2))
        return 0

    if args.provider == "commandcode" and args.model and args.model != COMMANDCODE_ALLOWED_MODEL:
        print("Provider 'commandcode' is strictly restricted to DeepSeek V4.1 Flash", file=sys.stderr)
        return 2

    if args.check_ollama:
        print(json.dumps(check_ollama_status(), ensure_ascii=False, indent=2))
        return 0

    try:
        full_prompt = build_prompt(args.prompt, args.file)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Erro de entrada: {exc}", file=sys.stderr)
        return 2

    if args.provider == "auto":
        cascade = TASK_CASCADES[args.task]
        last_exc: Exception | None = None
        for idx, (cand_provider, cand_model) in enumerate(cascade):
            chosen_model = args.model if (args.model and idx == 0) else cand_model
            try:
                output = execute_provider(
                    cand_provider,
                    chosen_model,
                    args.task,
                    full_prompt,
                    args.system,
                    args.temperature,
                    args.max_tokens,
                    args.timeout,
                    explicit_model=bool(args.model and idx == 0),
                    wait_ollama=False,
                )
                provider = cand_provider
                model = chosen_model
                break
            except OllamaBusyError as exc:
                last_exc = exc
                if idx + 1 < len(cascade):
                    next_p, next_m = cascade[idx + 1]
                    print(
                        f"[auto-fallback] Ollama ocupado/carregado ({exc}); "
                        f"desviando para nuvem ({next_p}/{next_m}).",
                        file=sys.stderr,
                    )
            except Exception as exc:
                last_exc = exc
                if idx + 1 < len(cascade):
                    next_p, next_m = cascade[idx + 1]
                    print(
                        f"[auto-fallback] {cand_provider}/{chosen_model} falhou ({exc}); "
                        f"tentando {next_p}/{next_m}.",
                        file=sys.stderr,
                    )
        else:
            print(f"Erro em toda a cascata ({args.task}): {last_exc}", file=sys.stderr)
            return 1
    else:
        provider = args.provider
        cascade_match = next(
            (m for p, m in TASK_CASCADES[args.task] if p == provider),
            DEFAULT_PROVIDER_MODELS[provider],
        )
        model = args.model or cascade_match
        try:
            output = execute_provider(
                provider,
                model,
                args.task,
                full_prompt,
                args.system,
                args.temperature,
                args.max_tokens,
                args.timeout,
                explicit_model=bool(args.model),
                wait_ollama=(provider == "ollama"),
            )
        except Exception as exc:
            print(f"Erro ao executar worker ({provider}/{model}): {exc}", file=sys.stderr)
            return 1

    if args.json:
        print(
            json.dumps(
                {"provider": provider, "model": model, "task": args.task, "output": output},
                ensure_ascii=False,
            )
        )
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

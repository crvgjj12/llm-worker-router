#!/usr/bin/env python3
"""Zero-dependency CLI router v2 for local Ollama and multi-provider cloud pools."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import re
import socket
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


class OllamaBusyError(RuntimeError):
    """Raised when Ollama is locked by another process or holding a heavy model in VRAM."""


_SUMMARIZE_CASCADE: list[tuple[str, str]] = [
    ("groq", "qwen/qwen3.8-27b"),
    ("gemini", "gemini-3-flash-preview"),
    ("openrouter", "openrouter/free"),
    ("cloudflare", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
    ("ollama", "qwen3.5:4b"),
    ("cerebras", "gpt-oss-120b"),
]

TASK_CASCADES: dict[str, list[tuple[str, str]]] = {
    "summarize": list(_SUMMARIZE_CASCADE),
    "general": list(_SUMMARIZE_CASCADE),
    "code": [
        ("groq", "qwen/qwen3.8-27b"),
        ("cloudflare", "@cf/qwen/qwen2.5-coder-32b-instruct"),
        ("github", "openai/gpt-4.1-mini"),
        ("openrouter", "openrouter/free"),
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
        ("openrouter", "openrouter/free"),
        ("gemini", "gemini-3.1-flash-lite-preview"),
        ("ollama", "qwen3.5:4b"),
        ("cerebras", "gpt-oss-120b"),
    ],
    "batch-cloud": [
        ("groq", "openai/gpt-oss-120b"),
        ("gemini", "gemini-3-flash-preview"),
        ("openrouter", "openrouter/free"),
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
        ("gemini", "gemini-3-flash-preview"),
        ("github", "deepseek/deepseek-r1"),
        ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        ("ollama", "qwen3.6:35b-a3b"),
        ("cerebras", "gpt-oss-120b"),
    ],
    "fallback-cloud": [
        ("openrouter", "openrouter/free"),
        ("github", "openai/gpt-4.1"),
        ("cloudflare", "@cf/qwen/qwen2.5-coder-32b-instruct"),
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
    "github": "openai/gpt-4o-mini",
    "openrouter": "openrouter/free",
    "opencode": "big-pickle",
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
    "freebuff",
]


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


def http_post_json(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: int = 600,
) -> dict:
    """POST JSON payload and return decoded JSON response."""
    req_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "llm-worker-router/2.0",
    }
    if headers:
        req_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
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

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    errors: list[str] = []
    for endpoint in endpoints:
        for attempt, api_key in enumerate(keys, start=1):
            headers = {"Authorization": f"Bearer {api_key}", **extra_headers}
            try:
                res = http_post_json(endpoint, payload, headers=headers, timeout=timeout)
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
    return parser.parse_args()


def load_env_candidates() -> None:
    """Load env files in priority order without overriding already-set variables:
    1. Already exported environment variables (os.environ)
    2. .env in current working directory (Path.cwd() / ".env")
    3. .env in skill root directory (Path(__file__).resolve().parent.parent / ".env")
    4. Global ~/.llm-router.env (Path.home() / ".llm-router.env")
    """
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

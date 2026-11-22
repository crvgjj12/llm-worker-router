---
name: llm-worker-router
description: >-
  Delega subtarefas de resumo, leitura de logs/arquivos grandes, geração de código e testes,
  embeddings locais, contexto longo e segunda opinião para Ollama local e 8 provedores em nuvem
  (Groq, Cerebras, Google AI Studio 5 chaves, NVIDIA NIM, Cloudflare Workers AI, GitHub Models,
  OpenRouter e OpenCode Zen) com rotação de até 9 chaves e fallback em cascata.
---

# LLM Worker Router (v2)

Skill para delegar subtarefas pesadas, repetitivas ou de leitura extensa a workers LLM externos/locais via CLI zero-dependências (`scripts/delegate_llm.py`), mantendo o contexto do agente principal limpo e com fallback em cascata no modo `--provider auto`.

## Matriz de Rotas (`--provider auto --task <TASK>`) — Cloud-First

> **Regra Cloud-First**: Sempre use `--provider auto` (nuvem primeiro). Reserve `--provider ollama` estritamente para **dados sensíveis/privados (offline)**, `--task embed` ou fallback final quando toda a nuvem falhar.

| `--task` | 1º Alvo (Primário) | Cascata de Fallback Automático | Uso Recomendado |
| :--- | :--- | :--- | :--- |
| `summarize` | `groq/qwen/qwen3.8-27b` | `gemini/gemini-3-flash-preview` → `openrouter/openrouter/free` → `cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast` → `ollama/qwen3.5:4b` → `cerebras/gpt-oss-120b` | Resumir logs, diffs e arquivos grandes (>300 linhas) em nuvem |
| `code` | `groq/qwen/qwen3.8-27b` | `cloudflare/@cf/qwen/qwen2.5-coder-32b-instruct` → `github/openai/gpt-4.1-mini` → `openference/DeepSeek-V4.1-Flash-1017-Free` → `openrouter/openrouter/free` → `ollama/qwen2.5-coder:7b` → `cerebras/qwen-3.8-27b` | Gerar código, scripts, docstrings e testes unitários em nuvem |
| `embed` | `ollama/qwen3-embedding:0.6b` | — | Gerar vetores de embedding (`1024` dims) em JSON (local) |
| `local-heavy` | `ollama/qwen3.6:35b-a3b` | `gemini/gemini-3-flash-preview` → `openrouter/openrouter/free` → `cerebras/gpt-oss-120b` | Raciocínio local pesado (quando solicitado explicitamente) |
| `fast-cloud` | `groq/qwen/qwen3.8-27b` | `openrouter/openrouter/free` → `gemini/gemini-3.1-flash-lite-preview` → `ollama/qwen3.5:4b` → `cerebras/gpt-oss-120b` | Inferência rápida na nuvem (`0.25s`) |
| `batch-cloud` | `groq/openai/gpt-oss-120b` | `gemini/gemini-3-flash-preview` → `openrouter/openrouter/free` → `ollama/qwen3.5:4b` → `cerebras/gpt-oss-120b` | Processamento em lote e síntese rápida |
| `long-ctx` | `gemini/gemini-3-flash-preview` | `nvidia/moonshotai/kimi-k3.1-pro` → `openrouter/nvidia/nemotron-3-ultra-550b-a55b:free` → `ollama/qwen3.5:4b` → `cerebras/gpt-oss-120b` | Contexto longo (5 chaves Gemini em rotação) |
| `opencode-go` | `opencode/big-pickle` | `openrouter/openrouter/free` → `ollama/qwen3.5:4b` → `cerebras/qwen-3.8-27b` | Modelos OpenCode Zen com fallback gratuito |
| `reason` | `nvidia/nvidia/nemotron-3-ultra-550b-a55b` | `nvidia/mistralai/mistral-large-4-1t-a52b` → `gemini/gemini-3-flash-preview` → `openference/GLM-5.3-Flash-Free` → `github/deepseek/deepseek-r1` → `openrouter/nvidia/nemotron-3-ultra-550b-a55b:free` → `ollama/qwen3.6:35b-a3b` → `cerebras/gpt-oss-120b` | Auditoria arquitetural e raciocínio profundo |
| `fallback-cloud` | `openrouter/openrouter/free` | `openference/DeepSeek-V4.1-Flash-1017-Free` → `github/openai/gpt-4.1` → `cloudflare/@cf/qwen/qwen2.5-coder-32b-instruct` → `opencode/big-pickle` → `ollama/qwen3.5:4b` → `cerebras/gpt-oss-120b` | Rede de segurança multi-provedor gratuita |
| `general` | `groq/qwen/qwen3.8-27b` | `gemini/gemini-3-flash-preview` → `openrouter/openrouter/free` → `cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast` → `ollama/qwen3.5:4b` → `cerebras/gpt-oss-120b` | Tarefas gerais com fallback completo |

> **Regra do Cerebras**: Em `--provider auto`, o `cerebras` é usado **EXCLUSIVAMENTE como último recurso pós-Ollama** (somente se os provedores primários E o `ollama` local falharem, ou quando chamado explicitamente com `--provider cerebras`).

Consulte todos os modelos validados (`HTTP 200`), latências medidas e slugs descontinuados em [references/model-catalog.md](references/model-catalog.md).

> **Proteção de VRAM / Concorrência Ollama (8 GB VRAM)**: Nunca dispare chamadas paralelas ao Ollama. O roteador usa `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`, lockfile exclusivo (`~/.llm-router-ollama.lock`) e checagem via `/api/ps`. Use `--check-ollama` para inspecionar o estado da GPU; em `--provider auto`, se o Ollama estiver ocupado ou com modelo pesado diferente carregado, o roteador **não espera** e desvia imediatamente para a nuvem gratuita (`groq`/`openrouter`/`gemini`). Para dados sensíveis que exijam execução estritamente local, use `--provider ollama` (aguarda liberação do lock).

> **Nota de timeout**: `--timeout 600` (10 minutos) é o padrão. Mantenha `600s` em modelos pesados ou com cold-start da NVIDIA NIM (`z-ai/glm-5.3-flash`, `moonshotai/kimi-k3.1-pro`, `nvidia/nemotron-3-ultra-550b-a55b`). Em `TimeoutError`, `401`, `403`, `408`, `429` ou `5xx`, o roteador tenta a próxima chave (`_1.._9`) antes de acionar o próximo provedor da cascata.

## Como Invocar

```powershell
# Resumo local de arquivo grande
python "<SKILL_DIR>\scripts\delegate_llm.py" --provider auto --task summarize --file "logs\app.log" --prompt "Resuma os erros críticos."

# Lote rápido via Cerebras
python "<SKILL_DIR>\scripts\delegate_llm.py" --provider auto --task batch-cloud --prompt "Extraia as entidades deste texto."

# Contexto longo via pool de 5 chaves Gemini
python "<SKILL_DIR>\scripts\delegate_llm.py" --provider auto --task long-ctx --file "docs\spec.md" --prompt "Audite inconsistências."

# Auditoria profunda via NVIDIA NIM
python "<SKILL_DIR>\scripts\delegate_llm.py" --provider auto --task reason --prompt "Revise esta arquitetura."

# Provedor explícito
python "<SKILL_DIR>\scripts\delegate_llm.py" --provider github --model "openai/gpt-4.1-mini" --prompt "Explique este trecho."
```

## Parâmetros Principais

| Flag | Valores / Descrição | Padrão |
| :--- | :--- | :--- |
| `--provider` | `auto`, `ollama`, `groq`, `nvidia`, `cerebras`, `gemini`, `cloudflare`, `github`, `openrouter`, `opencode`, `openference`, `venice` | `auto` |
| `--task` | `summarize`, `code`, `embed`, `local-heavy`, `fast-cloud`, `batch-cloud`, `long-ctx`, `opencode-go`, `reason`, `fallback-cloud` | `summarize` |
| `--file` | Caminho de arquivo local lido pelo script (pode repetir) | — |
| `--prompt` | Instrução para o modelo worker | — |
| `--model` | Sobrescreve o modelo padrão do provedor/tarefa | automático |
| `--system` | Mensagem de sistema opcional | — |
| `--temperature` | Temperatura de amostragem (`float`) | `0.2` |
| `--max-tokens` | Limite máximo de tokens na resposta (`int`) | `2048` |
| `--timeout` | Timeout em segundos por requisição HTTP | `600` |
| `--check-ollama` | Inspeciona lockfile e `/api/ps` do Ollama em JSON | `false` |
| `--json` | Retorna metadados + resposta em JSON | `false` |

## Credenciais e Segurança

- Variáveis lidas de `~\.llm-router.env` (`*_API_KEY_1..9`, `CLOUDFLARE_ACCOUNT_ID`, `OLLAMA_BASE_URL`) sem sobrescrever variáveis já definidas no ambiente e nunca impressas em logs.

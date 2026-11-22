# Catálogo de Modelos Ativos — `llm-worker-router` v2

Catálogo validado ao vivo em `2026-11-21` contra Ollama local e os 8 provedores em nuvem suportados, com rotação de até 9 chaves (`_1.._9`) e `--timeout 600`.

## 1. Modelos Locais (Ollama — Fallback Final / Dados Sensíveis, 8 GB VRAM)

> **Regra Cloud-First & Concorrência**: Sempre use `--provider auto` (nuvem primeiro). Reserve `--provider ollama` estritamente para **dados sensíveis/privados (offline)**, `--task embed` ou fallback final quando toda a nuvem falhar. `OLLAMA_MAX_LOADED_MODELS=1` e `OLLAMA_NUM_PARALLEL=1` ativos no Windows; nunca dispare chamadas paralelas ao Ollama (`--check-ollama` / `~/.llm-router-ollama.lock`).

| Model ID exato | Provedor | Latência | Timeout | Bizu de Uso (Especialidade) |
| :--- | :--- | :--- | :--- | :--- |
| `qwen3.5:4b` | `ollama` | Local (rápido) | `180s`–`600s` | **Fallback local** de `summarize`/`fast-cloud`/`batch-cloud`/`long-ctx`/`fallback-cloud`, ou `--provider ollama` para resumos privados. |
| `qwen2.5-coder:7b` | `ollama` | Local (rápido) | `180s`–`600s` | **Fallback local** de `code`, ou `--provider ollama --task code` para código confidencial offline. |
| `qwen3-embedding:0.6b` | `ollama` | Local (rápido) | `120s` | **Padrão `--task embed`**: gerar vetores de embedding (`1024` dims). |
| `qwen3.6:35b-a3b` | `ollama` | Local (pesado) | `600s` | **Padrão `--task local-heavy`** e fallback local de `reason`. |

## 2. Groq (Pool 3 Chaves)

| Model ID exato | Provedor | Latência medida | Timeout | Bizu de Uso (Especialidade) |
| :--- | :--- | :--- | :--- | :--- |
| `qwen/qwen3.8-27b` | `groq` | `0.25s` | `60s`–`600s` | **Padrão `--task fast-cloud`**: menor latência na Groq; código e resumos técnicos. |
| `openai/gpt-oss-20b` | `groq` | `0.31s` | `60s`–`600s` | Triagem, classificação e extração rápida. |
| `openai/gpt-oss-120b` | `groq` | `0.41s` | `60s`–`600s` | Raciocínio rápido e síntese geral em nuvem. |
| `allam-2-7b` | `groq` | `0.34s` | `60s`–`600s` | Leve multilíngue (ênfase árabe/inglês); evitar para código/PT-BR. |

## 3. Cerebras (`https://api.cerebras.ai/v1`) — Último Recurso Pós-Ollama

> **REGRA ESTRITA**: O provedor `cerebras` é usado **EXCLUSIVAMENTE como último recurso absoluto** no modo `--provider auto` — acionado somente se os provedores primários **E** o `ollama` local falharem (ou quando chamado explicitamente via `--provider cerebras`). Nunca escolha `cerebras` antes do Ollama.

| Model ID exato | Provedor | Latência medida | Timeout | Bizu de Uso (Especialidade) |
| :--- | :--- | :--- | :--- | :--- |
| `qwen-3.8-27b` | `cerebras` | `0.25s` | `60s`–`600s` | **Último recurso pós-Ollama** em `summarize`, `code`, `fast-cloud`, `opencode-go` (ou `--provider cerebras`). |
| `gpt-oss-120b` | `cerebras` | `0.34s` | `60s`–`600s` | **Último recurso pós-Ollama** em `local-heavy`, `batch-cloud`, `long-ctx`, `reason`, `fallback-cloud`, `general` (ou `--provider cerebras`). |

## 4. Google AI Studio (`gemini`, Pool 5 Chaves `GEMINI_API_KEY_1..5`)

| Model ID exato | Provedor | Latência medida | Timeout | Bizu de Uso (Especialidade) |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-3-flash-preview` | `gemini` | `1.02s` | `600s` | **Padrão `--task long-ctx`** e fallback de `reason`: contexto longo (1M+) e análise multimodal/texto com 5 chaves em rotação. |
| `gemini-3.1-flash-lite-preview` | `gemini` | `1.00s` | `600s` | Alternativa mais leve e rápida no pool de 5 chaves Gemini. |

## 5. NVIDIA NIM (Pool 3 Chaves)

| Model ID exato | Provedor | Latência medida | Timeout | Bizu de Uso (Especialidade) |
| :--- | :--- | :--- | :--- | :--- |
| `nvidia/nemotron-3-ultra-550b-a55b` | `nvidia` | `1.31s` | `600s` | **Padrão `--task reason`**: flagship 550B MoE para auditoria de arquitetura e raciocínio complexo. |
| `mistralai/mistral-large-4-1t-a52b` | `nvidia` | `2.73s` | `600s` | Top 1T MoE para revisão profunda de design/código. |
| `nvidia/nemotron-3.5-super-vl-120b-a12b` | `nvidia` | `0.88s` | `600s` | 120B MoE rápido para síntese técnica. |
| `poolside/laguna-xs-2.1` | `nvidia` | `0.86s` | `600s` | Especialista em engenharia de software e patches. |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | `nvidia` | `1.38s` | `600s` | Equilíbrio velocidade/qualidade. |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `nvidia` | `1.66s` | `600s` | Raciocínio rápido estruturado. |
| `nvidia/nemotron-3.5-bolt-e4b` | `nvidia` | `0.52s` | `600s` | Ultraleve para formatação e extração simples. |
| `nvidia/ising-calibration-1.5-31b` | `nvidia` | `0.45s` | `600s` | Raciocínio técnico enxuto de baixa latência. |
| `nvidia/riva-translate-4b-instruct-v2` | `nvidia` | `0.56s` | `600s` | Exclusivo para tradução entre idiomas. |
| `google/gemma-4-31b-it` | `nvidia` | `29.56s` | `600s` | Redação e revisão de documentação. |
| `meta/muse-glimmer-30b` | `nvidia` | `40.28s` | `600s` | Geração e análise geral de 30B. |
| `moonshotai/kimi-k3.1-pro` | `nvidia` | `42.98s` | `600s` | **Fallback de `--task long-ctx`**: contexto longo e segunda opinião detalhada. |
| `stepfun-ai/step-5-preview` | `nvidia` | `43.72s` | `600s` | Raciocínio avançado (latência moderada). |
| `z-ai/glm-5.3-flash` | `nvidia` | `832.25s` (2 tent.) | `600s` | Ativo, porém muito lento (cold-start); usar apenas se necessário com `--timeout 600`. |

## 6. Cloudflare Workers AI (`cloudflare`)

| Model ID exato | Provedor | Latência medida | Timeout | Bizu de Uso (Especialidade) |
| :--- | :--- | :--- | :--- | :--- |
| `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | `cloudflare` | `0.41s` | `600s` | **Fallback em `--task fallback-cloud`**: inferência rápida 70B FP8 na borda Cloudflare. |
| `@cf/qwen/qwen2.5-coder-32b-instruct` | `cloudflare` | `0.52s` | `600s` | Geração/revisão de código em nuvem via Cloudflare. |
| `@cf/openai/gpt-oss-120b` | `cloudflare` | `1.23s` | `600s` | Raciocínio 120B na Cloudflare Workers AI. |

## 7. GitHub Models (`github` — `https://models.github.ai/inference`)

| Model ID exato | Provedor | Latência medida | Timeout | Bizu de Uso (Especialidade) |
| :--- | :--- | :--- | :--- | :--- |
| `openai/gpt-4o-mini` | `github` | `<1s` | `600s` | **Fallback em `--task fallback-cloud`**: tarefas rápidas gerais via PAT GitHub. |
| `openai/gpt-4.1-mini` | `github` | `<1s` | `600s` | Instruções estruturadas e revisão leve de código. |
| `openai/gpt-4.1` | `github` | `<1s` | `600s` | Raciocínio e código de alta precisão via GitHub Models. |
| `deepseek/deepseek-r1` | `github` | `<1s` | `600s` | Raciocínio passo a passo (chain-of-thought) via GitHub Models. |
| `meta/llama-3.3-70b-instruct` | `github` | `<1s` | `600s` | Modelo aberto 70B para síntese e análise geral. |

## 8. OpenRouter (`openrouter`)

| Model ID exato | Provedor | Latência medida | Timeout | Bizu de Uso (Especialidade) |
| :--- | :--- | :--- | :--- | :--- |
| `openrouter/free` | `openrouter` | `2.19s` | `600s` | **Padrão `--task fallback-cloud`** e fallback de `opencode-go`: meta-roteador automático para modelos gratuitos ativos. |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | `openrouter` | `0.59s` | `600s` | 550B MoE gratuito via OpenRouter com baixíssima latência. |
| `nvidia/nemotron-3.5-super-vl:free` | `openrouter` | `0.75s` | `600s` | 120B MoE gratuito para análise geral. |
| `baidu/ernie-5.5-flash-preview:free` | `openrouter` | `1.27s` | `600s` | Flash gratuito alternativo para texto e resumos. |

## 9. OpenCode Zen/Go (`opencode`) & Freebuff CLI

- **OpenCode Zen (`opencode`)**: A chave `OPENCODE_API_KEY_1` e o endpoint `https://opencode.ai/zen/v1` estão ativos (`39` modelos listados em `/models`, incluindo `big-pickle`, `beam-free`, `mimo-v2.6-flash-free`, `deepseek-v4.1-flash-free`, `flint-40b-free`, `nemotron-3.5-super-vl-free`, `jev-1.13-free`, `nemotron-3.5-lightning-free`). No teste ao vivo de `2026-11-21`, os modelos `-free` retornaram `HTTP 429 FreeUsageLimitError` (cota gratuita atingida na conta) e os modelos pagos retornaram `HTTP 402 Insufficient balance`. A rota `--task opencode-go` tenta `opencode/big-pickle` e faz **fallback automático** para `openrouter/free` → `ollama/qwen3.5:4b`.
- **Freebuff CLI (`freebuff`)**: Quando instalado globalmente via npm (`freebuff`), `freebuff --help` confirma apenas uso interativo (`login`, `--continue`, `--cwd`, `--trust-agents`), **sem flag headless/não-interativa** (`-p` / `--print`). Não integrado como worker CLI em `delegate_llm.py`.

## 10. Indisponíveis / Deprecados na Conta (NÃO CHAMAR)

- **Groq (`404`)**: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `minimaxai/minimax-m2.7`, `meta-llama/muse-spark-open`.
- **Cerebras (`404`)**: `llama3.1-8b`, `llama-3.3-70b`, `qwen-3-32b`, `zai-glm-4.7`.
- **Google AI Studio**: `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro`, `gemma-3-27b-it` (`404`); `gemma-4-31b-it`, `gemma-4-26b-a4b-it` (`503`); `gemini-3.1-pro-preview*` (`429`).
- **Cloudflare**: `@cf/meta/llama-3.1-8b-instruct`, `@cf/deepseek-ai/deepseek-r1-distill-qwen-32b` (`410`); `@cf/google/gemma-3-12b-it` (`403`).
- **NVIDIA NIM**: `deepseek-ai/deepseek-v4.1-flash` (`410`), `z-ai/glm-5.3` (`410`), `mistralai/mistral-large-4` (`404`), `minimaxai/minimax-m3.1-flash` (`503`), `bigcode/starcoder2-15b`, `google/gemma-3-12b-it`, `google/gemma-3-4b-it`, `nvidia/cosmos-reason2-8b` (`404`).

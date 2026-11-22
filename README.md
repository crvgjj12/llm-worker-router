# llm-worker-router

Skill portável (Windows, Linux e macOS) e zero-dependências em Python 3 (`stdlib` pura) para delegar subtarefas pesadas — resumo de logs/arquivos grandes, geração de código/testes, embeddings locais, contexto longo e segunda opinião arquitetural — para múltiplos provedores LLM em nuvem com fallback local no Ollama.

## Arquitetura *Cloud-First* e Proteção de VRAM

- **Cloud-first em `--provider auto`**: todas as rotas principais (`summarize`, `code`, `fast-cloud`, `batch-cloud`, `long-ctx`, `reason`, `fallback-cloud`, `general`) consultam provedores gratuitos em nuvem primeiro (`Groq`, `Google AI Studio`, `NVIDIA NIM`, `Cloudflare Workers AI`, `GitHub Models`, `OpenRouter`, `OpenCode Zen`).
- **Ollama como fallback final / uso offline**: o Ollama local entra apenas quando a nuvem falha, para `--task embed` / `--task local-heavy`, ou quando invocado explicitamente via `--provider ollama` (dados sensíveis/privados).
- **Cerebras como último recurso absoluto**: em `--provider auto`, `cerebras` só é acionado se os provedores primários **e** o Ollama local falharem (ou via `--provider cerebras`).
- **Rotação de até 9 chaves por provedor (`_1.._9`)**: embaralhamento a cada execução com failover automático em `401`, `403`, `408`, `429`, `5xx`, `504` e `TimeoutError` (timeout padrão de `600s`).
- **Proteção de VRAM no Ollama (GPUs de 8 GB)**: lockfile exclusivo (`~/.llm-router-ollama.lock` com verificação de PID vivo e expiração de órfãos >10 min) + checagem de `GET /api/ps` + flag `--check-ollama`. Em `--provider auto`, se o Ollama estiver ocupado ou com modelo pesado diferente na VRAM, o roteador desvia imediatamente para a nuvem.

## Tabela Resumida de `--task` (`--provider auto`)

| `--task` | 1º Alvo (Primário) | Cascata de Fallback | Uso Principal |
| :--- | :--- | :--- | :--- |
| `summarize` | `groq/qwen/qwen3.8-27b` | `gemini` → `openrouter` → `cloudflare` → `ollama` → `cerebras` | Resumir logs e arquivos grandes (`--file`) |
| `code` | `groq/qwen/qwen3.8-27b` | `cloudflare` → `github` → `openrouter` → `ollama` → `cerebras` | Gerar código, docstrings e testes |
| `fast-cloud` | `groq/qwen/qwen3.8-27b` | `openrouter` → `gemini` → `ollama` → `cerebras` | Tarefas rápidas em nuvem |
| `batch-cloud` | `groq/openai/gpt-oss-120b` | `gemini` → `openrouter` → `ollama` → `cerebras` | Lotes e síntese rápida |
| `long-ctx` | `gemini/gemini-3-flash-preview` | `nvidia` → `openrouter` → `ollama` → `cerebras` | Contexto longo (pool de 5 chaves Gemini) |
| `reason` | `nvidia/nvidia/nemotron-3-ultra-550b-a55b` | `nvidia` → `gemini` → `github` → `openrouter` → `ollama` → `cerebras` | Auditoria e segunda opinião profunda |
| `opencode-go` | `opencode/big-pickle` | `openrouter` → `ollama` → `cerebras` | Modelos OpenCode Zen |
| `fallback-cloud` | `openrouter/openrouter/free` | `github` → `cloudflare` → `opencode` → `ollama` → `cerebras` | Rede de segurança multi-provedor |
| `embed` | `ollama/qwen3-embedding:0.6b` | — | Embeddings locais (`1024` dims) |
| `local-heavy` | `ollama/qwen3.6:35b-a3b` | `gemini` → `openrouter` → `cerebras` | Raciocínio local pesado |

Consulte o catálogo completo de modelos validados e latências em [references/model-catalog.md](references/model-catalog.md).

## Configuração Rápida

1. Copie o template de credenciais para o seu diretório `HOME` (ou `.env` local):

   ```bash
   cp .env.example ~/.llm-router.env
   ```

2. Preencha apenas as chaves dos provedores que deseja utilizar em `~/.llm-router.env`. O roteador carrega variáveis na ordem: variáveis já exportadas no sistema → `./.env` → `.env` na raiz da skill → `~/.llm-router.env`.

3. (Opcional, para GPUs de 8 GB) Configure no sistema `OLLAMA_MAX_LOADED_MODELS=1` e `OLLAMA_NUM_PARALLEL=1`.

## Modelos Locais (Ollama): Instalação e Por que Foram Escolhidos

O Ollama atua como camada **offline/privada** (`--provider ollama`), provedor de **embeddings** (`--task embed`) e **fallback final** caso todos os provedores em nuvem fiquem indisponíveis.

### 1. Como instalar o Ollama

- **Windows**:
  ```powershell
  winget install Ollama.Ollama
  ```
  *(Ou baixe o instalador em [ollama.com/download](https://ollama.com/download).)*
- **Linux**:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- **macOS**:
  ```bash
  brew install --cask ollama
  ```

### 2. Como baixar os modelos locais do projeto

Após instalar e iniciar o serviço do Ollama (`ollama serve`), baixe os 4 modelos configurados no roteador:

```bash
ollama pull qwen3.5:4b
ollama pull qwen2.5-coder:7b
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.6:35b-a3b
```

### 3. Por que cada modelo local foi escolhido

| Modelo | Papel no Roteador | Por que foi escolhido |
| :--- | :--- | :--- |
| `qwen3.5:4b` | Fallback local de `summarize`, `fast-cloud`, `batch-cloud`, `long-ctx`, `fallback-cloud` e `--provider ollama` | Modelo leve (4B) com baixíssimo consumo de VRAM (~3–4 GB), ideal para resumir logs e documentos rapidamente sem saturar GPUs de 8 GB. |
| `qwen2.5-coder:7b` | Fallback local de `code` e geração de código offline | Especializado em programação (código, refatoração, docstrings e testes unitários) e cabe com folga em 8 GB de VRAM em quantização padrão. |
| `qwen3-embedding:0.6b` | Padrão exclusivo de `--task embed` | Gera vetores de `1024` dimensões com pegada mínima de memória (~0.6B), permitindo indexação semântica local quase instantânea. |
| `qwen3.6:35b-a3b` | Padrão de `--task local-heavy` e fallback local de `reason` | Arquitetura MoE (*Mixture-of-Experts* com apenas 3B de parâmetros ativos por token sobre 35B totais), entregando raciocínio local de alta qualidade com uso viável em máquinas de 8 GB VRAM + 32 GB RAM. |

## Instalação nos Agentes

Copie a pasta `llm-worker-router` para o diretório global de skills do(s) seu(s) agente(s):

- **Agent Skills compartilhado**: `~/.agents/skills/llm-worker-router`
- **Claude Code**: `~/.claude/skills/llm-worker-router`
- **Antigravity**: `~/.gemini/antigravity/skills/llm-worker-router`
- **DeepSeek Harness (DSH)**: `~/.dsh/skills/llm-worker-router`

## Exemplos de Uso CLI

```bash
# Resumir arquivo grande em nuvem (cloud-first)
python ~/.agents/skills/llm-worker-router/scripts/delegate_llm.py --provider auto --task summarize --file app.log --prompt "Resuma os erros críticos."

# Verificar disponibilidade do Ollama local
python ~/.agents/skills/llm-worker-router/scripts/delegate_llm.py --check-ollama

# Execução estritamente offline para dados sensíveis
python ~/.agents/skills/llm-worker-router/scripts/delegate_llm.py --provider ollama --task summarize --file segredo.txt --prompt "Extraia os pontos principais."
```

## Licença

[MIT](LICENSE)

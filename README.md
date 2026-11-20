# The Receptionist

Cost-aware, safety-first AI query routing between local (edge) and cloud foundation models with transparent metrics and streaming chat UI.

## Product scope

The Receptionist routes everyday developer questions between local Ollama and cloud OpenAI models. It is for individual developers who want control over when prompts leave their machine and a clear view of which model answered, why it was selected, and how routing affects estimated cloud spend.

Local Only is the default. Select Auto to let the classifier route prompts to the cloud, or select Send next prompt to cloud to send one prompt before returning to your selected mode. Routing explanations and the cost dashboard make each choice visible.

| Workflow | A passing answer |
| --- | --- |
| Quick developer lookup | Gives the correct command, syntax, or concept with enough context to use it. |
| Code explanation | Explains what the snippet does and identifies the relevant behavior or edge case. |
| Small-snippet debugging | Identifies a plausible cause and gives a concrete next step the developer can verify. |

### Success criteria

- **Privacy** — Local Only submissions stay local even when Ollama fails. The one-shot cloud override explicitly authorizes a separate cloud submission.
- **Answer quality** — Compare pass rates for automatic routing, all-local, and all-cloud runs using the same held-out prompts and workflow criteria.
- **Estimated cost** — Compare cloud spend with the configured all-cloud baseline and document the pricing and usage assumptions.
- **Latency** — Compare median and p95 time to a complete answer across the same prompts and record the local hardware used.

Use the first baseline run to set a minimum acceptable answer pass rate, minimum worthwhile savings, and maximum acceptable p95 latency before tuning the router.

## ✨ Key Features
* Heuristic safety & complexity classifier (no heavy deps) decides local vs cloud.
* One-shot cloud override for user-controlled escalation.
* Unified routing layer with streaming support (OpenAI implemented; graceful fallbacks otherwise).
* Native Streamlit `st.chat_message` interface + high-contrast custom theme (`.streamlit/config.toml`).
* Per‑1k token pricing model + savings vs all‑cloud baseline (selects highest available baseline model).
* Model provenance badges (Local · model / Cloud · model) inline with each answer.
* Carbon proxy metric approximating avoided remote tokens.
* GPT‑5 preview ready (just set `cloud_model.name: gpt-5-preview` and export OPENAI_API_KEY).
* Sanitized config: no embedded secrets; environment variable substitution pattern supported.

## 🧩 Architecture
```
User → Classifier → Router ─┬─ LocalModel (Ollama) 
                                           └─ CloudModel (OpenAI, extensible stubs for Anthropic/Gemini)
                               ↓
                         MetricsTracker (pricing, savings, carbon)
```

## 🔐 Security & Privacy
* API keys pulled from environment (e.g. `OPENAI_API_KEY`).
* Config allows `${VAR_NAME}` placeholders; do not commit secrets.
* In Auto mode, some keywords trigger cloud routing. Cloud requests send the current prompt and configured system prompt. Chat history stays in the Streamlit session, and the dashboard records usage metadata.

## 🚀 Getting Started
1. Clone the repo
2. Create / activate a Python 3.11+ environment
3. Install dependencies:
    ```shell
    pip install -r requirements.txt
    ```
4. Export your keys (PowerShell example):
    ```powershell
    $env:OPENAI_API_KEY = "sk-..."
    ```
5. (Optional) Pull a local Ollama model (example):
    ```shell
    ollama pull gemma2:2b
    ```
6. Run the app:
    ```shell
    streamlit run streamlit_app.py
    ```

## ⚙️ Configuration (`config.yaml`)
```
local_model:
   name: gemma2:2b
   provider: ollama
cloud_model:
   name: gpt-4o-mini  # or gpt-5-preview
   provider: openai
pricing:
   openai/gpt-4o-mini: { per_1k: 0.150 }
   openai/gpt-5-preview: { per_1k: 0.300 }
   ollama/gemma2:2b: { per_1k: 0.0 }
   # ...
carbon_proxy:
   enabled: true
system_prompt: >
   You are The Receptionist's Cloud Supervisor ...
```

## 📊 Metrics Logic
* Baseline = all tokens hypothetically sent to a premium cloud model (prefers `gpt-5-preview` > `gpt-4-turbo` > `gpt-4o-mini`).
* Actual = sum(model_tokens * per-token price).
* Savings = Baseline − Actual.

## 🧪 Extending Providers
Add logic inside `CloudModel` for `anthropic` / `google` — implement both `generate()` and `stream()` to match the OpenAI pattern.

## 🛣️ Roadmap Ideas
* Anthropic / Gemini streaming
* Fine-grained token accounting (tiktoken)
* Adaptive learning router with feedback loop
* PII redaction + opt‑in analytics
* Evaluation harness for routing precision

## 📝 License
MIT

---
PRs welcome. Enjoy saving tokens 🌱

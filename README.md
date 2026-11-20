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
* In Auto mode, a lightweight classifier uses keywords, prompt features, and thresholds to choose local or cloud. Its routing score and confidence are heuristics, not calibrated probabilities.
* One-shot cloud override for user-controlled escalation.
* OpenAI streaming and buffered Ollama output with original formatting. Provider failures appear in chat with the selected model, and partial output stays visible.
* Native Streamlit `st.chat_message` interface + high-contrast custom theme (`.streamlit/config.toml`).
* Estimated cloud spend and savings versus the configured cloud model.
* Model provenance badges (Local · model / Cloud · model) inline with each answer.
* Local request share and separate input/output usage.
* Sanitized config: no embedded secrets; environment variable substitution pattern supported.

## 🧩 Architecture
```
User → Classifier → Router ─┬─ LocalModel (Ollama) 
                                           └─ CloudModel (OpenAI, extensible stubs for Anthropic/Gemini)
                               ↓
                         MetricsTracker (usage, spend, savings)
```

## 🔐 Security & Privacy
* API keys pulled from environment (e.g. `OPENAI_API_KEY`).
* Config allows `${VAR_NAME}` placeholders; do not commit secrets.
* In Auto mode, some keywords trigger cloud routing. Cloud requests send the current prompt and configured system prompt. Chat history stays in the Streamlit session, and the dashboard records usage metadata.

## Getting started

Use Python 3.11 and [Ollama](https://ollama.com/download). Run these commands from the repository root after cloning.

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

```shell
python -m pip install -r requirements.txt
ollama pull gemma2:2b
python -m streamlit run streamlit_app.py
```

Start Ollama before pulling the model. If needed, run `ollama serve` in a separate terminal. Leave the app in **Local Only** and send a simple prompt to verify the local setup. The response should show the local model. No cloud key is needed.

To check OpenAI, set `OPENAI_API_KEY` as an environment variable before launching Streamlit. Select **Send next prompt to cloud** and submit a synthetic prompt such as “Reply with OK.” Confirm the cloud model label and usage in the dashboard. This request may incur a charge.

## Automated checks

```shell
python -m unittest discover -s tests -v
python routing_benchmark.py --mock
```

GitHub Actions runs both commands on Python 3.11. The suite uses simulated providers, so it needs neither Ollama nor OpenAI credentials. Run the local and cloud checks above separately to verify the configured models.

## Routing benchmark

Run from the repository root after installing dependencies.

- `python routing_benchmark.py --mock` runs the smoke comparison without model calls.
- `python routing_benchmark.py --live --split holdout` runs the held-out prompts with Ollama and the configured cloud model. Live mode sends synthetic benchmark prompts to the cloud and can incur charges.

The starter set uses simple expected-content checks for quality. Review misses by case ID, and use the development split for tuning. The Local Only policy probe runs separately with mocked providers. Cost is estimated from word counts and configured pricing. Latency covers one non-streaming request per case and strategy.

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
cloud_budget_usd: null
system_prompt: >
   You are The Receptionist's Cloud Supervisor ...
```

## 📊 Metrics Logic
* Failed requests with reported usage still count toward spend. Partial replies without provider usage use estimates. Failures without usage or output are excluded from estimates.
* Cloud usage uses OpenAI token counts when available, including streaming usage. Other usage falls back to separate word-count estimates for input and output. Estimated cloud input includes the system prompt.
* Spend uses the configured rates. A model can use a flat `per_1k` rate or separate `input_per_1k` and `output_per_1k` rates. Missing pricing is shown as unavailable.
* The all-cloud baseline prices the recorded input and output counts at the configured cloud model's rates. It estimates routing savings for the same recorded workload, so it does not predict how another model would tokenize or answer a locally handled request.
* Set `cloud_budget_usd` to a nonnegative dollar amount to route new cloud-bound submissions to Ollama once the session's estimated spend reaches the limit. Cloud-bound submissions also stay local if pricing is missing and a budget is set. The budget is checked before each request, so an in-flight request can cross it. This is a per-session estimate and a separate limit from provider billing.

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

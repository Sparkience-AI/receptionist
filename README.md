# The Receptionist

![The Receptionist cover, showing a red waveform splitting into local and cloud routes](assets/receptionist-cover.jpg)

The Receptionist runs developer prompts through Ollama on your machine or sends them to a cloud model you choose. It starts in **Local Only**. Each reply shows which model handled the request, why it took that route, how long it took, and the estimated cloud cost.

## Run locally

You need Python 3.11 and [Ollama](https://ollama.com/download). Clone the repo and open a terminal in the checkout. Start Ollama first.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull gemma2:2b
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

**macOS or Linux**

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
ollama pull gemma2:2b
.venv/bin/python -m streamlit run streamlit_app.py
```

Open the URL Streamlit prints. Leave the sidebar set to **Local Only** and try “Explain what `git status` shows.” You should see an Ollama model label. Local use requires no API key.

## Choose where prompts go

| Mode | Behavior |
| --- | --- |
| **Local Only** | Runs prompts through Ollama. An Ollama failure stays local. |
| **Auto** | Uses prompt features, keywords, and configured thresholds to choose Ollama or the selected cloud model. |
| **Send next prompt to cloud** | Uses the selected cloud model once, subject to the session budget guard, then returns to the current mode. |

For cloud use, select OpenAI, Anthropic, or Gemini and enter a model ID and API key in the sidebar. Keys entered there last for the current Streamlit session. You can also set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY` in the environment before launching. Gemini also accepts `GEMINI_API_KEY` through the environment. An entered key overrides the configured credential for that provider.

Cloud requests send the current prompt and the configured system instructions. Earlier messages stay visible in the current session and are excluded from the request. OpenAI, Anthropic, and Gemini stream text when **Stream Responses** is on. Ollama returns a complete reply at once.

## Configure models and spending

See [config.yaml](config.yaml) for model defaults, pricing, and the optional `cloud_budget_usd` setting. Set `RECEPTIONIST_CONFIG` to load a separate config file. Use the provider/model key under `pricing`, such as `openai/gpt-4o-mini`. Use `google` for Gemini. Prices are in USD per thousand tokens, either a flat `per_1k` or separate `input_per_1k` and `output_per_1k` rates.

Cost estimates use provider-reported usage when available and word counts otherwise. The all-cloud baseline prices the recorded workload using the selected cloud model's rates. The optional session budget routes new cloud-bound requests locally once the recorded estimate reaches the limit or pricing is unavailable. A request already in flight can cross the limit.

## Test and benchmark

[Read the routing case study](docs/routing-case-study.md) for the live comparison of Auto, all-local, and all-cloud, with links to the run reports.

From the repository root, run these with the virtual environment's Python.

```sh
python -m unittest discover -s tests -v
python routing_benchmark.py --mock
```

The automated checks use mocked providers and run on Windows and Ubuntu in GitHub Actions. For a live preflight, run `python routing_benchmark.py --verify`. It checks the configured cloud model's reply and usage, then makes a local request.

```sh
python routing_benchmark.py --live --split holdout --report benchmark-results.json
```

The benchmark compares Auto, all-local, and all-cloud on synthetic prompts. Live runs send eligible prompts to the configured cloud provider and can incur charges. It reports keyword-based answer checks, cloud routes, latency, and estimated spend. The optional JSON report records per-case metrics without saving model replies or credentials. Use the development split for tuning and the holdout for evaluation. To verify one Auto policy, use `--strategy auto --score-threshold 0.16 --max-local-words 8` with the desired split. The flags affect only the benchmark run.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for bug reports and pull requests, [SECURITY.md](SECURITY.md) for private vulnerability reporting, and [LICENSE](LICENSE) for the MIT terms.

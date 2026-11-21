# Contributing

Use the [README](README.md) to set up Python 3.11. Keep the default path working without a cloud key.

## Report a bug

Include your OS and Python version, the routing mode, the provider and model, and the steps to reproduce. Replace real prompts with a synthetic example where possible. Remove credentials and private data from logs and screenshots. Report vulnerabilities through [SECURITY.md](SECURITY.md).

## Open a pull request

Explain the behavior change and include the tests you ran. For routing changes, check Local Only, Auto, and the one-shot cloud override. For cloud adapters, check usage reporting and how failures appear in chat. Use mocks for automated checks that would otherwise make model calls.

```sh
python -m pip check
python -m unittest discover -s tests -v
python routing_benchmark.py --mock
```

If you ran a live check, include the model and outcome. For benchmark changes, include the split and run mode so readers can distinguish a smoke check from measured model performance.

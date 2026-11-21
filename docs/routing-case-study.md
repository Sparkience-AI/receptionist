# When is a cloud model worth calling?

The Receptionist sends developer questions to either Ollama on the user's machine or a cloud model they choose. Local Only is the default. I built Auto to keep routine questions on the laptop and send harder ones to the cloud. In the first live run, Auto sent **one prompt out of 100** to the cloud and passed fewer answer checks than the all-local baseline.

## What I measured

I ran the same 100 synthetic developer prompts through all-local, all-cloud, and the original Auto rules. The local model was Ollama's `gemma2:2b`, running on the CPU of a Windows laptop with an i7-1165G7, 16 GB of RAM, and Iris Xe graphics. The cloud model was OpenAI's `gpt-6-luna`.

A reply passed when it contained all the expected terms or patterns for that prompt. That's a quick check of answer quality and can miss valid paraphrases or let bad advice through. Provider errors count as failed checks and their elapsed time counts toward latency. Cloud API cost is estimated from reported usage and excludes local electricity and hardware.

| Strategy | Checks passed | Provider errors | Cloud calls | Median | p95 | Est. cloud API cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-local | 60/100 | 19 | 0 | 23.46 s | 45.02 s | $0 |
| All-cloud | 93/100 | 0 | 100 | 1.46 s | 2.51 s | $0.004024 |
| Original Auto | 56/100 | 25 | 1 | 28.46 s | 45.03 s | $0.000112 |

### Where the checks failed

| Prompt category | All-local | All-cloud | Original Auto |
| --- | ---: | ---: | ---: |
| Git | 10/14 | 13/14 | 7/14 |
| Python | 15/28 | 25/28 | 16/28 |
| SQL | 11/17 | 16/17 | 10/17 |
| HTTP | 8/14 | 14/14 | 8/14 |
| Command line | 7/12 | 12/12 | 7/12 |
| Debugging | 4/8 | 7/8 | 3/8 |
| Testing | 3/4 | 4/4 | 2/4 |
| Other | 2/3 | 2/3 | 3/3 |

## Three prompts from the logs

| Prompt | Route and reason | What happened |
| --- | --- | --- |
| Which Git operation downloads changes without integrating them? | Original Auto → Gemma. Score below the cutoff. | Passed the `fetch` check in 12.14 s. |
| Why does `time.sleep(2)` in an asyncio handler stall unrelated requests? | Original Auto → Gemma. Score below the cutoff. | Provider error after 45.03 s. The all-cloud run passed the event loop and `asyncio.sleep` checks in 1.49 s. |
| Which SQL keyword removes duplicate result rows from `SELECT`? | Tuned Auto → Luna. Prompt exceeded the eight-word limit. | Passed the `DISTINCT` check in 1.22 s. Gemma also passed in the original Auto run, taking 24.34 s. |

## What I kept

I tuned the score cutoff and word limit on 75 development prompts. On a later live run of the 25 held-out prompts with Iris Xe enabled, tuned Auto passed **19/25**, compared with **11/25** for original Auto and **21/25** for all-cloud in the earlier run. Tuned Auto called the cloud 21 times and saved about 6% in estimated cloud API cost. Two local timeouts still put p95 at 45.02 s. The runs used different local compute setups, so the improvement can't be attributed to tuning alone. An offline learned candidate chose cloud for every holdout prompt.

I kept **Local Only** as the default, left **Auto** opt-in, and kept the tuned settings in the benchmark. On this laptop, even a correct local answer can take longer than a cloud call. I’d revisit Auto after testing a faster local setup, then judge the router on fresh prompts using answer checks, p95, and cloud spend together.

To repeat the comparison, set the models above in your config and run `python routing_benchmark.py --live --split all --report benchmark-results.json`. Live runs can incur API charges. [Benchmark runner](../routing_benchmark.py) · [Prompts and answer checks](../routing_benchmark_cases.json)

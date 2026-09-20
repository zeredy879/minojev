# Decision engine vs generative baseline

Workload: 40 requests, 120 decisions.

| metric | engine | generative |
|---|---:|---:|
| accuracy | 31.7% | 40.0% |
| teacher top-set | 31.7% | — |
| ECE | 0.197 | — |
| gold NLL | 1.309 | — |
| input tokens / decision | 245.700 | 100.800 |
| output tokens / decision | 0.000 | 2.517 |
| decode steps / decision | 0.000 | 2.517 |
| time to first token (p50) | 0 ms | 121.6 ms |
| latency per request p50 | 529.3 ms | 543.4 ms |
| latency per request p95 | 720.8 ms | 848.7 ms |
| decisions / second | 5.650 | 5.200 |
| format / parse failures | 0 | 0.0% |

| source | engine accuracy | generative accuracy |
|---|---:|---:|
| code-review | 33.3% | 38.1% |
| leads | 27.8% | 33.3% |
| moderation | 33.3% | 42.9% |
| refunds | 19.0% | 9.5% |
| support | 38.1% | 61.9% |
| triage | 38.9% | 55.6% |


# Decision engine vs generative baseline

Workload: 120 requests, 120 decisions.

| metric | engine | generative |
|---|---:|---:|
| accuracy | 95.8% | 80.0% |
| teacher top-set | 95.8% | — |
| ECE | 0.024 | — |
| gold NLL | 0.110 | — |
| input tokens / decision | 271.400 | 141.100 |
| output tokens / decision | 0.000 | 3.483 |
| decode steps / decision | 0.000 | 3.483 |
| time to first token (p50) | 0 ms | 132.2 ms |
| latency per request p50 | 500.3 ms | 549.1 ms |
| latency per request p95 | 1057.8 ms | 5319.9 ms |
| decisions / second | 1.760 | 3.500 |
| format / parse failures | 0 | 0.0% |

| source | engine accuracy | generative accuracy |
|---|---:|---:|
| clinc/clinc_oos | 100.0% | 96.7% |
| fancyzhx/amazon_polarity | 83.3% | 76.7% |
| mteb/banking77 | 100.0% | 96.7% |
| openai/gsm8k | 100.0% | 50.0% |


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
| time to first token (p50) | 0 ms | 74.7 ms |
| latency per request p50 | 264.3 ms | 361.4 ms |
| latency per request p95 | 582.6 ms | 3311.6 ms |
| decisions / second | 3.300 | 5.550 |
| format / parse failures | 0 | 0.0% |

| source | engine accuracy | generative accuracy |
|---|---:|---:|
| clinc/clinc_oos | 100.0% | 96.7% |
| fancyzhx/amazon_polarity | 83.3% | 76.7% |
| mteb/banking77 | 100.0% | 96.7% |
| openai/gsm8k | 100.0% | 50.0% |


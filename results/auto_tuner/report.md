# AutoTuner 结果

- 生成时间：2026-10-31T11:52:12+00:00
- 实测事实：`{"weights_gib": 28.75, "kv_per_token_kib": 32.0, "full_attention_layers": 16, "max_position_embeddings": 262144, "max_model_len": 32768, "kv_capacity_tokens_measured": 299132, "kv_capacity_source": "/root/logs/vllm/serve_accuracy_20261031_080915.log", "vram_gib": 47.99}`
- 评分权重：`{"alpha_throughput": 0.45, "beta_latency": 0.35, "gamma_quality": 0.15, "delta_memory": 0.05, "quality_threshold": 0.7, "latency_metric": "e2e_p95"}`

| 排名 | 候选 | 维度 | 输出 tok/s | TTFT P95 | TPOT P95 | E2E P95 | 峰值显存 | 质量 | 评分 | 状态 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | mtp_on | mtp | 91.79 | 11505 | 47.42 | 32114 | 42.28 | — | 0.6247 | 完成 |
| 2 | ctx16384_seq8 | context | 84.95 | 24878 | 78.47 | 64031 | 42.07 | — | 0.4173 | 完成 |
| 3 | ctx16384_seq16 | context | 84.86 | 24924 | 78.52 | 64125 | 42.09 | — | 0.4163 | 完成 |

# 实测结果快照

本目录由 `scripts/export_results.py` 自动导出，只包含能说明结论的小文件；
原始日志与逐条预测体积过大，未纳入仓库。

| 文件 | 大小 | 来源 |
|---|---:|---|

校验：`python3 -c "import json;print(json.load(open('results/manifest.json'))['count'])"`

每个 `summary.json` 的 `status` 为「完整」才计入正式成绩。

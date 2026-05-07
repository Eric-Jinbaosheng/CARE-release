# Limitations and Checklist Notes

- Main evidence is currently strongest in the SmolVLM2-2.2B pipeline; cross-backbone validation is still limited unless second-backbone runs complete.
- Improvements are benchmark-dependent; we do not claim universal gains.
- Optional CF improves only sparse text-grounded cases under strict gating; ungated CF can strongly harm.
- n=1000 subset evaluations may require full-set confirmation for final claims.
- Heuristic answer-space rules may need adaptation on other VLMs/tasks.
- Deterministic multi-view inference and optional CF scoring increase test-time compute relative to single-view base inference.

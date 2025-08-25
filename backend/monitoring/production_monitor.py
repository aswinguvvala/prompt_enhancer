"""
Lightweight production monitoring facade.
Uses in-memory counters to avoid external dependencies.
Integrate with Prometheus/Grafana later by swapping implementation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional


class ProductionMonitor:
    def __init__(self) -> None:
        self.logger = logging.getLogger("production_monitor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                "{\"timestamp\": %(asctime)s, \"level\": \"%(levelname)s\", \"message\": %(message)s}"
            )
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.counters = {"requests": 0, "errors": 0}

    async def monitor_enhancement(self, enhancement_func, prompt: str, target_model: str, **kwargs) -> Dict[str, Any]:
        self.counters["requests"] += 1
        start = time.time()
        try:
            result = await enhancement_func(prompt, target_model, **kwargs)
            latency = time.time() - start
            self.logger.info(json.dumps({
                "event": "enhancement_success",
                "model": target_model,
                "latency": latency,
                "prompt_length": len(prompt),
            }))
            return {"success": True, "result": result, "latency": latency}
        except Exception as e:  # pragma: no cover - defensive
            self.counters["errors"] += 1
            self.logger.error(json.dumps({
                "event": "enhancement_error",
                "model": target_model,
                "error": str(e),
                "prompt_length": len(prompt),
            }))
            return {"success": False, "error": str(e)}



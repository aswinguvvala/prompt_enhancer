"""
Local RAG-style knowledge base for prompt patterns.
Dependencies (faiss, sentence-transformers, redis) are optional. If missing,
the system gracefully degrades to in-memory lookup.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio


class PromptKnowledgeBase:
    def __init__(self) -> None:
        self._store: List[Dict[str, Any]] = []

    async def add_prompt_pattern(
        self,
        prompt: str,
        enhanced_prompt: str,
        metrics: Dict[str, Any],
        target_model: str,
        tags: Optional[List[str]] = None,
    ) -> None:
        self._store.append(
            {
                "original": prompt,
                "enhanced": enhanced_prompt,
                "metrics": metrics,
                "target_model": target_model,
                "tags": tags or [],
            }
        )

    async def find_similar_patterns(self, query_prompt: str, k: int = 3) -> List[Dict[str, Any]]:
        # Fallback: naive similarity by common token overlap
        def score(a: str, b: str) -> float:
            sa = set(a.lower().split())
            sb = set(b.lower().split())
            inter = len(sa & sb)
            union = max(1, len(sa | sb))
            return inter / union

        ranked = sorted(
            (
                {**item, "similarity": score(query_prompt, item["original"])}
                for item in self._store
            ),
            key=lambda x: x["similarity"],
            reverse=True,
        )
        return ranked[:k]

    async def get_enhancement_insights(self, prompt: str, target_model: str) -> Dict[str, Any]:
        sims = await self.find_similar_patterns(prompt)
        if not sims:
            return {"has_insights": False}
        common = []
        for s in sims:
            if "example" in s.get("enhanced", "").lower():
                common.append("Includes examples")
            if "steps" in s.get("enhanced", "").lower():
                common.append("Uses stepwise structure")
        return {
            "has_insights": True,
            "similar_prompts_found": len(sims),
            "common_enhancements": sorted(set(common)),
        }



"""
Advanced Multi-Stage Prompt Enhancement Pipeline (2025-ready)

This module demonstrates modern LLM orchestration skills:
- Multi-stage pipeline with clear stage contracts
- Parallel generation and synthesis
- Lightweight evaluation hooks and metrics
- Graceful degradation when optional dependencies are missing

Note: The pipeline reuses the existing PromptEnhancer for model calls to
avoid duplicating API integration logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


class EnhancementStrategy(Enum):
    CLARITY_FOCUSED = "clarity"
    CREATIVITY_FOCUSED = "creativity"
    TECHNICAL_PRECISION = "technical"
    CONTEXT_EXPANSION = "context"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT_LEARNING = "few_shot"
    CONSTITUTIONAL_AI = "constitutional"
    TREE_OF_THOUGHTS = "tree_of_thoughts"


@dataclass
class EnhancementMetrics:
    clarity_score: float = 0.0
    specificity_score: float = 0.0
    context_completeness: float = 0.0
    technical_accuracy: float = 0.0
    creativity_index: float = 0.0
    token_efficiency: float = 0.0
    estimated_performance_gain: float = 0.0
    confidence_score: float = 0.0
    reasoning_depth: float = 0.0
    factual_grounding: float = 0.0
    instruction_following: float = 0.0
    safety_alignment: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_metrics": {
                "clarity": self.clarity_score,
                "specificity": self.specificity_score,
                "context_completeness": self.context_completeness,
                "technical_accuracy": self.technical_accuracy,
                "creativity": self.creativity_index,
                "reasoning_depth": self.reasoning_depth,
                "factual_grounding": self.factual_grounding,
            },
            "efficiency_metrics": {
                "token_efficiency": self.token_efficiency,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "estimated_cost": self.estimated_cost,
                "processing_time_ms": self.processing_time_ms,
            },
            "performance_prediction": {
                "estimated_gain": self.estimated_performance_gain,
                "confidence": self.confidence_score,
            },
        }


@dataclass
class PipelineContext:
    original_prompt: str
    target_model: str
    strategy: EnhancementStrategy
    user_context: Dict[str, Any] = field(default_factory=dict)
    experiment_id: Optional[str] = None

    # evolving state
    analysis: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    final_prompt: str = ""
    stage_metrics: Dict[str, EnhancementMetrics] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)

    def add_error(self, stage: str, error: str) -> None:
        self.errors.append({"stage": stage, "error": error})

    def add_trace(self, stage: str, info: Dict[str, Any]) -> None:
        self.trace.append({"stage": stage, **info})

    def set_stage_metrics(self, stage: str, metrics: EnhancementMetrics) -> None:
        self.stage_metrics[stage] = metrics

    def get_last_stage_metrics(self) -> Dict[str, Any]:
        if not self.stage_metrics:
            return {}
        last_stage = list(self.stage_metrics.keys())[-1]
        return {last_stage: self.stage_metrics[last_stage].to_dict()}

    def get_trace(self) -> List[Dict[str, Any]]:
        return self.trace


class EnhancementStage:
    name: str = "base"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError


class ContextAnalysisStage(EnhancementStage):
    name = "context_analysis"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        prompt = context.original_prompt.strip()
        context.analysis = {
            "length": len(prompt),
            "words": len(prompt.split()),
            "has_examples": "example" in prompt.lower(),
            "has_role": any(k in prompt.lower() for k in ["you are", "act as", "as a"]),
        }
        context.add_trace(self.name, {"analysis": context.analysis})
        context.set_stage_metrics(self.name, EnhancementMetrics(clarity_score=0.5))
        return context


class ParallelEnhancementStage(EnhancementStage):
    name = "parallel_generation"

    def __init__(self, enhancer: Any, num_candidates: int = 3):
        self.enhancer = enhancer
        self.num_candidates = max(2, num_candidates)

    async def _gen(self, context_prompt: str, original_prompt: str, target_model: str, idx: int) -> Dict[str, Any]:
        # Increase temperature for diversity across candidates
        try:
            result = await self.enhancer.enhance_prompt(
                context_injection_prompt=context_prompt,
                original_prompt=original_prompt,
                target_model=target_model,
                temperature=min(1.0, 0.6 + 0.1 * idx),
                enhancement_type="comprehensive",
            )
            return {"text": result.enhanced_prompt, "backend": result.backend_used}
        except Exception as e:
            return {"text": original_prompt, "backend": f"error:{e}"}

    async def execute(self, context: PipelineContext) -> PipelineContext:
        # Lightweight context prompt
        from ..simplified_guides import SIMPLIFIED_MODEL_GUIDES  # type: ignore

        guide = SIMPLIFIED_MODEL_GUIDES.get(context.target_model, {})
        rules = "\n".join([f"- {r}" for r in guide.get("rules", [])][:5])

        base_context = (
            f"You improve prompts for {guide.get('name', context.target_model)}.\n"
            f"Key rules:\n{rules}\n\n"
            f"Original:\n\"{context.original_prompt}\"\n\n"
            f"Write only the improved prompt.\nENHANCED PROMPT:"
        )

        tasks = [
            self._gen(base_context, context.original_prompt, context.target_model, i)
            for i in range(self.num_candidates)
        ]
        candidates = await asyncio.gather(*tasks)
        context.candidates = candidates
        context.alternatives = [c["text"] for c in candidates]
        context.add_trace(self.name, {"num_candidates": len(candidates)})
        context.set_stage_metrics(self.name, EnhancementMetrics(creativity_index=0.6))
        return context


class SynthesisStage(EnhancementStage):
    name = "synthesis"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        # Simple heuristic: choose the longest candidate with structure
        best = ""
        best_score = -1
        for c in context.candidates:
            text = c["text"].strip()
            score = ("\n" in text) + (len(text) / 1000.0)
            if score > best_score:
                best = text
                best_score = score
        context.final_prompt = best or (context.candidates[0]["text"] if context.candidates else context.original_prompt)
        context.add_trace(self.name, {"selected_length": len(context.final_prompt)})
        context.set_stage_metrics(self.name, EnhancementMetrics(specificity_score=0.7))
        return context


class QualityAssuranceStage(EnhancementStage):
    name = "quality_assurance"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        # Primitive checks: ensure no XML/angle brackets, non-empty delta
        text = context.final_prompt.strip()
        if any(tag in text for tag in ["<", "></", "</"]):
            text = text.replace("<", "").replace(">", "")
        if len(text) < len(context.original_prompt):
            text = text + "\n\nPlease ensure clarity, examples, and stepwise structure."
        context.final_prompt = text
        context.add_trace(self.name, {"post_processed": True})
        context.set_stage_metrics(self.name, EnhancementMetrics(clarity_score=0.75))
        return context


class OptimizationStage(EnhancementStage):
    name = "optimization"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        # Add explicit success criteria
        if "Success criteria:" not in context.final_prompt:
            context.final_prompt += "\n\nSuccess criteria: clear steps, concrete examples, and explicit constraints."
        context.add_trace(self.name, {"optimized": True})
        context.set_stage_metrics(self.name, EnhancementMetrics(token_efficiency=0.5))
        return context


@dataclass
class EnhancementResult:
    original_prompt: str
    enhanced_prompt: str
    alternative_versions: List[str]
    metrics: Dict[str, Any]
    experiment_id: Optional[str]
    pipeline_trace: List[Dict[str, Any]]


class NoopExperimentTracker:
    def start_experiment(self, meta: Dict[str, Any]) -> str:
        return f"exp_{int(datetime.utcnow().timestamp())}"

    def log_stage(self, experiment_id: str, stage_name: str, metrics: Dict[str, Any]) -> None:
        return None

    def complete_experiment(self, experiment_id: str, result: EnhancementResult) -> None:
        return None


class AdvancedEnhancementPipeline:
    """High-level orchestration that composes all stages and evaluation."""

    def __init__(self, enhancer: Any, evaluator: Optional[Any] = None, experiment_tracker: Optional[Any] = None):
        self.enhancer = enhancer
        self.evaluator = evaluator
        self.experiment_tracker = experiment_tracker or NoopExperimentTracker()
        self.stages: List[EnhancementStage] = [
            ContextAnalysisStage(),
            ParallelEnhancementStage(self.enhancer),
            SynthesisStage(),
            QualityAssuranceStage(),
            OptimizationStage(),
        ]

    async def enhance(
        self,
        prompt: str,
        target_model: str,
        strategy: EnhancementStrategy = EnhancementStrategy.CLARITY_FOCUSED,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> EnhancementResult:
        experiment_id = self.experiment_tracker.start_experiment(
            {
                "prompt_len": len(prompt),
                "target_model": target_model,
                "strategy": strategy.value,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        context = PipelineContext(
            original_prompt=prompt,
            target_model=target_model,
            strategy=strategy,
            user_context=user_context or {},
            experiment_id=experiment_id,
        )

        for stage in self.stages:
            try:
                context = await stage.execute(context)
                self.experiment_tracker.log_stage(experiment_id, stage.name, context.get_last_stage_metrics())
            except Exception as e:  # pragma: no cover - defensive
                context.add_error(stage.name, str(e))
                continue

        metrics: Dict[str, Any] = {"overall_score": 0.0, "dimensional_scores": {}}
        if self.evaluator:
            try:
                result = await self.evaluator.evaluate_comprehensive(
                    original_prompt=prompt,
                    enhanced_prompt=context.final_prompt,
                    target_model=target_model,
                )
                # Normalize to dict
                metrics = {
                    "overall_score": getattr(result, "overall_score", 0.0),
                    "dimensional_scores": getattr(result, "dimensional_scores", {}),
                    "recommendations": getattr(result, "recommendations", []),
                    "confidence_intervals": getattr(result, "confidence_intervals", {}),
                    "comparison_to_baseline": getattr(result, "comparison_to_baseline", {}),
                }
            except Exception:
                # Evaluation failure should not break enhancement
                metrics = {"overall_score": 0.0, "dimensional_scores": {}}

        return EnhancementResult(
            original_prompt=prompt,
            enhanced_prompt=context.final_prompt or prompt,
            alternative_versions=context.alternatives,
            metrics=metrics,
            experiment_id=experiment_id,
            pipeline_trace=context.get_trace(),
        )



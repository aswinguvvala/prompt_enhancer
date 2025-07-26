"""
AI-Based Prompt Quality Evaluation System
Replaces outdated hardcoded metrics with intelligent AI assessment
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from models.prompt_enhancer import PromptEnhancer
from config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Quality dimensions for AI-based assessment."""
    CLARITY = "clarity"
    EFFECTIVENESS = "effectiveness"
    SPECIFICITY = "specificity"
    OPTIMIZATION = "optimization"
    TARGET_ALIGNMENT = "target_alignment"

@dataclass
class AIQualityMetrics:
    """AI-generated quality metrics."""
    overall_score: float
    improvement_score: float
    target_model_alignment: float
    optimization_quality: float
    clarity_enhancement: float
    ai_confidence: float
    detailed_feedback: str
    strengths: List[str]
    suggestions: List[str]

class AIPromptEvaluator:
    """AI-powered prompt quality evaluation system."""
    
    def __init__(self):
        self.prompt_enhancer = PromptEnhancer()
        self._evaluation_cache = {}
    
    def create_evaluation_prompt(self, 
                                original_prompt: str, 
                                enhanced_prompt: str, 
                                target_model: str,
                                methodology_used: str) -> str:
        """Create a comprehensive evaluation prompt for AI assessment."""
        
        evaluation_prompt = f"""You are an expert prompt engineering evaluator specializing in assessing prompt quality and optimization effectiveness.

EVALUATION TASK:
Analyze the quality of prompt enhancement for {target_model} optimization.

ORIGINAL PROMPT:
\"\"\"{original_prompt}\"\"\"

ENHANCED PROMPT:
\"\"\"{enhanced_prompt}\"\"\"

TARGET MODEL: {target_model}
METHODOLOGY APPLIED: {methodology_used}

EVALUATION CRITERIA:
1. OVERALL EFFECTIVENESS (0-100): How much better is the enhanced prompt?
2. TARGET ALIGNMENT (0-100): How well does it follow {target_model} best practices?
3. OPTIMIZATION QUALITY (0-100): Quality of applied optimization techniques
4. CLARITY ENHANCEMENT (0-100): Improvement in clarity and structure
5. AI CONFIDENCE (0-100): Your confidence in this assessment

DETAILED ANALYSIS REQUIRED:
- Identify specific improvements made
- Assess adherence to {target_model} methodology
- Evaluate potential effectiveness gains
- Note any missed optimization opportunities
- Provide actionable feedback

Please respond in this exact JSON format:
{{
    "overall_score": <0-100>,
    "improvement_score": <0-100>,
    "target_model_alignment": <0-100>,
    "optimization_quality": <0-100>,
    "clarity_enhancement": <0-100>,
    "ai_confidence": <0-100>,
    "detailed_feedback": "<comprehensive analysis>",
    "strengths": ["<strength1>", "<strength2>", "<strength3>"],
    "suggestions": ["<suggestion1>", "<suggestion2>", "<suggestion3>"]
}}

EVALUATION:"""

        return evaluation_prompt
    
    async def evaluate_enhancement(self,
                                 original_prompt: str,
                                 enhanced_prompt: str,
                                 target_model: str,
                                 methodology_used: str = "comprehensive_optimization") -> AIQualityMetrics:
        """Perform AI-based evaluation of prompt enhancement quality."""
        
        # Check cache first
        cache_key = self._generate_cache_key(original_prompt, enhanced_prompt, target_model)
        if cache_key in self._evaluation_cache:
            cached_result = self._evaluation_cache[cache_key]
            if time.time() - cached_result['timestamp'] < 3600:  # 1 hour cache
                logger.info("Using cached evaluation result")
                return cached_result['metrics']
        
        try:
            # Create evaluation prompt
            evaluation_prompt = self.create_evaluation_prompt(
                original_prompt, enhanced_prompt, target_model, methodology_used
            )
            
            # Get AI evaluation
            evaluation_result = await self.prompt_enhancer._call_ai_model(evaluation_prompt)
            
            # Parse the AI response
            metrics = self._parse_evaluation_response(evaluation_result, original_prompt, enhanced_prompt)
            
            # Cache the result
            self._evaluation_cache[cache_key] = {
                'metrics': metrics,
                'timestamp': time.time()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"AI evaluation failed: {str(e)}")
            # Return fallback metrics
            return self._generate_fallback_metrics(original_prompt, enhanced_prompt)
    
    def _parse_evaluation_response(self, 
                                 ai_response: str, 
                                 original_prompt: str, 
                                 enhanced_prompt: str) -> AIQualityMetrics:
        """Parse AI evaluation response into structured metrics."""
        
        try:
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                
                # Clean invalid control characters that cause JSON parsing errors
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control characters
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')  # Replace newlines
                json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
                
                eval_data = json.loads(json_str)
                
                return AIQualityMetrics(
                    overall_score=float(eval_data.get('overall_score', 75)) / 100,
                    improvement_score=float(eval_data.get('improvement_score', 70)) / 100,
                    target_model_alignment=float(eval_data.get('target_model_alignment', 80)) / 100,
                    optimization_quality=float(eval_data.get('optimization_quality', 75)) / 100,
                    clarity_enhancement=float(eval_data.get('clarity_enhancement', 70)) / 100,
                    ai_confidence=float(eval_data.get('ai_confidence', 85)) / 100,
                    detailed_feedback=eval_data.get('detailed_feedback', 'AI evaluation completed successfully.'),
                    strengths=eval_data.get('strengths', ['Enhanced structure', 'Better clarity', 'Improved specificity']),
                    suggestions=eval_data.get('suggestions', ['Consider more examples', 'Add context', 'Refine instructions'])
                )
            else:
                # Fallback: parse from text analysis
                return self._analyze_text_response(ai_response, original_prompt, enhanced_prompt)
                
        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {str(e)}")
            return self._generate_fallback_metrics(original_prompt, enhanced_prompt)
    
    def _analyze_text_response(self, 
                             ai_response: str, 
                             original_prompt: str, 
                             enhanced_prompt: str) -> AIQualityMetrics:
        """Analyze AI response text to extract quality insights."""
        
        response_lower = ai_response.lower()
        
        # Basic sentiment and keyword analysis
        positive_words = ['improved', 'better', 'enhanced', 'clear', 'effective', 'optimized', 'excellent', 'good']
        negative_words = ['poor', 'unclear', 'ineffective', 'missing', 'weak', 'bad', 'inadequate']
        
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        # Calculate scores based on analysis
        sentiment_score = max(0.5, min(1.0, (positive_count - negative_count + 5) / 10))
        
        # Length and structure analysis
        length_improvement = len(enhanced_prompt) / max(len(original_prompt), 1)
        length_score = min(1.0, max(0.3, length_improvement / 2))  # Reasonable enhancement
        
        # Extract strengths and suggestions from text
        strengths = self._extract_points(ai_response, ['strength', 'good', 'improved', 'better'])
        suggestions = self._extract_points(ai_response, ['suggest', 'recommend', 'could', 'should'])
        
        return AIQualityMetrics(
            overall_score=sentiment_score * 0.7 + length_score * 0.3,
            improvement_score=sentiment_score,
            target_model_alignment=0.75,  # Default assumption
            optimization_quality=length_score,
            clarity_enhancement=sentiment_score,
            ai_confidence=0.70,  # Lower confidence for text analysis
            detailed_feedback=ai_response[:500] + "..." if len(ai_response) > 500 else ai_response,
            strengths=strengths[:3] if strengths else ['Structure improved', 'Better formatting', 'Enhanced clarity'],
            suggestions=suggestions[:3] if suggestions else ['Add more context', 'Consider examples', 'Refine instructions']
        )
    
    def _extract_points(self, text: str, keywords: List[str]) -> List[str]:
        """Extract bullet points or sentences containing specific keywords."""
        
        lines = text.split('\n')
        points = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in keywords):
                # Clean up the line
                clean_line = line.replace('*', '').replace('-', '').strip()
                if len(clean_line) > 10 and len(clean_line) < 200:
                    points.append(clean_line)
        
        return points[:5]  # Limit to 5 points
    
    def _generate_fallback_metrics(self, 
                                 original_prompt: str, 
                                 enhanced_prompt: str) -> AIQualityMetrics:
        """Generate fallback metrics when AI evaluation fails."""
        
        # Basic heuristic analysis
        length_ratio = len(enhanced_prompt) / max(len(original_prompt), 1)
        
        # Assume reasonable improvement based on length and structure
        improvement_score = min(0.9, max(0.6, 0.5 + (length_ratio - 1) * 0.3))
        
        return AIQualityMetrics(
            overall_score=improvement_score,
            improvement_score=improvement_score,
            target_model_alignment=0.75,
            optimization_quality=0.70,
            clarity_enhancement=improvement_score,
            ai_confidence=0.50,  # Low confidence for fallback
            detailed_feedback="Fallback evaluation: Enhancement appears to improve prompt structure and length. AI evaluation unavailable.",
            strengths=["Increased detail", "Better structure", "Enhanced formatting"],
            suggestions=["Verify AI model availability", "Check enhancement quality manually", "Consider alternative evaluation methods"]
        )
    
    def _generate_cache_key(self, original: str, enhanced: str, target_model: str) -> str:
        """Generate cache key for evaluation results."""
        
        import hashlib
        key_data = f"{original[:100]}{enhanced[:100]}{target_model}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def compare_alternatives(self, 
                                 original_prompt: str,
                                 alternatives: List[str],
                                 target_model: str) -> List[Dict[str, Any]]:
        """Compare multiple alternative enhancements."""
        
        results = []
        
        for i, alternative in enumerate(alternatives):
            try:
                metrics = await self.evaluate_enhancement(
                    original_prompt, alternative, target_model, f"alternative_{i+1}"
                )
                
                results.append({
                    'alternative_id': i + 1,
                    'enhanced_prompt': alternative,
                    'metrics': metrics,
                    'rank_score': metrics.overall_score * 0.4 + metrics.target_model_alignment * 0.6
                })
                
            except Exception as e:
                logger.error(f"Failed to evaluate alternative {i+1}: {str(e)}")
                results.append({
                    'alternative_id': i + 1,
                    'enhanced_prompt': alternative,
                    'metrics': self._generate_fallback_metrics(original_prompt, alternative),
                    'rank_score': 0.6  # Default moderate score
                })
        
        # Sort by rank score
        results.sort(key=lambda x: x['rank_score'], reverse=True)
        
        return results
    
    def get_evaluation_summary(self, metrics: AIQualityMetrics) -> Dict[str, Any]:
        """Get a summary of evaluation metrics."""
        
        return {
            'overall_quality': 'Excellent' if metrics.overall_score > 0.9 else 
                             'Good' if metrics.overall_score > 0.75 else
                             'Fair' if metrics.overall_score > 0.6 else 'Poor',
            'score_percentage': round(metrics.overall_score * 100, 1),
            'improvement_percentage': round(metrics.improvement_score * 100, 1),
            'alignment_percentage': round(metrics.target_model_alignment * 100, 1),
            'confidence_level': 'High' if metrics.ai_confidence > 0.8 else
                              'Medium' if metrics.ai_confidence > 0.6 else 'Low',
            'top_strengths': metrics.strengths[:2],
            'key_suggestions': metrics.suggestions[:2],
            'detailed_available': len(metrics.detailed_feedback) > 50
        }
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        
        current_time = time.time()
        expired_keys = [
            key for key, data in self._evaluation_cache.items()
            if current_time - data['timestamp'] > 3600  # 1 hour expiry
        ]
        
        for key in expired_keys:
            del self._evaluation_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired evaluation cache entries")

# Legacy compatibility function
def evaluate_prompt_quality(prompt: str) -> Dict[str, float]:
    """Legacy function for backward compatibility."""
    
    logger.warning("Using legacy evaluate_prompt_quality - consider upgrading to AI-based evaluation")
    
    # Simple heuristic fallback
    word_count = len(prompt.split())
    char_count = len(prompt)
    
    # Basic scoring based on length and content
    specificity = min(1.0, word_count / 50)
    clarity = 0.8 if len(prompt) > 50 else 0.6
    completeness = min(1.0, char_count / 200)
    actionability = 0.7 if any(word in prompt.lower() for word in ['create', 'write', 'analyze']) else 0.5
    overall = (specificity + clarity + completeness + actionability) / 4
    
    return {
        'specificity': specificity,
        'clarity': clarity,
        'completeness': completeness,
        'actionability': actionability,
        'overall': overall
    }

# Global instance
ai_evaluator = AIPromptEvaluator()
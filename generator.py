import json
import logging
import hashlib
from openai import OpenAI
from typing import Dict

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Handles AI-powered question generation using OpenAI's GPT models."""
    
    # Simplified schema - reduces JSON parsing errors
    SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "question_generation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "qid": {"type": "string"},
                                "question": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "correct": {"type": "integer"},
                                "difficulty": {
                                    "type": "string",
                                    "enum": ["easy", "medium", "hard"]
                                },
                                "topic": {"type": "string"},
                                "subtopic": {"type": "string"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["qid", "question", "options", "correct", "difficulty", "topic", "subtopic", "tags"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["questions"],
                "additionalProperties": False
            }
        }
    }
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        """Initialize OpenAI client."""
        logger.info(f"Initializing OpenAI - Model: {model_name}")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self._generation_count = 0
    
    def generate(self, seed_json: str, count: int = 5) -> str:
        """Generate questions based on seed data."""
        seed_data = json.loads(seed_json)
        topic = seed_data.get('topic', 'Fill in the Blanks')
        subtopic = seed_data.get('subtopic', '')
        exam_slug = seed_data.get('examSlug', 'exam')
        
        # Increment generation counter for variety
        self._generation_count += 1
        variety_seed = hashlib.md5(f"{topic}{self._generation_count}".encode()).hexdigest()[:8]
        
        # Build context-aware prompt
        prompt = self._build_prompt(topic, subtopic, exam_slug, count, variety_seed)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert question creator for competitive government exams in India. Generate high-quality, diverse multiple-choice questions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format=self.SCHEMA,
                temperature=0.8,
                max_tokens=4000
            )
            
            # Extract questions
            content = json.loads(response.choices[0].message.content)
            questions = content.get('questions', [])
            
            return json.dumps(questions)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _build_prompt(self, topic: str, subtopic: str, exam_slug: str, count: int, variety_seed: str) -> str:
        """Build generation prompt."""
        contexts = self._get_exam_contexts(exam_slug)
        
        return f"""Create {count} unique multiple-choice questions for Indian government exams.

TOPIC: {topic}
{f"SUBTOPIC: {subtopic}" if subtopic else ""}
VARIETY SEED: {variety_seed}

REQUIREMENTS:
1. Each question must be UNIQUE with different vocabulary and contexts
2. Use diverse contexts: {', '.join(contexts[:5])}
3. Mix different concepts across questions
4. For fill-in-blank questions, use '____' to mark the blank
5. Provide exactly 4 options
6. Only ONE option should be correct

DIFFICULTY DISTRIBUTION:
- Easy (30%): Simple vocabulary, basic concepts
- Medium (50%): Moderate complexity
- Hard (20%): Advanced vocabulary, complex concepts

OUTPUT FORMAT:
- qid: "GEN_1", "GEN_2", etc.
- question: Full question text
- options: Array of 4 options (strings)
- correct: Index of correct answer (0-3)
- difficulty: "easy", "medium", or "hard"
- topic: "{topic}"
- subtopic: "{subtopic if subtopic else topic}"
- tags: Array of relevant tags

IMPORTANT: Avoid apostrophes and quotes in question text where possible. Use simple punctuation.

Generate {count} questions now."""
    
    def _get_exam_contexts(self, exam_slug: str) -> list:
        """Return context suggestions based on exam type."""
        context_map = {
            'ssc-cgl': ['government', 'administration', 'clerical', 'finance'],
            'ibps': ['banking', 'finance', 'customer service', 'loans'],
            'sbi': ['banking', 'finance', 'customer service', 'loans'],
            'upsc': ['civil services', 'policy', 'governance', 'relations'],
            'nda': ['military', 'defense', 'security', 'leadership'],
            'railway': ['transport', 'logistics', 'safety', 'operations'],
            'rrb': ['railway', 'transport', 'logistics', 'safety'],
        }
        
        for key, contexts in context_map.items():
            if key in exam_slug.lower():
                return contexts
        
        return ['business', 'education', 'technology', 'social', 'government', 'science']
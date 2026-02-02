import json
import logging
from google import genai
from google.genai import types
from typing import Dict

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Handles AI-powered question generation using Google's Gemini model."""
    
    # Class-level schema (defined once, reused for all instances)
    _SCHEMA = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "qid": {"type": "STRING"},
                "question": {"type": "STRING"},
                "options": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "minItems": 4,
                    "maxItems": 4
                },
                "correct": {"type": "INTEGER"},
                "difficulty": {"type": "STRING"},
                "topic": {"type": "STRING"},
                "subtopic": {"type": "STRING"}
            },
            "required": ["qid", "question", "options", "correct"]
        }
    }
    
    # Class-level blueprint (constant)
    _BLUEPRINT = (
        "FORMAT: Fill in the Blanks. The 'question' string must contain a '____' gap. "
        "DO NOT use (a)/(b)/(c) segments. "
        "The 'options' array must contain 4 single words or short phrases that could potentially fill the gap. "
        "QUALITY: Focus on testing subject-verb agreement, phrasal verbs, prepositions, or context-heavy vocabulary. "
        "The sentence must provide enough context for only ONE option to be correct."
    )
    
    def __init__(self, project_id: str, location: str, model_name: str):
        """Initialize Vertex AI client with Google Gemini."""
        logger.info(f"Initializing Vertex AI - Project: {project_id}, Location: {location}, Model: {model_name}")
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.model_name = model_name
    
    def generate(self, seed_json: str, count: int = 5) -> str:
        """
        Generate questions based on seed data.
        
        Args:
            seed_json: JSON string containing seed question data
            count: Number of questions to generate (default: 5)
            
        Returns:
            JSON string containing generated questions
        """
        seed_data = json.loads(seed_json)
        topic = seed_data.get('topic', 'Fill in the Blanks')
        
        prompt = f"""Act as a Senior Exam Setter for SSC-CGL and IBPS.

CRITICAL: Create UNIQUE questions. DO NOT repeat patterns.
Each question MUST be distinctly different in:
- Vocabulary used
- Sentence structure
- Context/scenario

Create {count} HIGH-QUALITY 'Fill in the Blank' questions.

TOPIC: {topic}

STRICT REQUIREMENTS:
1. QUESTION FORMAT: Use '____' to represent the blank
2. OPTIONS: Provide exactly 4 options
3. UNIQUENESS: Each question must test a DIFFERENT concept
4. VARIETY: Mix verb tenses, contexts, and vocabulary

Examples of VARIETY:
- Question 1: Past tense in business context
- Question 2: Present perfect in education context  
- Question 3: Future tense in government context
- Question 4: Modal verbs in social context
"""        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self._SCHEMA
            )
        )
        return response.text
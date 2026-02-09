import json
import logging
import hashlib
from openai import OpenAI
from typing import Dict

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Handles AI-powered question generation using OpenAI's GPT models."""
    
    # JSON schema for structured output
    _SCHEMA = {
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
                                    "items": {"type": "string"},
                                    "minItems": 4,
                                    "maxItems": 4
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
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model_name: Model to use (default: gpt-4o, can use gpt-4o-mini for cheaper)
        """
        logger.info(f"Initializing OpenAI - Model: {model_name}")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self._generation_count = 0
    
    def generate(self, seed_json: str, count: int = 5) -> str:
        """
        Generate questions based on seed data with enhanced diversity.
        
        Args:
            seed_json: JSON string containing seed question data
            count: Number of questions to generate (default: 5)
            
        Returns:
            JSON string containing generated questions (as array)
        """
        seed_data = json.loads(seed_json)
        topic = seed_data.get('topic', 'Fill in the Blanks')
        subtopic = seed_data.get('subtopic', '')
        exam_slug = seed_data.get('examSlug', 'exam')
        
        # Increment generation counter for variety
        self._generation_count += 1
        variety_seed = hashlib.md5(f"{topic}{self._generation_count}".encode()).hexdigest()[:8]
        
        # Build context-aware prompt
        prompt = self._build_enhanced_prompt(topic, subtopic, exam_slug, count, variety_seed)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert question creator for competitive government exams in India. Generate high-quality, diverse questions in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format=self._SCHEMA,
                temperature=1.2,
                max_tokens=4000
            )
            
            # Extract the questions array from the response
            content = json.loads(response.choices[0].message.content)
            questions = content.get('questions', [])
            
            # Return as JSON array (matching the old format)
            return json.dumps(questions)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _build_enhanced_prompt(self, topic: str, subtopic: str, exam_slug: str, count: int, variety_seed: str) -> str:
        """Build a context-rich prompt that encourages diversity."""
        
        contexts = self._get_exam_contexts(exam_slug)
        
        prompt = f"""Create {count} COMPLETELY UNIQUE "Fill in the Blank" questions for competitive exams.

📚 TOPIC: {topic}
{f"📌 SUBTOPIC: {subtopic}" if subtopic else ""}
🔑 VARIETY SEED: {variety_seed}

⚠️ CRITICAL UNIQUENESS REQUIREMENTS:
1. Each question MUST use DIFFERENT vocabulary, contexts, and sentence structures
2. NO REPETITIVE PATTERNS - vary the subject matter completely
3. Use diverse contexts: {', '.join(contexts[:5])}
4. Mix different grammatical concepts across questions
5. Ensure questions test DIFFERENT skills (vocabulary, grammar, idioms, contextual meaning)

✅ QUESTION FORMAT RULES:
- Use '____' to mark the blank (exactly 4 underscores)
- Provide exactly 4 options as array of strings
- Only ONE option should be correct
- Options should be plausible but only one fits the context
- Question must have enough context to determine the correct answer

🎨 DIVERSITY STRATEGIES:

**Vocabulary Variety:**
- Questions 1-2: Business/Corporate vocabulary
- Questions 3-4: Social/Cultural contexts
- Questions 5-6: Scientific/Technical terms
- Questions 7-8: Historical/Political references

**Grammar Patterns:**
- Mix: Present, Past, Future, Perfect tenses
- Include: Active and Passive voice
- Vary: Simple, Compound, Complex sentences
- Use: Prepositions, phrasal verbs, idioms

**Context Types:**
- Professional scenarios (workplace, business meetings)
- Academic situations (classroom, research, exams)
- Social events (celebrations, conversations)
- News/Current affairs (politics, economy, environment)
- Personal life (family, relationships, daily activities)
- Technology and innovation
- Nature and Environment

**Difficulty Distribution:**
- Easy (30%): Common vocabulary, simple grammar
- Medium (50%): Moderate vocabulary, compound sentences
- Hard (20%): Advanced vocabulary, complex grammar

REQUIRED OUTPUT:
- qid: string (use "GEN_1", "GEN_2", etc.)
- question: string (with '____' for blank)
- options: array of exactly 4 strings
- correct: integer (0-3, index of correct answer)
- difficulty: string (must be "easy", "medium", or "hard")
- topic: string (use "{topic}")
- subtopic: string (use "{subtopic if subtopic else topic}")
- tags: array of relevant tags (e.g., ["prepositions", "business-context"])

🚫 AVOID:
- Repeating the same vocabulary
- Similar sentence patterns
- Generic or boring contexts
- Ambiguous questions where multiple options could fit

Generate {count} highly diverse, unique questions now."""

        return prompt
    
    def _get_exam_contexts(self, exam_slug: str) -> list:
        """Return context suggestions based on exam type."""
        
        context_map = {
            'ssc-cgl': ['government services', 'public administration', 'clerical work', 'financial management'],
            'ibps': ['banking operations', 'financial markets', 'customer service', 'loan processing'],
            'sbi': ['banking operations', 'financial markets', 'customer service', 'loan processing'],
            'upsc': ['civil services', 'international relations', 'public policy', 'governance'],
            'nda': ['military operations', 'defense strategy', 'national security', 'leadership'],
            'railway': ['transportation', 'logistics', 'safety protocols', 'technical operations'],
            'rrb': ['railway operations', 'transportation', 'logistics', 'safety'],
            'jee': ['physics concepts', 'chemistry reactions', 'mathematics problems', 'engineering'],
            'neet': ['medical science', 'biology concepts', 'health care', 'anatomy'],
            'cat': ['business management', 'analytical reasoning', 'data interpretation']
        }
        
        for key, contexts in context_map.items():
            if key in exam_slug.lower():
                return contexts
        
        return [
            'business and commerce', 'education and learning', 'technology and innovation',
            'social interactions', 'government and politics', 'science and research',
            'arts and culture', 'sports and fitness', 'environment and nature',
            'health and wellness', 'travel and tourism', 'media and communication'
        ]
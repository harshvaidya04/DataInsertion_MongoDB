import json
import logging
import hashlib
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
                "subtopic": {"type": "STRING"},
                "tags": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                }
            },
            "required": ["qid", "question", "options", "correct"]
        }
    }
    
    def __init__(self, project_id: str, location: str, model_name: str):
        """Initialize Vertex AI client with Google Gemini."""
        logger.info(f"Initializing Vertex AI - Project: {project_id}, Location: {location}, Model: {model_name}")
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.model_name = model_name
        self._generation_count = 0
    
    def generate(self, seed_json: str, count: int = 5) -> str:
        """
        Generate questions based on seed data with enhanced diversity.
        
        Args:
            seed_json: JSON string containing seed question data
            count: Number of questions to generate (default: 5)
            
        Returns:
            JSON string containing generated questions
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
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self._SCHEMA,
                temperature=1.2,  # Increased for more creativity
                top_p=0.95,
                top_k=40
            )
        )
        return response.text
    
    def _build_enhanced_prompt(self, topic: str, subtopic: str, exam_slug: str, count: int, variety_seed: str) -> str:
        """Build a context-rich prompt that encourages diversity."""
        
        # Define diverse contexts for different exam types
        contexts = self._get_exam_contexts(exam_slug)
        
        prompt = f"""You are an Expert Question Creator for competitive exams like SSC-CGL, IBPS, NDA, and other government exams.

🎯 MISSION: Create {count} COMPLETELY UNIQUE "Fill in the Blank" questions.

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

🎨 DIVERSITY STRATEGIES (Apply these across your {count} questions):

**Vocabulary Variety:**
- Question 1-2: Business/Corporate vocabulary
- Question 3-4: Social/Cultural contexts
- Question 5-6: Scientific/Technical terms
- Question 7-8: Historical/Political references
- Question 9-10: Literary/Artistic expressions

**Grammar Patterns:**
- Mix: Present, Past, Future, Perfect tenses
- Include: Active and Passive voice
- Vary: Simple, Compound, Complex sentences
- Add: Conditionals, Subjunctive mood
- Use: Different prepositions, phrasal verbs, idioms

**Context Types:**
- Professional scenarios (workplace, business meetings)
- Academic situations (classroom, research, exams)
- Social events (celebrations, gatherings, conversations)
- News/Current affairs (politics, economy, environment)
- Personal life (family, relationships, daily activities)
- Technology (innovation, digital world)
- Nature/Environment (wildlife, climate, geography)

**Sentence Structures:**
- Declarative statements
- Questions converted to statements
- Conditional sentences (if-then)
- Cause-effect relationships
- Comparisons and contrasts

OUTPUT FORMAT:
Return a JSON array of {count} question objects with these fields:
- qid: string (use "GEN_<number>")
- question: string (with '____' for blank)
- options: array of 4 strings
- correct: integer (0-3, index of correct answer)
- difficulty: string ("easy", "medium", "hard")
- topic: string (use "{topic}")
- subtopic: string (use "{subtopic if subtopic else topic}")
- tags: array of relevant tags (grammar concepts, themes)

🚫 AVOID:
- Repeating the same vocabulary across questions
- Using similar sentence patterns
- Creating questions that all test the same concept
- Generic or boring contexts
- Ambiguous questions where multiple options could fit

💡 EXAMPLE OF GOOD VARIETY:
BAD: All questions about "running" or "walking"
GOOD: Question 1 about business negotiations, Q2 about scientific research, Q3 about historical events, Q4 about artistic expression

NOW CREATE {count} HIGHLY DIVERSE, UNIQUE QUESTIONS:"""

        return prompt
    
    def _get_exam_contexts(self, exam_slug: str) -> list:
        """Return context suggestions based on exam type."""
        
        context_map = {
            'ssc-cgl': ['government services', 'public administration', 'clerical work', 'financial management', 'law enforcement'],
            'ibps': ['banking operations', 'financial markets', 'customer service', 'loan processing', 'risk management'],
            'upsc': ['civil services', 'international relations', 'public policy', 'governance', 'constitutional matters'],
            'nda': ['military operations', 'defense strategy', 'national security', 'discipline', 'leadership'],
            'railway': ['transportation', 'logistics', 'safety protocols', 'technical operations', 'public service'],
            'jee': ['physics concepts', 'chemistry reactions', 'mathematics problems', 'engineering principles', 'scientific methods'],
            'neet': ['medical science', 'biology concepts', 'health care', 'anatomy', 'disease management'],
            'cat': ['business management', 'analytical reasoning', 'data interpretation', 'logical thinking', 'quantitative skills']
        }
        
        # Check if exam_slug contains any key
        for key, contexts in context_map.items():
            if key in exam_slug.lower():
                return contexts
        
        # Default diverse contexts
        return [
            'business and commerce', 'education and learning', 'technology and innovation',
            'social interactions', 'government and politics', 'science and research',
            'arts and culture', 'sports and fitness', 'environment and nature',
            'health and wellness', 'travel and tourism', 'media and communication'
        ]
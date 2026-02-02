import json
import logging
import hashlib
from google import genai
from google.genai import types
from typing import Dict, List

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Handles AI-powered question generation for multiple question formats."""
    
    # Define schemas for different question types
    _SCHEMAS = {
        'fill_in_blanks': {
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
                    "tags": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["qid", "question", "options", "correct"]
            }
        },
        'error_correction': {
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
                    "tags": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["qid", "question", "options", "correct"]
            }
        },
        'sentence_arrangement': {
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
                    "tags": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["qid", "question", "options", "correct"]
            }
        },
        'sentence_improvement': {
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
                    "tags": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["qid", "question", "options", "correct"]
            }
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
        Generate questions based on seed data with automatic format detection.
        
        Args:
            seed_json: JSON string containing seed question data
            count: Number of questions to generate (default: 5)
            
        Returns:
            JSON string containing generated questions
        """
        seed_data = json.loads(seed_json)
        topic = seed_data.get('topic', '')
        subtopic = seed_data.get('subtopic', '')
        exam_slug = seed_data.get('examSlug', 'exam')
        
        # Detect question format from seed
        question_format = self._detect_format(seed_data)
        logger.info(f"🎯 Detected format: {question_format}")
        
        # Increment generation counter for variety
        self._generation_count += 1
        variety_seed = hashlib.md5(f"{topic}{self._generation_count}".encode()).hexdigest()[:8]
        
        # Build format-specific prompt
        prompt = self._build_prompt(question_format, topic, subtopic, exam_slug, count, variety_seed, seed_data)
        
        # Get appropriate schema
        schema = self._SCHEMAS.get(question_format, self._SCHEMAS['fill_in_blanks'])
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=1.2,
                top_p=0.95,
                top_k=40
            )
        )
        return response.text
    
    def _detect_format(self, seed_data: Dict) -> str:
        """Detect question format from seed data."""
        question = seed_data.get('question', '').lower()
        topic = seed_data.get('topic', '').lower()
        subtopic = seed_data.get('subtopic', '').lower()
        
        # Check for error correction
        if ('error' in topic or 'error' in subtopic or 
            'find the error' in question or 'incorrect' in question or
            'grammatical error' in question):
            return 'error_correction'
        
        # Check for sentence arrangement/ordering
        if ('arrange' in topic or 'order' in topic or 'jumbled' in topic or
            'arrange' in question or 'proper order' in question or
            'sequence' in question):
            return 'sentence_arrangement'
        
        # Check for sentence improvement
        if ('improvement' in topic or 'improve' in topic or
            'better' in question or 'best expresses' in question):
            return 'sentence_improvement'
        
        # Check for fill in blanks
        if '____' in question or 'blank' in topic or 'fill' in topic:
            return 'fill_in_blanks'
        
        # Default to fill in blanks
        return 'fill_in_blanks'
    
    def _build_prompt(self, format_type: str, topic: str, subtopic: str, 
                      exam_slug: str, count: int, variety_seed: str, seed_data: Dict) -> str:
        """Build format-specific prompt."""
        
        if format_type == 'error_correction':
            return self._build_error_correction_prompt(topic, subtopic, exam_slug, count, variety_seed)
        elif format_type == 'sentence_arrangement':
            return self._build_sentence_arrangement_prompt(topic, subtopic, exam_slug, count, variety_seed)
        elif format_type == 'sentence_improvement':
            return self._build_sentence_improvement_prompt(topic, subtopic, exam_slug, count, variety_seed)
        else:
            return self._build_fill_blanks_prompt(topic, subtopic, exam_slug, count, variety_seed)
    
    def _build_fill_blanks_prompt(self, topic: str, subtopic: str, exam_slug: str, 
                                   count: int, variety_seed: str) -> str:
        """Build prompt for Fill in the Blanks questions."""
        contexts = self._get_exam_contexts(exam_slug)
        
        return f"""You are an Expert Question Creator for competitive exams like SSC-CGL, IBPS, NDA, and other government exams.

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

🎨 DIVERSITY STRATEGIES:
- Mix: Present, Past, Future, Perfect tenses
- Include: Active and Passive voice
- Use: Prepositions, phrasal verbs, idioms, collocations
- Vary: Simple, Compound, Complex sentences
- Add: Business, Academic, Social, Political contexts

📋 OUTPUT FORMAT: Return JSON array with: qid, question (with '____'), options (4 strings), correct (0-3), difficulty, topic, subtopic, tags

NOW CREATE {count} HIGHLY DIVERSE, UNIQUE QUESTIONS:"""
    
    def _build_error_correction_prompt(self, topic: str, subtopic: str, exam_slug: str, 
                                        count: int, variety_seed: str) -> str:
        """Build prompt for Error Correction questions."""
        contexts = self._get_exam_contexts(exam_slug)
        
        return f"""You are an Expert Question Creator for competitive exams like SSC-CGL, IBPS, NDA, and other government exams.

🎯 MISSION: Create {count} COMPLETELY UNIQUE "Spot the Error" questions.

📚 TOPIC: {topic}
{f"📌 SUBTOPIC: {subtopic}" if subtopic else ""}
🔑 VARIETY SEED: {variety_seed}

⚠️ CRITICAL UNIQUENESS REQUIREMENTS:
1. Each question MUST test DIFFERENT grammatical errors
2. NO REPETITIVE PATTERNS - vary error types completely
3. Use diverse contexts: {', '.join(contexts[:5])}
4. Mix different error categories across questions

✅ QUESTION FORMAT RULES:
- Present a sentence divided into 3 parts: (A), (B), (C)
- ONE part contains a grammatical error OR all parts are correct
- Provide 4 options: "Error in part A", "Error in part B", "Error in part C", "No error"
- The 'question' field should contain the full sentence with parts marked clearly
- Example: "Despite of (A) / the heavy rain, (B) / the match continued. (C)"

🎯 ERROR TYPES TO VARY:
1. Subject-Verb Agreement (He don't go → He doesn't go)
2. Tense Errors (I am knowing → I know)
3. Preposition Errors (Good in → Good at)
4. Article Errors (An university → A university)
5. Pronoun Errors (Me and him → He and I)
6. Word Choice (Despite of → Despite)
7. Adjective-Adverb confusion (He runs quick → quickly)
8. Comparative/Superlative errors (more better → better)
9. Double negatives (can't hardly → can hardly)
10. Redundancy (past history → history)

🎨 DIVERSITY STRATEGIES:
- Question 1-2: Subject-verb agreement in different tenses
- Question 3-4: Preposition errors in various contexts
- Question 5-6: Article usage errors
- Question 7-8: Tense consistency errors
- Question 9-10: Pronoun and word choice errors
- Include 2-3 "No error" questions to test careful reading

📋 OUTPUT FORMAT:
{{
  "qid": "GEN_1",
  "question": "The committee have decided (A) / to postpone the meeting (B) / until next week. (C)",
  "options": ["Error in part A", "Error in part B", "Error in part C", "No error"],
  "correct": 0,
  "difficulty": "medium",
  "topic": "{topic}",
  "subtopic": "{subtopic if subtopic else topic}",
  "tags": ["subject-verb-agreement", "plural-noun"]
}}

NOW CREATE {count} HIGHLY DIVERSE ERROR CORRECTION QUESTIONS:"""
    
    def _build_sentence_arrangement_prompt(self, topic: str, subtopic: str, exam_slug: str, 
                                            count: int, variety_seed: str) -> str:
        """Build prompt for Sentence Arrangement questions."""
        contexts = self._get_exam_contexts(exam_slug)
        
        return f"""You are an Expert Question Creator for competitive exams like SSC-CGL, IBPS, NDA, and other government exams.

🎯 MISSION: Create {count} COMPLETELY UNIQUE "Sentence Arrangement" questions.

📚 TOPIC: {topic}
{f"📌 SUBTOPIC: {subtopic}" if subtopic else ""}
🔑 VARIETY SEED: {variety_seed}

⚠️ CRITICAL UNIQUENESS REQUIREMENTS:
1. Each question MUST have DIFFERENT topic/context
2. Use diverse contexts: {', '.join(contexts[:5])}
3. Vary the paragraph types (narrative, descriptive, argumentative, expository)
4. Mix different logical patterns (chronological, cause-effect, problem-solution)

✅ QUESTION FORMAT RULES:
- Present 4-5 jumbled sentences labeled P, Q, R, S (and T if 5 sentences)
- One sentence is already fixed as the first sentence
- 'question' field contains: "Rearrange the following sentences to form a coherent paragraph. Sentence 1 is fixed."
- Then list: "1. [Fixed first sentence]", "P. [sentence]", "Q. [sentence]", "R. [sentence]", "S. [sentence]"
- Provide 4 options with different arrangements (e.g., "PQRS", "QRPS", "RPSQ", "SPRQ")
- Only ONE arrangement should create a logical, coherent paragraph

🎯 LOGICAL PATTERNS TO USE:
1. Chronological (First... then... finally)
2. Cause and Effect (Because... therefore)
3. Problem-Solution (Issue... however... thus)
4. General to Specific (Overall... for example... specifically)
5. Comparison-Contrast (While... on the other hand)
6. Definition-Example (X is... for instance)

🎨 CONTEXT VARIETY:
- Question 1: Historical narrative
- Question 2: Scientific explanation
- Question 3: Social issue analysis
- Question 4: Business case study
- Question 5: Environmental discussion
- Question 6: Technology development
- Question 7: Cultural description
- Question 8: Economic trend

📋 OUTPUT FORMAT EXAMPLE:
{{
  "qid": "GEN_1",
  "question": "Rearrange the following sentences to form a coherent paragraph. Sentence 1 is fixed.\\n\\n1. Climate change is one of the most pressing issues of our time.\\nP. Rising sea levels threaten coastal communities worldwide.\\nQ. Scientists have documented a steady increase in global temperatures.\\nR. This has led to more frequent and severe weather events.\\nS. Immediate action is needed to mitigate these effects.",
  "options": ["QRPS", "PQRS", "RPSQ", "QPRS"],
  "correct": 0,
  "difficulty": "medium",
  "topic": "{topic}",
  "subtopic": "{subtopic if subtopic else topic}",
  "tags": ["logical-sequence", "coherence"]
}}

CRITICAL: Ensure only ONE option creates perfect logical flow. Other options should break coherence.

NOW CREATE {count} HIGHLY DIVERSE SENTENCE ARRANGEMENT QUESTIONS:"""
    
    def _build_sentence_improvement_prompt(self, topic: str, subtopic: str, exam_slug: str, 
                                            count: int, variety_seed: str) -> str:
        """Build prompt for Sentence Improvement questions."""
        contexts = self._get_exam_contexts(exam_slug)
        
        return f"""You are an Expert Question Creator for competitive exams like SSC-CGL, IBPS, NDA, and other government exams.

🎯 MISSION: Create {count} COMPLETELY UNIQUE "Sentence Improvement" questions.

📚 TOPIC: {topic}
{f"📌 SUBTOPIC: {subtopic}" if subtopic else ""}
🔑 VARIETY SEED: {variety_seed}

⚠️ CRITICAL UNIQUENESS REQUIREMENTS:
1. Each question MUST test DIFFERENT improvement aspects
2. Use diverse contexts: {', '.join(contexts[:5])}
3. Vary the type of improvement needed
4. Mix different grammatical and stylistic issues

✅ QUESTION FORMAT RULES:
- Present a sentence with an underlined portion that may need improvement
- 'question' field format: "Select the best way to improve the underlined portion: The company decided to **implement the new policy** immediately."
- Use **double asterisks** to mark the underlined portion
- Provide 4 options: 3 alternatives + 1 "No improvement needed"
- Only ONE option should be the best improvement (or "No improvement needed" if original is best)

🎯 IMPROVEMENT TYPES TO VARY:
1. Word Choice (utilize → use, purchase → buy)
2. Conciseness (at this point in time → now)
3. Clarity (ambiguous → precise)
4. Grammar correction (less people → fewer people)
5. Tone adjustment (formal ↔ informal appropriate)
6. Redundancy removal (past history → history)
7. Parallel structure (running, to swim → running, swimming)
8. Idiomatic expression (make a decision → decide)
9. Active vs Passive voice
10. Sentence structure improvement

🎨 DIVERSITY STRATEGIES:
- Question 1-2: Wordiness/conciseness
- Question 3-4: Word choice precision
- Question 5-6: Grammar corrections
- Question 7-8: Parallel structure
- Question 9-10: Tone/register appropriateness
- Include 2-3 "No improvement needed" to test judgment

📋 OUTPUT FORMAT EXAMPLE:
{{
  "qid": "GEN_1",
  "question": "Select the best way to improve the underlined portion: The report was prepared by the team **in a manner that was very thorough**.",
  "options": ["thoroughly", "with thoroughness", "in a thorough manner", "No improvement needed"],
  "correct": 0,
  "difficulty": "medium",
  "topic": "{topic}",
  "subtopic": "{subtopic if subtopic else topic}",
  "tags": ["conciseness", "adverb-usage"]
}}

NOW CREATE {count} HIGHLY DIVERSE SENTENCE IMPROVEMENT QUESTIONS:"""
    
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
        
        for key, contexts in context_map.items():
            if key in exam_slug.lower():
                return contexts
        
        return [
            'business and commerce', 'education and learning', 'technology and innovation',
            'social interactions', 'government and politics', 'science and research',
            'arts and culture', 'sports and fitness', 'environment and nature',
            'health and wellness', 'travel and tourism', 'media and communication'
        ]
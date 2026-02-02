import json
import time
import random
import logging
from typing import List, Dict
from rapidfuzz import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import config
from db_manager import DBManager
from generator import QuestionGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContentAgent:
    """Manages automated question generation and insertion."""
    
    def __init__(self):
        self.db = DBManager(config.MONGO_URI, config.DB_NAME)
        self.ai = QuestionGenerator(config.PROJECT_ID, config.LOCATION, config.MODEL_NAME)
        self._question_counter = 0
        self._stats = {
            'total_generated': 0,
            'total_inserted': 0,
            'total_duplicates': 0,
            'exact_duplicates': 0,
            'fuzzy_duplicates': 0
        }
    
    def run(self):
        """Main execution loop."""
        logger.info("🚀 Gyapak Content Agent Started")
        logger.info(f"📊 Configuration: Batch Size={config.BATCH_SIZE}, Fuzzy Threshold={config.FUZZY_MATCH_THRESHOLD}")
        
        try:
            while True:
                self._process_round()
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            self._print_stats()
        finally:
            self.db.close()
    
    def _print_stats(self):
        """Print generation statistics."""
        logger.info("=" * 60)
        logger.info("📈 GENERATION STATISTICS:")
        logger.info(f"  Total Generated: {self._stats['total_generated']}")
        logger.info(f"  Total Inserted: {self._stats['total_inserted']}")
        logger.info(f"  Total Duplicates: {self._stats['total_duplicates']}")
        logger.info(f"    - Exact: {self._stats['exact_duplicates']}")
        logger.info(f"    - Fuzzy: {self._stats['fuzzy_duplicates']}")
        if self._stats['total_generated'] > 0:
            success_rate = (self._stats['total_inserted'] / self._stats['total_generated']) * 100
            logger.info(f"  Success Rate: {success_rate:.1f}%")
        logger.info("=" * 60)
    
    def _process_round(self):
        """Process one complete round of gap filling with PARALLEL processing."""
        try:
            logger.info("🔍 Checking for content gaps...")
            low_count_exams = self.db.get_low_count_exams(threshold=config.GAP_THRESHOLD)
            
            if not low_count_exams:
                logger.info(f"🎉 All exams reached threshold of {config.GAP_THRESHOLD}")
                self._print_stats()
                logger.info(f"Sleeping for {config.NO_GAPS_DELAY_SECONDS}s...")
                time.sleep(config.NO_GAPS_DELAY_SECONDS)
                return
            
            logger.info(f"Found {len(low_count_exams)} exams requiring content")
            
            # PARALLEL PROCESSING: Process multiple exams simultaneously
            max_workers = min(config.MAX_PARALLEL_EXAMS, len(low_count_exams))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_exam, exam): exam for exam in low_count_exams}
                
                for future in as_completed(futures):
                    exam = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing {exam['_id']}: {e}")
            
            logger.info("✅ Round complete. Restarting gap check...")
            self._print_stats()
            time.sleep(config.ROUND_DELAY_SECONDS)
            
        except Exception as e:
            self._handle_error(e)
    
    def _process_exam(self, exam: Dict):
        """Process a single exam."""
        exam_slug = exam['_id']
        current_count = exam['count']
        logger.info(f"🚀 Processing {exam_slug} (Current: {current_count})")
        
        seeds = self.db.get_seed_questions(exam_slug, limit=config.SEED_SAMPLE_SIZE)
        if not seeds:
            logger.warning(f"⚠️ No seed questions found for {exam_slug}")
            return
        
        seed = random.choice(seeds)
        logger.info(f"📝 Using seed - Topic: {seed.get('topic', 'N/A')}, Subtopic: {seed.get('subtopic', 'N/A')}")
        
        try:
            raw_response = self.ai.generate(json.dumps(seed, default=str), count=config.BATCH_SIZE)
            new_questions = self._parse_ai_response(raw_response)
            
            logger.info(f"🤖 AI generated {len(new_questions)} questions")
            self._stats['total_generated'] += len(new_questions)
            
            # Show sample of generated questions for debugging
            if new_questions:
                sample_q = new_questions[0]['question'][:80]
                logger.info(f"   Sample: {sample_q}...")
            
            processed = self._process_questions(new_questions, seed)
            
            if processed:
                inserted = self.db.bulk_insert_questions(processed)
                self._stats['total_inserted'] += inserted
                logger.info(f"✅ Added {inserted}/{len(new_questions)} questions to {exam_slug}")
                
                if inserted < len(processed):
                    logger.warning(f"⚠️ {len(processed) - inserted} questions failed to insert (likely DB duplicates)")
            else:
                logger.warning(f"⚠️ All {len(new_questions)} questions were duplicates!")
            
            time.sleep(config.BATCH_DELAY_SECONDS)
            
        except Exception as e:
            if "429" in str(e):
                raise
            logger.error(f"⚠️ Error processing {exam_slug}: {e}")
    
    def _parse_ai_response(self, raw_response: str) -> List[Dict]:
        """Parse and clean AI response."""
        cleaned = raw_response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    
    def _process_questions(self, questions: List[Dict], seed: Dict) -> List[Dict]:
        """Filter duplicates and hydrate questions with metadata."""
        topic = seed.get('topic')
        
        # Get existing questions for this topic
        existing_questions = self.db.get_questions_by_topic(topic)
        logger.info(f"🔍 Checking against {len(existing_questions)} existing questions in topic '{topic}'")
        
        processed = []
        for i, q in enumerate(questions):
            is_dup, dup_type = self._is_duplicate(q['question'], existing_questions)
            if is_dup:
                if dup_type == 'exact':
                    self._stats['exact_duplicates'] += 1
                else:
                    self._stats['fuzzy_duplicates'] += 1
                logger.debug(f"   Q{i+1}: Duplicate ({dup_type})")
                continue
            
            hydrated = self._hydrate_question(q, seed)
            processed.append(hydrated)
            logger.debug(f"   Q{i+1}: ✓ Unique")
        
        duplicates_found = len(questions) - len(processed)
        self._stats['total_duplicates'] += duplicates_found
        
        logger.info(f"🔬 Duplicate Analysis: {duplicates_found}/{len(questions)} filtered")
        logger.info(f"   Exact: {self._stats['exact_duplicates']}, Fuzzy: {self._stats['fuzzy_duplicates']}")
        
        return processed
    
    def _is_duplicate(self, question_text: str, existing_questions: List[str]) -> tuple:
        """
        Check if question is duplicate (exact or fuzzy match).
        
        Returns:
            Tuple of (is_duplicate: bool, type: str)
        """
        # Check exact match first (faster)
        if self.db.find_exact_match(question_text):
            return (True, 'exact')
        
        # Check fuzzy matches
        for existing in existing_questions:
            score = fuzz.token_set_ratio(question_text, existing)
            if score > config.FUZZY_MATCH_THRESHOLD:
                logger.debug(f"   Fuzzy match score: {score}")
                return (True, 'fuzzy')
        
        return (False, None)
    
    def _hydrate_question(self, question: Dict, seed: Dict) -> Dict:
        """Add metadata and system fields to question."""
        question['examId'] = seed.get('examId')
        question['examSlug'] = seed.get('examSlug')
        question['section'] = seed.get('section')
        question['sectionName'] = seed.get('sectionName')
        question['topic'] = seed.get('topic')
        question['subtopic'] = seed.get('subtopic')
        
        question['status'] = config.DEFAULT_STATUS
        question['__v'] = config.DEFAULT_VERSION
        question['tags'] = question.get('tags', [])
        
        self._question_counter += 1
        timestamp = int(time.time() * 1000)
        original_qid = question.get('qid', 'GEN')
        question['qid'] = f"{original_qid}_{timestamp}_{self._question_counter}"
        
        question.pop('_id', None)
        question.pop('createdAt', None)
        question.pop('updatedAt', None)
        
        return question
    
    def _handle_error(self, error: Exception):
        """Handle errors with appropriate backoff."""
        error_msg = str(error)
        
        if "429" in error_msg:
            wait_time = random.randint(config.QUOTA_BACKOFF_MIN, config.QUOTA_BACKOFF_MAX)
            logger.warning(f"🚨 QUOTA EXHAUSTED (429). Sleeping for {wait_time}s...")
            time.sleep(wait_time)
        else:
            logger.error(f"🚨 CRITICAL ERROR: {error}")
            logger.info(f"Restarting in {config.RETRY_DELAY_SECONDS}s...")
            time.sleep(config.RETRY_DELAY_SECONDS)


def main():
    agent = ContentAgent()
    agent.run()


if __name__ == "__main__":
    main()
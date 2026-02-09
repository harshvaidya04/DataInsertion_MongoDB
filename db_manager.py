from pymongo import MongoClient, ASCENDING
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DBManager:
    """Manages MongoDB operations for exam questions."""
    
    def __init__(self, uri: str, db_name: str):
        """
        Initialize MongoDB connection with optimized settings.
        
        Args:
            uri: MongoDB connection string
            db_name: Database name
        """
        self.client = MongoClient(
            uri,
            maxPoolSize=50,
            minPoolSize=10,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000
        )
        self.db = self.client[db_name]
        self.collection = self.db['examquestions']
        self._ensure_indexes()
        logger.info(f"Connected to MongoDB database: {db_name}")
    
    def _ensure_indexes(self):
        """Create indexes for optimal query performance."""
        try:
            # Index for exact duplicate detection
            self.collection.create_index([("question", ASCENDING)], background=True)
            # Index for topic-based queries (fuzzy matching)
            self.collection.create_index([("topic", ASCENDING)], background=True)
            # Index for exam slug queries
            self.collection.create_index([("examSlug", ASCENDING)], background=True)
            # ✅ Compound index for scoped queries
            self.collection.create_index([("topic", ASCENDING), ("examSlug", ASCENDING)], background=True)
            logger.info("Database indexes verified")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    def get_low_count_exams(self, threshold: int) -> List[Dict]:
        """
        Get exams with question count below threshold.
        
        Args:
            threshold: Minimum question count threshold
            
        Returns:
            List of exam slugs with their current counts
        """
        pipeline = [
            {"$group": {"_id": "$examSlug", "count": {"$sum": 1}}},
            {"$match": {"count": {"$lt": threshold}}},
            {"$sort": {"count": ASCENDING}}  # Process exams with fewest questions first
        ]
        return list(self.collection.aggregate(pipeline))
    
    def get_seed_questions(self, exam_slug: str, limit: int = 20) -> List[Dict]:
        """
        Fetch seed questions for content generation.
        
        Args:
            exam_slug: Exam identifier
            limit: Maximum number of seeds to fetch
            
        Returns:
            List of seed question documents
        """
        return list(
            self.collection.find(
                {"examSlug": exam_slug},
                {"_id": 0}  # Exclude MongoDB _id from results
            ).limit(limit)
        )
    
    def find_exact_match(self, question_text: str) -> bool:
        """
        Check if exact question text already exists.
        
        Args:
            question_text: Question text to check
            
        Returns:
            True if question exists, False otherwise
        """
        return self.collection.find_one(
            {"question": question_text.strip()},
            {"_id": 1}  # Only fetch _id field for faster query
        ) is not None
    
    def get_questions_by_topic(self, topic: str) -> List[str]:
        """
        Get all question texts for a specific topic (for fuzzy matching).
        
        Args:
            topic: Topic name
            
        Returns:
            List of question texts
        """
        cursor = self.collection.find(
            {"topic": topic},
            {"question": 1, "_id": 0}
        )
        return [doc['question'] for doc in cursor]
    
    def get_questions_by_topic_and_exam(self, topic: str, exam_slug: str) -> List[str]:
        """
        ✅ SCOPED METHOD: Get questions for both topic AND exam.
        
        This prevents comparing questions across different exams.
        
        Args:
            topic: Topic name
            exam_slug: Exam identifier
            
        Returns:
            List of question texts for this topic in this exam
        """
        cursor = self.collection.find(
            {"topic": topic, "examSlug": exam_slug},
            {"question": 1, "_id": 0}
        )
        return [doc['question'] for doc in cursor]
    
    def bulk_insert_questions(self, questions: List[Dict]) -> int:
        """
        Insert multiple questions in a single batch operation.
        
        Args:
            questions: List of question documents to insert
            
        Returns:
            Number of successfully inserted questions
        """
        if not questions:
            return 0
        
        try:
            result = self.collection.insert_many(questions, ordered=False)
            return len(result.inserted_ids)
        except Exception as e:
            # Handle partial success in bulk insert
            error_msg = str(e)
            if "writeErrors" in error_msg or "inserted" in error_msg.lower():
                # Some documents were inserted successfully
                logger.warning(f"Partial bulk insert: {e}")
                # Count successful insertions
                success_count = sum(
                    1 for q in questions 
                    if not self.find_exact_match(q.get('question', ''))
                )
                return len(questions) - success_count
            logger.error(f"Bulk insert failed: {e}")
            return 0
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
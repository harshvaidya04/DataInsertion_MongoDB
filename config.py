import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # MongoDB Configuration
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = "test"
    COLLECTION_NAME = "examquestions"
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-4o-mini"  # or "gpt-4o" for better quality
    
    # Content Generation Thresholds
    GAP_THRESHOLD = 10000  # Minimum questions per exam
    BATCH_SIZE = 60  # Questions to generate per batch
    SEED_SAMPLE_SIZE = 50  # Number of seed questions to sample from
    
    # Duplicate Detection Settings
    # ✅ FIXED: Changed from 0 to 85 for proper duplicate detection
    FUZZY_MATCH_THRESHOLD = 45  # Score above which questions are considered duplicates (0-100)
    # 0 = Everything is a duplicate (nothing gets inserted)
    # 85 = Good balance (recommended)
    # 95 = Very strict (may allow near-duplicates)
    
    # Processing Configuration
    MAX_PARALLEL_EXAMS = 3  # Process up to 3 exams simultaneously
    BATCH_DELAY_SECONDS = 10  # Delay between batches for same exam
    ROUND_DELAY_SECONDS = 2  # Delay between complete rounds
    NO_GAPS_DELAY_SECONDS = 300  # Wait time when all exams reach threshold (5min for testing, 3600 for production)
    
    # Error Handling
    RETRY_DELAY_SECONDS = 30  # Delay before retrying after general error
    QUOTA_BACKOFF_MIN = 120  # Minimum backoff time for quota errors (2 min)
    QUOTA_BACKOFF_MAX = 300  # Maximum backoff time for quota errors (5 min)
    
    # Question Metadata Defaults
    DEFAULT_STATUS = "pending_review"
    DEFAULT_VERSION = 0


config = Config()
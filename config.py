import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # MongoDB Configuration
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = "test"
    COLLECTION_NAME = "examquestions"
    
    # Google Cloud / Vertex AI Configuration
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    LOCATION = "global"
    MODEL_NAME = "gemini-2.5-flash"
    
    # Content Generation Thresholds
    GAP_THRESHOLD = 10000  # Minimum questions per exam
    BATCH_SIZE = 60  # Questions to generate per batch
    SEED_SAMPLE_SIZE = 50  # Number of seed questions to sample from
    
    # Duplicate Detection Settings
    # ADJUSTED: Increased threshold from 50 to 75 to allow more variations
    # This reduces false positives in duplicate detection
    FUZZY_MATCH_THRESHOLD = 75  # Score above which questions are considered duplicates (0-100)
    
    # Processing Configuration
    MAX_PARALLEL_EXAMS = 3  # Process up to 3 exams simultaneously
    BATCH_DELAY_SECONDS = 10  # Delay between batches for same exam
    ROUND_DELAY_SECONDS = 2  # Delay between complete rounds
    NO_GAPS_DELAY_SECONDS = 3600  # Wait time when all exams reach threshold (1 hour)
    
    # Error Handling
    RETRY_DELAY_SECONDS = 30  # Delay before retrying after general error
    QUOTA_BACKOFF_MIN = 120  # Minimum backoff time for quota errors (2 min)
    QUOTA_BACKOFF_MAX = 300  # Maximum backoff time for quota errors (5 min)
    
    # Question Metadata Defaults
    DEFAULT_STATUS = "pending_review"
    DEFAULT_VERSION = 0

config = Config()
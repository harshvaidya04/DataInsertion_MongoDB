import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = "test"
    COLLECTION_NAME = "examquestions"
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    LOCATION = "global"
    MODEL_NAME = "gemini-2.5-flash"
    GAP_THRESHOLD = 10000
    BATCH_SIZE = 60
    SEED_SAMPLE_SIZE = 50
    FUZZY_MATCH_THRESHOLD = 50
    MAX_PARALLEL_EXAMS = 5
    BATCH_DELAY_SECONDS = 10
    ROUND_DELAY_SECONDS = 2
    NO_GAPS_DELAY_SECONDS = 3600
    RETRY_DELAY_SECONDS = 30
    QUOTA_BACKOFF_MIN = 120
    QUOTA_BACKOFF_MAX = 300
    DEFAULT_STATUS = "pending_review"
    DEFAULT_VERSION = 0

config = Config()

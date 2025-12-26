import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    
    @classmethod
    def validate(cls):
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        # YOUTUBE_API_KEY is optional. Warn but do not block startup.
        if not cls.YOUTUBE_API_KEY:
            print("⚠️ YOUTUBE_API_KEY not set — YouTube Data API features disabled (optional).")

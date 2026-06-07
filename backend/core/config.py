import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY: str    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str      = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
MAX_ITERATIONS: int    = int(os.getenv("MAX_ITERATIONS", "3"))
QUALITY_THRESHOLD: int = int(os.getenv("QUALITY_THRESHOLD", "7"))

"""
Configuration settings for the GenAI Image Comparison project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REFERENCE_IMAGES_DIR = PROJECT_ROOT / "reference_images"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, REFERENCE_IMAGES_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# API Keys (replace with your actual keys or use environment variables)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_openai_key")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "your_key")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "your_secret")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Test prompts with varying complexity
TEST_PROMPTS = {
    'simple': [
        "A red apple on a wooden table",
        "A cute cat sitting in sunlight", 
        "A blue car on a city street",
        "A white flower in a green field"
    ],
    'medium': [
        "A vintage bicycle parked outside a cozy cafe with warm lighting",
        "An elderly man reading a newspaper in a park during autumn",
        "A modern kitchen with marble countertops and hanging plants",
        "Children playing soccer in a schoolyard during golden hour"
    ],
    'complex': [
        "A surreal landscape where giant musical instruments grow like trees under a purple sky",
        "A steampunk-style robot serving tea to Victorian-era people in an ornate garden",
        "An underwater city with bioluminescent architecture and swimming whales",
        "A time-lapse effect showing a flower blooming while seasons change around it"
    ],
    'text_heavy': [
        "A vintage poster with text 'GRAND OPENING' in bold red letters",
        "A street sign showing 'Main Street' and '5th Avenue' intersection",
        "A book cover with title 'The Art of AI' in elegant typography",
        "A cafe menu board with prices and coffee names clearly visible"
    ]
}

# Evaluation parameters
EVALUATION_METRICS = {
    'clip_score': True,
    'fid_score': True,
    'human_evaluation': True,
    'generation_time': True
}

# Image generation parameters
IMAGE_SIZE = (1024, 1024)
IMAGE_QUALITY = "standard"  # Options: "standard", "hd" (for models that support it)

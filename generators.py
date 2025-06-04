"""
Image generation modules for different AI models.
"""

import time
import json
import base64
import io
import requests
import boto3
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline
import openai

from config import (
    OPENAI_API_KEY,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    IMAGE_SIZE,
    IMAGE_QUALITY
)


class DallE3Generator:
    """Generate images using OpenAI's DALL-E 3 model."""
    
    def __init__(self):
        """Initialize the DALL-E 3 generator."""
        openai.api_key = OPENAI_API_KEY
    
    def generate(self, prompt):
        """
        Generate an image using DALL-E 3.
        
        Args:
            prompt: Text prompt for image generation
            
        Returns:
            tuple: (PIL Image, generation time in seconds)
        """
        start_time = time.time()
        
        try:
            response = openai.Image.create(
                model="dall-e-3",
                prompt=prompt,
                size=f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
                quality=IMAGE_QUALITY,
                n=1,
            )
            
            image_url = response['data'][0]['url']
            image = Image.open(requests.get(image_url, stream=True).raw)
            
        except Exception as e:
            print(f"Error generating image with DALL-E 3: {e}")
            # Return a blank image in case of error
            image = Image.new('RGB', IMAGE_SIZE, color='white')
        
        generation_time = time.time() - start_time
        return image, generation_time


class NovaCanvasGenerator:
    """Generate images using Amazon's Nova Canvas model."""
    
    def __init__(self):
        """Initialize the Nova Canvas generator."""
        self.session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        self.bedrock = self.session.client('bedrock-runtime')
    
    def generate(self, prompt):
        """
        Generate an image using Nova Canvas.
        
        Args:
            prompt: Text prompt for image generation
            
        Returns:
            tuple: (PIL Image, generation time in seconds)
        """
        start_time = time.time()
        
        try:
            request_body = {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": prompt
                },
                "imageGenerationConfig": {
                    "quality": IMAGE_QUALITY,
                    "width": IMAGE_SIZE[0],
                    "height": IMAGE_SIZE[1]
                }
            }
            
            response = self.bedrock.invoke_model(
                modelId='amazon.nova-canvas-v1:0',
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            image_data = base64.b64decode(response_body['images'][0])
            image = Image.open(io.BytesIO(image_data))
            
        except Exception as e:
            print(f"Error generating image with Nova Canvas: {e}")
            # Return a blank image in case of error
            image = Image.new('RGB', IMAGE_SIZE, color='white')
        
        generation_time = time.time() - start_time
        return image, generation_time


class StableDiffusionGenerator:
    """Generate images using Stability AI's Stable Diffusion XL model."""
    
    def __init__(self):
        """Initialize the Stable Diffusion XL generator."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ).to(self.device)
    
    def generate(self, prompt):
        """
        Generate an image using Stable Diffusion XL.
        
        Args:
            prompt: Text prompt for image generation
            
        Returns:
            tuple: (PIL Image, generation time in seconds)
        """
        start_time = time.time()
        
        try:
            # Set a fixed seed for reproducibility
            generator = torch.Generator(device=self.device).manual_seed(42)
            
            image = self.pipeline(
                prompt=prompt,
                height=IMAGE_SIZE[1],
                width=IMAGE_SIZE[0],
                generator=generator
            ).images[0]
            
        except Exception as e:
            print(f"Error generating image with Stable Diffusion XL: {e}")
            # Return a blank image in case of error
            image = Image.new('RGB', IMAGE_SIZE, color='white')
        
        generation_time = time.time() - start_time
        return image, generation_time


# Factory function to get the appropriate generator
def get_generator(model_name):
    """
    Get an image generator based on the model name.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        object: An instance of the appropriate generator class
    """
    generators = {
        'DALL-E-3': DallE3Generator,
        'Nova-Canvas': NovaCanvasGenerator,
        'SDXL': StableDiffusionGenerator
    }
    
    if model_name not in generators:
        raise ValueError(f"Unknown model: {model_name}")
    
    return generators[model_name]()

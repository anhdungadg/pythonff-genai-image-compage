"""
Implementation of evaluation metrics for AI-generated images.
"""

import torch
import numpy as np
from PIL import Image
import io
import base64
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
import lpips
from transformers import CLIPProcessor, CLIPModel


class CLIPEvaluator:
    """Evaluates semantic consistency between images and text using CLIP."""
    
    def __init__(self):
        """Initialize the CLIP model and processor."""
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def calculate_clip_score(self, image, text):
        """
        Calculate CLIP score between image and text.
        
        Args:
            image: PIL Image object
            text: String prompt
            
        Returns:
            float: CLIP score (higher means better alignment)
        """
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        clip_score = torch.nn.functional.softmax(logits_per_image, dim=1)[0][0].item()
        return clip_score


class LPIPSEvaluator:
    """Evaluates perceptual similarity between images using LPIPS."""
    
    def __init__(self):
        """Initialize the LPIPS model."""
        self.model = lpips.LPIPS(net='alex')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def calculate_lpips_score(self, image1, image2):
        """
        Calculate LPIPS score between two images.
        
        Args:
            image1: PIL Image object
            image2: PIL Image object
            
        Returns:
            float: LPIPS score (lower means more similar)
        """
        # Convert PIL images to tensors
        transform = lpips.transforms.ToTensor()
        img1_tensor = transform(image1).unsqueeze(0).to(self.device)
        img2_tensor = transform(image2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            lpips_score = self.model(img1_tensor, img2_tensor).item()
        
        return lpips_score


def calculate_ssim_score(image1, image2):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        image1: PIL Image object
        image2: PIL Image object
        
    Returns:
        float: SSIM score (higher means more similar)
    """
    # Convert PIL images to numpy arrays
    img1_array = np.array(image1.convert('L'))
    img2_array = np.array(image2.convert('L'))
    
    # Ensure images are the same size
    if img1_array.shape != img2_array.shape:
        # Resize the second image to match the first
        image2 = image2.resize(image1.size)
        img2_array = np.array(image2.convert('L'))
    
    # Calculate SSIM
    score = ssim(img1_array, img2_array)
    return score


def extract_inception_features(images):
    """
    Extract features using InceptionV3 for FID calculation.
    This is a placeholder - in a real implementation, you would use a proper
    InceptionV3 model to extract features.
    
    Args:
        images: List of PIL Image objects
        
    Returns:
        numpy.ndarray: Feature vectors
    """
    # This is a simplified placeholder
    # In a real implementation, you would use:
    # from torchvision.models import inception_v3
    # or a library like cleanfid
    
    # Placeholder implementation
    features = np.random.randn(len(images), 2048)
    return features


def calculate_fid_score(real_images, generated_images):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.
    
    Args:
        real_images: List of PIL Image objects (real images)
        generated_images: List of PIL Image objects (AI-generated images)
        
    Returns:
        float: FID score (lower means more similar distributions)
    """
    # Extract features
    real_features = extract_inception_features(real_images)
    gen_features = extract_inception_features(generated_images)
    
    # Calculate statistics
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check if covmean has complex parts due to numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid


def create_human_evaluation_form(image_path, prompt):
    """
    Create a form for human evaluation of generated images.
    
    Args:
        image_path: Path to the generated image
        prompt: Text prompt used to generate the image
        
    Returns:
        dict: Evaluation form template
        str: Instructions for evaluators
    """
    evaluation_form = {
        'image_path': image_path,
        'prompt': prompt,
        'photorealism': None,  # 1-5 scale
        'detail_quality': None,  # 1-5 scale
        'color_quality': None,  # 1-5 scale
        'prompt_adherence': None,  # 1-5 scale
        'artifacts': None,  # 1-5 scale (5 = no artifacts)
        'overall_quality': None,  # 1-5 scale
        'comments': ""
    }
    
    # Instructions for evaluators
    instructions = """
    Đánh giá từ 1-5 cho mỗi tiêu chí:
    1 = Rất kém, 2 = Kém, 3 = Trung bình, 4 = Tốt, 5 = Xuất sắc
    
    Photorealism: Mức độ giống ảnh thật
    Detail Quality: Độ sắc nét và chi tiết
    Color Quality: Chất lượng màu sắc và tương phản
    Prompt Adherence: Mức độ tuân thủ yêu cầu
    Artifacts: Không có lỗi kỹ thuật (5 = không lỗi)
    Overall Quality: Đánh giá tổng thể
    """
    
    return evaluation_form, instructions


def calculate_weighted_score(model_name, analysis, human_eval_results):
    """
    Calculate a weighted score for a model based on automated metrics and human evaluation.
    
    Args:
        model_name: Name of the model
        analysis: Dictionary containing automated metrics
        human_eval_results: Dictionary containing human evaluation results
        
    Returns:
        float: Weighted score
    """
    # Define weights for different metrics
    weights = {
        'clip_score': 0.3,
        'generation_time': 0.1,
        'human_overall': 0.4,
        'human_prompt_adherence': 0.2
    }
    
    # Extract metrics
    model_data = analysis[model_name]
    
    # Calculate average CLIP score across categories
    avg_clip = np.mean([
        model_data[cat]['avg_clip_score'] 
        for cat in ['simple', 'medium', 'complex', 'text_heavy']
    ])
    
    # Normalize generation time (lower is better)
    avg_time = np.mean([
        model_data[cat]['avg_generation_time'] 
        for cat in ['simple', 'medium', 'complex', 'text_heavy']
    ])
    max_time = 30.0  # Assume 30 seconds is the maximum acceptable time
    normalized_time = 1.0 - min(avg_time / max_time, 1.0)
    
    # Get human evaluation scores if available
    human_overall = 0.0
    human_adherence = 0.0
    
    if model_name in human_eval_results:
        human_scores = human_eval_results[model_name]
        human_overall = np.mean([s['overall_quality'] for s in human_scores]) / 5.0  # Normalize to 0-1
        human_adherence = np.mean([s['prompt_adherence'] for s in human_scores]) / 5.0  # Normalize to 0-1
    
    # Calculate weighted score
    weighted_score = (
        weights['clip_score'] * avg_clip +
        weights['generation_time'] * normalized_time +
        weights['human_overall'] * human_overall +
        weights['human_prompt_adherence'] * human_adherence
    )
    
    return weighted_score

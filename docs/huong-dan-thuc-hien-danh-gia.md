# Hướng dẫn thực hiện đánh giá chi tiết các Model AI Generate Images

## 1. Chuẩn bị environment và tools

### A. Python libraries cần thiết
```python
# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Image processing
import cv2
from PIL import Image
import torch
from torchvision import transforms

# Evaluation metrics
from transformers import CLIPProcessor, CLIPModel
import lpips
from skimage.metrics import structural_similarity as ssim

# API clients
import openai
import boto3
from diffusers import StableDiffusionXLPipeline
```

### B. Setup API connections
```python
# OpenAI DALL-E 3 setup
openai.api_key = "your_openai_key"

# AWS Nova Canvas setup  
session = boto3.Session(
    aws_access_key_id='your_key',
    aws_secret_access_key='your_secret',
    region_name='us-east-1'
)

# Stability AI setup
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
)
```

## 2. Dataset chuẩn bị cho testing

### A. Tạo bộ prompts test chuẩn
```python
# Prompts với độ khó tăng dần
test_prompts = {
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
```

### B. Reference images cho comparison
```python
# Load reference dataset cho FID calculation
def load_reference_dataset(path):
    """Load real images for FID comparison"""
    reference_images = []
    for img_path in glob.glob(f"{path}/*.jpg"):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512))
        reference_images.append(np.array(img))
    return np.array(reference_images)

reference_dataset = load_reference_dataset("path/to/real_images")
```

## 3. Implementation các metrics đánh giá

### A. CLIP Score cho semantic consistency
```python
class CLIPEvaluator:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def calculate_clip_score(self, image, text):
        """Calculate CLIP score between image and text"""
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        clip_score = torch.nn.functional.softmax(logits_per_image, dim=1)[0][0].item()
        return clip_score

clip_evaluator = CLIPEvaluator()
```

### B. FID Score implementation
```python
def calculate_fid_score(real_images, generated_images):
    """Calculate FID between real and generated images"""
    # This is a simplified version - use cleanfid library for production
    from scipy.linalg import sqrtm
    
    # Extract features using Inception v3
    real_features = extract_inception_features(real_images)
    gen_features = extract_inception_features(generated_images)
    
    # Calculate statistics
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid
```

### C. Human evaluation interface
```python
def human_evaluation_interface(image_path, prompt):
    """Create interface for human evaluation"""
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
    """
    
    return evaluation_form, instructions
```

## 4. Testing workflow implementation

### A. Automated testing pipeline
```python
class ModelEvaluationPipeline:
    def __init__(self):
        self.results = []
        self.clip_evaluator = CLIPEvaluator()
    
    def test_single_model(self, model_name, prompts, generation_function):
        """Test một model với bộ prompts"""
        model_results = {
            'model': model_name,
            'results': []
        }
        
        for category, prompt_list in prompts.items():
            for prompt in prompt_list:
                # Generate image
                start_time = time.time()
                image = generation_function(prompt)
                generation_time = time.time() - start_time
                
                # Calculate metrics
                clip_score = self.clip_evaluator.calculate_clip_score(image, prompt)
                
                result = {
                    'prompt': prompt,
                    'category': category,
                    'generation_time': generation_time,
                    'clip_score': clip_score,
                    'image_path': f"outputs/{model_name}_{hash(prompt)}.png"
                }
                
                # Save image
                image.save(result['image_path'])
                model_results['results'].append(result)
        
        return model_results
    
    def run_complete_evaluation(self):
        """Chạy đánh giá toàn bộ cho tất cả models"""
        # Test GPT DALL-E 3
        dalle_results = self.test_single_model(
            'DALL-E-3', 
            test_prompts, 
            self.generate_dalle3
        )
        
        # Test Amazon Nova Canvas
        nova_results = self.test_single_model(
            'Nova-Canvas',
            test_prompts,
            self.generate_nova
        )
        
        # Test Stability Diffusion XL  
        sdxl_results = self.test_single_model(
            'SDXL',
            test_prompts,
            self.generate_sdxl
        )
        
        return [dalle_results, nova_results, sdxl_results]
```

### B. Model-specific generation functions
```python
def generate_dalle3(prompt):
    """Generate image using DALL-E 3"""
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    
    image_url = response['data'][0]['url']
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image

def generate_nova_canvas(prompt):
    """Generate image using Amazon Nova Canvas"""
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    request_body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "quality": "standard",
            "width": 1024,
            "height": 1024
        }
    }
    
    response = bedrock.invoke_model(
        modelId='amazon.nova-canvas-v1:0',
        body=json.dumps(request_body)
    )
    
    # Process response and return image
    response_body = json.loads(response['body'].read())
    image_data = base64.b64decode(response_body['images'][0])
    image = Image.open(io.BytesIO(image_data))
    return image

def generate_sdxl(prompt):
    """Generate image using Stability Diffusion XL"""
    image = pipeline(prompt).images[0]
    return image
```

## 5. Analysis và reporting

### A. Statistical analysis
```python
def analyze_results(all_results):
    """Phân tích kết quả và tạo báo cáo"""
    analysis = {}
    
    for model_result in all_results:
        model_name = model_result['model']
        results = model_result['results']
        
        # Calculate averages by category
        category_stats = {}
        for category in ['simple', 'medium', 'complex', 'text_heavy']:
            category_results = [r for r in results if r['category'] == category]
            
            if category_results:
                avg_clip = np.mean([r['clip_score'] for r in category_results])
                avg_time = np.mean([r['generation_time'] for r in category_results])
                
                category_stats[category] = {
                    'avg_clip_score': avg_clip,
                    'avg_generation_time': avg_time,
                    'sample_count': len(category_results)
                }
        
        analysis[model_name] = category_stats
    
    return analysis

def generate_report(analysis, human_eval_results):
    """Tạo báo cáo tổng hợp"""
    report = {
        'summary': {},
        'detailed_analysis': analysis,
        'human_evaluation': human_eval_results,
        'recommendations': {}
    }
    
    # Calculate weighted scores cho mỗi model
    for model in analysis.keys():
        weighted_score = calculate_weighted_score(model, analysis, human_eval_results)
        report['summary'][model] = weighted_score
    
    return report
```

### B. Visualization
```python
def create_evaluation_charts(analysis):
    """Tạo charts để visualize kết quả"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = list(analysis.keys())
    categories = ['simple', 'medium', 'complex', 'text_heavy']
    
    # CLIP Score comparison
    clip_data = []
    for model in models:
        model_clips = []
        for cat in categories:
            model_clips.append(analysis[model][cat]['avg_clip_score'])
        clip_data.append(model_clips)
    
    axes[0,0].bar(range(len(models)), [np.mean(scores) for scores in clip_data])
    axes[0,0].set_title('Average CLIP Scores')
    axes[0,0].set_xticks(range(len(models)))
    axes[0,0].set_xticklabels(models)
    
    # Generation time comparison  
    time_data = []
    for model in models:
        model_times = []
        for cat in categories:
            model_times.append(analysis[model][cat]['avg_generation_time'])
        time_data.append(model_times)
    
    axes[0,1].bar(range(len(models)), [np.mean(times) for times in time_data])
    axes[0,1].set_title('Average Generation Time (seconds)')
    axes[0,1].set_xticks(range(len(models)))
    axes[0,1].set_xticklabels(models)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## 6. Quality control và validation

### A. Inter-rater reliability
```python
def calculate_inter_rater_reliability(evaluations):
    """Tính toán độ tin cậy giữa các đánh giá viên"""
    from scipy.stats import pearsonr
    
    reliability_scores = {}
    evaluators = list(evaluations.keys())
    
    for i, eval1 in enumerate(evaluators):
        for j, eval2 in enumerate(evaluators[i+1:], i+1):
            scores1 = [e['overall_quality'] for e in evaluations[eval1]]
            scores2 = [e['overall_quality'] for e in evaluations[eval2]]
            
            correlation, p_value = pearsonr(scores1, scores2)
            reliability_scores[f"{eval1}_vs_{eval2}"] = {
                'correlation': correlation,
                'p_value': p_value
            }
    
    return reliability_scores
```

### B. Validation checks
```python
def validate_evaluation_results(results):
    """Kiểm tra tính hợp lệ của kết quả đánh giá"""
    validation_report = {
        'completeness': {},
        'consistency': {},
        'outliers': {}
    }
    
    # Check completeness
    for model_result in results:
        model_name = model_result['model']
        total_expected = len(test_prompts['simple']) + len(test_prompts['medium']) + \
                        len(test_prompts['complex']) + len(test_prompts['text_heavy'])
        actual_count = len(model_result['results'])
        
        validation_report['completeness'][model_name] = {
            'expected': total_expected,
            'actual': actual_count,
            'completion_rate': actual_count / total_expected
        }
    
    return validation_report
```

## 7. Automation và scheduling

### A. Automated evaluation runner
```python
def run_automated_evaluation():
    """Chạy đánh giá tự động định kỳ"""
    pipeline = ModelEvaluationPipeline()
    
    try:
        # Run evaluation
        results = pipeline.run_complete_evaluation()
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"evaluation_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Send notification
        send_completion_notification(report_path)
        
        return True
        
    except Exception as e:
        send_error_notification(str(e))
        return False

# Schedule evaluation (using cron or task scheduler)
```

## 8. Best practices và troubleshooting

### A. Common issues và solutions
- **API rate limits**: Implement exponential backoff
- **Memory issues**: Process images in batches
- **Inconsistent results**: Use fixed seeds when possible
- **Evaluation bias**: Randomize order, use blind evaluation

### B. Performance optimization
- Cache model loading
- Parallel processing cho independent evaluations  
- GPU optimization cho local models
- Efficient image storage và retrieval

Bằng cách follow guide này, bạn sẽ có thể implement một comprehensive evaluation framework cho việc so sánh các AI image generation models một cách objective và systematic.
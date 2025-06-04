"""
Main evaluation pipeline for comparing AI image generation models.
"""

import os
import time
import json
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

from config import (
    TEST_PROMPTS,
    OUTPUT_DIR,
    REFERENCE_IMAGES_DIR,
    REPORTS_DIR
)
from metrics import (
    CLIPEvaluator,
    calculate_fid_score,
    create_human_evaluation_form,
    calculate_weighted_score
)
from generators import get_generator


class ModelEvaluationPipeline:
    """Pipeline for evaluating and comparing AI image generation models."""
    
    def __init__(self):
        """Initialize the evaluation pipeline."""
        self.results = []
        self.clip_evaluator = CLIPEvaluator()
        
        # Ensure output directories exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
    
    def test_single_model(self, model_name, prompts, subset=None):
        """
        Test a single model with a set of prompts.
        
        Args:
            model_name: Name of the model to test
            prompts: Dictionary of prompts categorized by difficulty
            subset: Optional list of categories to test (e.g., ['simple', 'medium'])
            
        Returns:
            dict: Results of the evaluation
        """
        print(f"Testing model: {model_name}")
        
        # Get the appropriate generator
        generator = get_generator(model_name)
        
        model_results = {
            'model': model_name,
            'results': []
        }
        
        # Filter categories if subset is specified
        categories = prompts.keys() if subset is None else subset
        
        for category in categories:
            if category not in prompts:
                continue
                
            print(f"  Processing category: {category}")
            prompt_list = prompts[category]
            
            for i, prompt in enumerate(prompt_list):
                print(f"    Prompt {i+1}/{len(prompt_list)}: {prompt[:30]}...")
                
                # Generate image
                image, generation_time = generator.generate(prompt)
                
                # Calculate CLIP score
                clip_score = self.clip_evaluator.calculate_clip_score(image, prompt)
                
                # Create a unique filename based on the prompt
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                image_filename = f"{model_name}_{category}_{prompt_hash}.png"
                image_path = os.path.join(OUTPUT_DIR, image_filename)
                
                # Save image
                image.save(image_path)
                
                result = {
                    'prompt': prompt,
                    'category': category,
                    'generation_time': generation_time,
                    'clip_score': clip_score,
                    'image_path': image_path
                }
                
                model_results['results'].append(result)
                
                print(f"    Generation time: {generation_time:.2f}s, CLIP score: {clip_score:.4f}")
        
        return model_results
    
    def run_complete_evaluation(self, models=None, categories=None):
        """
        Run a complete evaluation on all specified models.
        
        Args:
            models: List of model names to evaluate (default: all available models)
            categories: List of prompt categories to use (default: all categories)
            
        Returns:
            list: Results for all models
        """
        if models is None:
            models = ['DALL-E-3', 'Nova-Canvas', 'SDXL']
        
        all_results = []
        
        for model_name in models:
            model_results = self.test_single_model(model_name, TEST_PROMPTS, categories)
            all_results.append(model_results)
        
        self.results = all_results
        return all_results
    
    def analyze_results(self):
        """
        Analyze the evaluation results.
        
        Returns:
            dict: Analysis of the results
        """
        if not self.results:
            raise ValueError("No results to analyze. Run evaluation first.")
        
        analysis = {}
        
        for model_result in self.results:
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
    
    def generate_report(self, analysis, human_eval_results=None):
        """
        Generate a comprehensive report of the evaluation.
        
        Args:
            analysis: Dictionary containing analysis of results
            human_eval_results: Optional dictionary containing human evaluation results
            
        Returns:
            dict: Complete evaluation report
        """
        if human_eval_results is None:
            human_eval_results = {}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_analysis': analysis,
            'human_evaluation': human_eval_results,
            'recommendations': {}
        }
        
        # Calculate weighted scores for each model
        for model in analysis.keys():
            weighted_score = calculate_weighted_score(model, analysis, human_eval_results)
            report['summary'][model] = weighted_score
        
        # Add recommendations based on the analysis
        best_model = max(report['summary'], key=report['summary'].get)
        report['recommendations']['best_overall'] = best_model
        
        # Find best model for each category
        for category in ['simple', 'medium', 'complex', 'text_heavy']:
            best_for_category = None
            best_score = -1
            
            for model in analysis.keys():
                if category in analysis[model]:
                    score = analysis[model][category]['avg_clip_score']
                    if score > best_score:
                        best_score = score
                        best_for_category = model
            
            if best_for_category:
                report['recommendations'][f'best_for_{category}'] = best_for_category
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(REPORTS_DIR, f"evaluation_report_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def create_evaluation_charts(self, analysis):
        """
        Create visualization charts for the evaluation results.
        
        Args:
            analysis: Dictionary containing analysis of results
            
        Returns:
            str: Path to the saved chart image
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        models = list(analysis.keys())
        categories = ['simple', 'medium', 'complex', 'text_heavy']
        
        # CLIP Score comparison
        clip_data = []
        for model in models:
            model_clips = []
            for cat in categories:
                if cat in analysis[model]:
                    model_clips.append(analysis[model][cat]['avg_clip_score'])
                else:
                    model_clips.append(0)
            clip_data.append(model_clips)
        
        # Bar chart for average CLIP scores
        x = np.arange(len(categories))
        width = 0.2
        for i, model in enumerate(models):
            axes[0,0].bar(x + i*width, clip_data[i], width, label=model)
        
        axes[0,0].set_title('Average CLIP Scores by Category')
        axes[0,0].set_xticks(x + width)
        axes[0,0].set_xticklabels(categories)
        axes[0,0].set_ylabel('CLIP Score')
        axes[0,0].legend()
        
        # Generation time comparison  
        time_data = []
        for model in models:
            model_times = []
            for cat in categories:
                if cat in analysis[model]:
                    model_times.append(analysis[model][cat]['avg_generation_time'])
                else:
                    model_times.append(0)
            time_data.append(model_times)
        
        # Bar chart for average generation times
        for i, model in enumerate(models):
            axes[0,1].bar(x + i*width, time_data[i], width, label=model)
        
        axes[0,1].set_title('Average Generation Time by Category')
        axes[0,1].set_xticks(x + width)
        axes[0,1].set_xticklabels(categories)
        axes[0,1].set_ylabel('Time (seconds)')
        axes[0,1].legend()
        
        # Overall CLIP score comparison
        overall_clip = [np.mean(scores) for scores in clip_data]
        axes[1,0].bar(range(len(models)), overall_clip)
        axes[1,0].set_title('Overall Average CLIP Scores')
        axes[1,0].set_xticks(range(len(models)))
        axes[1,0].set_xticklabels(models)
        axes[1,0].set_ylabel('CLIP Score')
        
        # Overall generation time comparison
        overall_time = [np.mean(times) for times in time_data]
        axes[1,1].bar(range(len(models)), overall_time)
        axes[1,1].set_title('Overall Average Generation Time')
        axes[1,1].set_xticks(range(len(models)))
        axes[1,1].set_xticklabels(models)
        axes[1,1].set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(REPORTS_DIR, f"evaluation_charts_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        
        return chart_path
    
    def load_reference_dataset(self, path=None):
        """
        Load reference images for FID calculation.
        
        Args:
            path: Path to the reference images directory
            
        Returns:
            list: List of PIL Image objects
        """
        if path is None:
            path = REFERENCE_IMAGES_DIR
        
        reference_images = []
        for img_path in glob.glob(f"{path}/*.jpg") + glob.glob(f"{path}/*.png"):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((512, 512))  # Resize for consistency
                reference_images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        return reference_images
    
    def validate_evaluation_results(self):
        """
        Validate the evaluation results for completeness and consistency.
        
        Returns:
            dict: Validation report
        """
        if not self.results:
            raise ValueError("No results to validate. Run evaluation first.")
        
        validation_report = {
            'completeness': {},
            'consistency': {},
            'outliers': {}
        }
        
        # Check completeness
        for model_result in self.results:
            model_name = model_result['model']
            total_expected = sum(len(TEST_PROMPTS[cat]) for cat in TEST_PROMPTS)
            actual_count = len(model_result['results'])
            
            validation_report['completeness'][model_name] = {
                'expected': total_expected,
                'actual': actual_count,
                'completion_rate': actual_count / total_expected if total_expected > 0 else 0
            }
        
        return validation_report


def run_automated_evaluation(models=None, categories=None):
    """
    Run an automated evaluation of the specified models.
    
    Args:
        models: List of model names to evaluate
        categories: List of prompt categories to use
        
    Returns:
        bool: True if evaluation completed successfully
    """
    pipeline = ModelEvaluationPipeline()
    
    try:
        # Run evaluation
        results = pipeline.run_complete_evaluation(models, categories)
        
        # Analyze results
        analysis = pipeline.analyze_results()
        
        # Generate report
        report = pipeline.generate_report(analysis)
        
        # Create charts
        chart_path = pipeline.create_evaluation_charts(analysis)
        
        print(f"Evaluation completed successfully.")
        print(f"Report saved to: {os.path.join(REPORTS_DIR)}")
        print(f"Charts saved to: {chart_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    run_automated_evaluation()

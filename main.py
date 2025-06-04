#!/usr/bin/env python3
"""
Main entry point for the GenAI Image Comparison tool.
"""

import argparse
import sys
import os
from datetime import datetime

from evaluation import ModelEvaluationPipeline, run_automated_evaluation
from config import TEST_PROMPTS


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare and evaluate AI image generation models"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["DALL-E-3", "Nova-Canvas", "SDXL", "all"],
        default=["all"],
        help="Models to evaluate (default: all)"
    )
    
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["simple", "medium", "complex", "text_heavy", "all"],
        default=["all"],
        help="Prompt categories to use (default: all)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output directory for reports and images"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with a minimal set of prompts"
    )
    
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate reports from existing evaluation results without running new evaluations"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the evaluation pipeline."""
    args = parse_arguments()
    
    # Process model selection
    if "all" in args.models:
        models = ["DALL-E-3", "Nova-Canvas", "SDXL"]
    else:
        models = args.models
    
    # Process category selection
    if "all" in args.categories:
        categories = None  # Use all categories
    else:
        categories = args.categories
    
    # Handle custom output directory
    if args.output:
        os.environ["OUTPUT_DIR"] = args.output
        os.environ["REPORTS_DIR"] = os.path.join(args.output, "reports")
        os.makedirs(os.environ["REPORTS_DIR"], exist_ok=True)
    
    # Quick test mode
    if args.quick_test:
        # Create a reduced set of prompts for quick testing
        quick_test_prompts = {
            category: TEST_PROMPTS[category][:1]  # Just use the first prompt from each category
            for category in TEST_PROMPTS
        }
        
        print("Running quick test with reduced prompt set...")
        pipeline = ModelEvaluationPipeline()
        results = []
        
        for model in models:
            result = pipeline.test_single_model(model, quick_test_prompts, categories)
            results.append(result)
        
        pipeline.results = results
        analysis = pipeline.analyze_results()
        report = pipeline.generate_report(analysis)
        chart_path = pipeline.create_evaluation_charts(analysis)
        
        print(f"Quick test completed. Report and charts saved to reports directory.")
        return 0
    
    # Report-only mode
    if args.report_only:
        print("Report-only mode not yet implemented.")
        # TODO: Implement loading previous results and generating new reports
        return 1
    
    # Full evaluation mode
    print(f"Starting full evaluation of models: {', '.join(models)}")
    if categories:
        print(f"Using prompt categories: {', '.join(categories)}")
    else:
        print("Using all prompt categories")
    
    success = run_automated_evaluation(models, categories)
    
    if success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

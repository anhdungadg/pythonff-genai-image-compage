"""
Human evaluation interface for the GenAI Image Comparison project.
"""

import os
import json
import glob
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime

from config import OUTPUT_DIR, REPORTS_DIR
from utils import save_human_evaluation_results, calculate_inter_rater_reliability


class HumanEvaluationApp:
    """GUI application for human evaluation of AI-generated images."""
    
    def __init__(self, root, images_dir=None, evaluator_name=None):
        """
        Initialize the human evaluation application.
        
        Args:
            root: Tkinter root window
            images_dir: Directory containing images to evaluate
            evaluator_name: Name of the evaluator
        """
        self.root = root
        self.images_dir = images_dir or OUTPUT_DIR
        self.evaluator_name = evaluator_name or "evaluator"
        
        # Set up the window
        self.root.title("GenAI Image Evaluation")
        self.root.geometry("1000x800")
        
        # Load image paths
        self.image_paths = self.load_image_paths()
        if not self.image_paths:
            self.show_error("No images found", "No images found in the specified directory.")
            return
        
        # Initialize variables
        self.current_index = 0
        self.evaluations = []
        
        # Create UI elements
        self.create_ui()
        
        # Load the first image
        self.load_current_image()
    
    def load_image_paths(self):
        """
        Load paths to all images in the images directory.
        
        Returns:
            list: List of image paths
        """
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        # Extract model and prompt information
        image_info = []
        for path in image_paths:
            filename = os.path.basename(path)
            parts = filename.split('_')
            
            if len(parts) >= 3:
                model = parts[0]
                category = parts[1]
                # The rest might be part of the prompt hash
                image_info.append({
                    'path': path,
                    'model': model,
                    'category': category,
                    'filename': filename
                })
        
        return image_info
    
    def create_ui(self):
        """Create the user interface elements."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(pady=10)
        
        # Image info
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(info_frame, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_label = ttk.Label(info_frame, text="")
        self.model_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(info_frame, text="Category:").grid(row=1, column=0, sticky=tk.W)
        self.category_label = ttk.Label(info_frame, text="")
        self.category_label.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(info_frame, text="Image:").grid(row=2, column=0, sticky=tk.W)
        self.filename_label = ttk.Label(info_frame, text="")
        self.filename_label.grid(row=2, column=1, sticky=tk.W)
        
        # Evaluation criteria
        eval_frame = ttk.LabelFrame(main_frame, text="Evaluation Criteria", padding=10)
        eval_frame.pack(fill=tk.X, pady=10)
        
        # Create sliders for each criterion
        self.criteria = {
            'photorealism': {'text': "Photorealism (1-5)", 'var': tk.IntVar(value=3)},
            'detail_quality': {'text': "Detail Quality (1-5)", 'var': tk.IntVar(value=3)},
            'color_quality': {'text': "Color Quality (1-5)", 'var': tk.IntVar(value=3)},
            'prompt_adherence': {'text': "Prompt Adherence (1-5)", 'var': tk.IntVar(value=3)},
            'artifacts': {'text': "No Artifacts (1-5)", 'var': tk.IntVar(value=3)},
            'overall_quality': {'text': "Overall Quality (1-5)", 'var': tk.IntVar(value=3)}
        }
        
        row = 0
        for key, data in self.criteria.items():
            ttk.Label(eval_frame, text=data['text']).grid(row=row, column=0, sticky=tk.W)
            slider = ttk.Scale(
                eval_frame, 
                from_=1, 
                to=5, 
                orient=tk.HORIZONTAL, 
                variable=data['var'],
                length=300
            )
            slider.grid(row=row, column=1, sticky=tk.W)
            
            # Add value label
            value_label = ttk.Label(eval_frame, text="3")
            value_label.grid(row=row, column=2, sticky=tk.W)
            
            # Update value label when slider changes
            data['var'].trace_add('write', lambda *args, label=value_label, var=data['var']: 
                                 label.config(text=str(var.get())))
            
            row += 1
        
        # Comments
        ttk.Label(eval_frame, text="Comments:").grid(row=row, column=0, sticky=tk.W)
        self.comments = tk.Text(eval_frame, height=3, width=40)
        self.comments.grid(row=row, column=1, columnspan=2, sticky=tk.W)
        
        # Navigation buttons
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=10)
        
        self.prev_button = ttk.Button(nav_frame, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(nav_frame, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(nav_frame, text="Save All", command=self.save_evaluations)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        # Progress indicator
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=len(self.image_paths))
        self.progress.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(main_frame, text="0 / 0")
        self.progress_label.pack()
    
    def load_current_image(self):
        """Load and display the current image and update UI elements."""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return
        
        image_info = self.image_paths[self.current_index]
        
        # Update image
        try:
            image = Image.open(image_info['path'])
            # Resize image to fit in the window
            image.thumbnail((600, 600))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
        except Exception as e:
            self.show_error("Error loading image", str(e))
            return
        
        # Update labels
        self.model_label.config(text=image_info['model'])
        self.category_label.config(text=image_info['category'])
        self.filename_label.config(text=image_info['filename'])
        
        # Update progress
        self.progress_var.set(self.current_index + 1)
        self.progress_label.config(text=f"{self.current_index + 1} / {len(self.image_paths)}")
        
        # Update navigation buttons
        self.prev_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(text="Next" if self.current_index < len(self.image_paths) - 1 else "Finish")
    
    def collect_current_evaluation(self):
        """
        Collect the evaluation data for the current image.
        
        Returns:
            dict: Evaluation data
        """
        image_info = self.image_paths[self.current_index]
        
        evaluation = {
            'image_path': image_info['path'],
            'model': image_info['model'],
            'category': image_info['category'],
            'filename': image_info['filename'],
            'comments': self.comments.get("1.0", tk.END).strip()
        }
        
        # Add criteria ratings
        for key, data in self.criteria.items():
            evaluation[key] = data['var'].get()
        
        return evaluation
    
    def next_image(self):
        """Save the current evaluation and move to the next image."""
        # Save current evaluation
        if self.current_index < len(self.image_paths):
            evaluation = self.collect_current_evaluation()
            
            # Update existing evaluation or add new one
            found = False
            for i, eval_data in enumerate(self.evaluations):
                if eval_data['image_path'] == evaluation['image_path']:
                    self.evaluations[i] = evaluation
                    found = True
                    break
            
            if not found:
                self.evaluations.append(evaluation)
        
        # Move to next image or finish
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_current_image()
            
            # Reset form for the new image
            for key, data in self.criteria.items():
                data['var'].set(3)
            self.comments.delete("1.0", tk.END)
        else:
            # We're at the last image, save and exit
            self.save_evaluations()
    
    def prev_image(self):
        """Save the current evaluation and move to the previous image."""
        # Save current evaluation
        if 0 <= self.current_index < len(self.image_paths):
            evaluation = self.collect_current_evaluation()
            
            # Update existing evaluation or add new one
            found = False
            for i, eval_data in enumerate(self.evaluations):
                if eval_data['image_path'] == evaluation['image_path']:
                    self.evaluations[i] = evaluation
                    found = True
                    break
            
            if not found:
                self.evaluations.append(evaluation)
        
        # Move to previous image
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
            
            # Load previous evaluation if it exists
            self.load_evaluation_data()
    
    def load_evaluation_data(self):
        """Load existing evaluation data for the current image."""
        current_path = self.image_paths[self.current_index]['path']
        
        for evaluation in self.evaluations:
            if evaluation['image_path'] == current_path:
                # Set criteria values
                for key, data in self.criteria.items():
                    if key in evaluation:
                        data['var'].set(evaluation[key])
                
                # Set comments
                self.comments.delete("1.0", tk.END)
                if 'comments' in evaluation:
                    self.comments.insert("1.0", evaluation['comments'])
                
                break
    
    def save_evaluations(self):
        """Save all evaluations to a file."""
        # Make sure we save the current evaluation
        if 0 <= self.current_index < len(self.image_paths):
            evaluation = self.collect_current_evaluation()
            
            # Update existing evaluation or add new one
            found = False
            for i, eval_data in enumerate(self.evaluations):
                if eval_data['image_path'] == evaluation['image_path']:
                    self.evaluations[i] = evaluation
                    found = True
                    break
            
            if not found:
                self.evaluations.append(evaluation)
        
        # Create a dictionary with evaluator name and evaluations
        data = {
            self.evaluator_name: self.evaluations
        }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"human_evaluation_{self.evaluator_name}_{timestamp}.json"
        file_path = save_human_evaluation_results(data, filename)
        
        self.show_info("Evaluations Saved", f"Evaluations saved to {file_path}")
    
    def show_error(self, title, message):
        """Show an error message dialog."""
        tk.messagebox.showerror(title, message)
    
    def show_info(self, title, message):
        """Show an information message dialog."""
        tk.messagebox.showinfo(title, message)


def run_human_evaluation(images_dir=None, evaluator_name=None):
    """
    Run the human evaluation application.
    
    Args:
        images_dir: Directory containing images to evaluate
        evaluator_name: Name of the evaluator
    """
    root = tk.Tk()
    app = HumanEvaluationApp(root, images_dir, evaluator_name)
    root.mainloop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Human evaluation interface for AI-generated images")
    parser.add_argument("--images-dir", help="Directory containing images to evaluate")
    parser.add_argument("--evaluator", help="Name of the evaluator")
    
    args = parser.parse_args()
    
    run_human_evaluation(args.images_dir, args.evaluator)

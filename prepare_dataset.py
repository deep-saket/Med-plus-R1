#!/usr/bin/env python3
"""
OmniMedVQA Dataset Preparation Script
====================================

This script helps you prepare and work with the OmniMedVQA dataset for Med-R1 training and evaluation.

Features:
- Load and merge multiple datasets
- Filter by modality or question type
- Create train/test splits
- Convert to formats compatible with Med-R1
- Generate statistics and reports

Usage:
    python prepare_dataset.py --mode merge --modality CT --output merged_ct_data.json
    python prepare_dataset.py --mode stats --dataset_path merged_ct_data.json
    python prepare_dataset.py --mode convert --input merged_ct_data.json --output inference_format.json
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import pandas as pd

class OmniMedVQAProcessor:
    """Processor for the OmniMedVQA dataset"""
    
    def __init__(self, data_root: str = "data/OmniMedVQA/OmniMedVQA"):
        """
        Initialize the processor
        
        Args:
            data_root: Root directory of the OmniMedVQA dataset
        """
        self.data_root = Path(data_root)
        self.qa_dir = self.data_root / "QA_information" / "Open-access"
        self.images_dir = self.data_root / "Images"
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {data_root}")
        
        print(f"📂 Dataset root: {self.data_root}")
        print(f"📋 QA directory: {self.qa_dir}")
        print(f"🖼️  Images directory: {self.images_dir}")
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset JSON files"""
        json_files = list(self.qa_dir.glob("*.json"))
        dataset_names = [f.stem for f in json_files]
        return sorted(dataset_names)
    
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load a specific dataset"""
        json_path = self.qa_dir / f"{dataset_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Dataset not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert relative image paths to absolute paths
        for item in data:
            if 'image_path' in item:
                item['image_path'] = str(self.data_root / item['image_path'])
        
        return data
    
    def get_dataset_stats(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a dataset or all datasets"""
        if dataset_name:
            datasets = [dataset_name]
        else:
            datasets = self.list_available_datasets()
        
        stats = {
            'total_samples': 0,
            'datasets': {},
            'modalities': Counter(),
            'question_types': Counter(),
            'gt_answers': Counter()
        }
        
        for ds_name in datasets:
            try:
                data = self.load_dataset(ds_name)
                ds_stats = {
                    'samples': len(data),
                    'modalities': Counter(),
                    'question_types': Counter(),
                    'gt_answers': Counter()
                }
                
                for item in data:
                    if 'modality' in item:
                        ds_stats['modalities'][item['modality']] += 1
                        stats['modalities'][item['modality']] += 1
                    
                    if 'question_type' in item:
                        ds_stats['question_types'][item['question_type']] += 1
                        stats['question_types'][item['question_type']] += 1
                    
                    if 'gt_answer' in item:
                        ds_stats['gt_answers'][item['gt_answer']] += 1
                        stats['gt_answers'][item['gt_answer']] += 1
                
                stats['datasets'][ds_name] = ds_stats
                stats['total_samples'] += len(data)
                
            except Exception as e:
                print(f"⚠️  Error loading {ds_name}: {e}")
        
        return stats
    
    def filter_by_modality(self, modality: str) -> List[Dict[str, Any]]:
        """Filter datasets by imaging modality"""
        modality_map = {
            'CT': 'CT(Computed Tomography)',
            'MRI': 'MR (Mag-netic Resonance Imaging)',
            'XRAY': 'X-Ray',
            'FUNDUS': 'Fundus Photography',
            'DERMOSCOPY': 'Dermoscopy',
            'OCT': 'OCT (Optical Coherence Tomography',
            'ULTRASOUND': 'ultrasound',
            'MICROSCOPY': 'Microscopy Images'
        }
        
        target_modality = modality_map.get(modality.upper(), modality)
        filtered_data = []
        
        for dataset_name in self.list_available_datasets():
            try:
                data = self.load_dataset(dataset_name)
                for item in data:
                    if item.get('modality') == target_modality:
                        filtered_data.append(item)
            except Exception as e:
                print(f"⚠️  Error processing {dataset_name}: {e}")
        
        return filtered_data
    
    def filter_by_question_type(self, question_type: str) -> List[Dict[str, Any]]:
        """Filter datasets by question type"""
        filtered_data = []
        
        for dataset_name in self.list_available_datasets():
            try:
                data = self.load_dataset(dataset_name)
                for item in data:
                    if item.get('question_type') == question_type:
                        filtered_data.append(item)
            except Exception as e:
                print(f"⚠️  Error processing {dataset_name}: {e}")
        
        return filtered_data
    
    def merge_datasets(self, dataset_names: List[str]) -> List[Dict[str, Any]]:
        """Merge multiple datasets"""
        merged_data = []
        
        for dataset_name in dataset_names:
            try:
                data = self.load_dataset(dataset_name)
                merged_data.extend(data)
                print(f"✅ Added {len(data)} samples from {dataset_name}")
            except Exception as e:
                print(f"⚠️  Error loading {dataset_name}: {e}")
        
        return merged_data
    
    def create_train_test_split(self, data: List[Dict[str, Any]], 
                               test_ratio: float = 0.2, 
                               random_seed: int = 42) -> tuple:
        """Create train/test split"""
        random.seed(random_seed)
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        test_size = int(len(shuffled_data) * test_ratio)
        test_data = shuffled_data[:test_size]
        train_data = shuffled_data[test_size:]
        
        return train_data, test_data
    
    def convert_to_inference_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert to format compatible with inference script"""
        inference_data = []
        
        for item in data:
            # Create question with multiple choice options if available
            question = item.get('question', '')
            
            if 'option_A' in item and item['option_A']:
                options = []
                for option_key in ['option_A', 'option_B', 'option_C', 'option_D']:
                    if option_key in item and item[option_key]:
                        letter = option_key[-1]
                        options.append(f"{letter}) {item[option_key]}")
                
                if options:
                    question += " " + ", ".join(options)
            
            inference_item = {
                'image': item.get('image_path', ''),
                'question': question,
                'question_type': 'multiple_choice' if 'option_A' in item else 'open',
                'ground_truth': item.get('gt_answer', ''),
                'modality': item.get('modality', ''),
                'dataset': item.get('dataset', ''),
                'question_id': item.get('question_id', '')
            }
            
            inference_data.append(inference_item)
        
        return inference_data
    
    def save_data(self, data: List[Dict[str, Any]], output_path: str):
        """Save data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(data)} samples to {output_path}")
    
    def print_stats_report(self, stats: Dict[str, Any]):
        """Print a formatted statistics report"""
        print("\n" + "="*60)
        print("📊 OMNIMEDVQA DATASET STATISTICS")
        print("="*60)
        
        print(f"\n🔢 Total Samples: {stats['total_samples']:,}")
        print(f"📁 Number of Datasets: {len(stats['datasets'])}")
        
        print(f"\n🏥 Top 10 Datasets by Size:")
        sorted_datasets = sorted(stats['datasets'].items(), 
                               key=lambda x: x[1]['samples'], reverse=True)
        for i, (name, ds_stats) in enumerate(sorted_datasets[:10], 1):
            print(f"  {i:2d}. {name:<25} {ds_stats['samples']:>6,} samples")
        
        print(f"\n🔬 Imaging Modalities:")
        for modality, count in stats['modalities'].most_common():
            percentage = (count / stats['total_samples']) * 100
            print(f"  • {modality:<35} {count:>6,} ({percentage:5.1f}%)")
        
        print(f"\n❓ Question Types:")
        for q_type, count in stats['question_types'].most_common():
            percentage = (count / stats['total_samples']) * 100
            print(f"  • {q_type:<35} {count:>6,} ({percentage:5.1f}%)")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="OmniMedVQA Dataset Preparation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dataset statistics
  python prepare_dataset.py --mode stats

  # Filter by CT modality
  python prepare_dataset.py --mode filter --modality CT --output ct_data.json

  # Filter by question type
  python prepare_dataset.py --mode filter --question_type "Disease Diagnosis" --output diagnosis_data.json

  # Merge specific datasets
  python prepare_dataset.py --mode merge --datasets "Chest CT Scan,Diabetic Retinopathy" --output merged_data.json

  # Convert to inference format
  python prepare_dataset.py --mode convert --input merged_data.json --output inference_data.json

  # Create train/test split
  python prepare_dataset.py --mode split --input merged_data.json --test_ratio 0.2 --output_dir splits/
        """
    )
    
    parser.add_argument("--mode", type=str, required=True,
                       choices=["stats", "filter", "merge", "convert", "split", "list"],
                       help="Operation mode")
    
    parser.add_argument("--data_root", type=str, 
                       default="data/OmniMedVQA/OmniMedVQA",
                       help="Root directory of OmniMedVQA dataset")
    
    # Filtering options
    parser.add_argument("--modality", type=str,
                       choices=["CT", "MRI", "XRAY", "FUNDUS", "DERMOSCOPY", "OCT", "ULTRASOUND", "MICROSCOPY"],
                       help="Filter by imaging modality")
    
    parser.add_argument("--question_type", type=str,
                       help="Filter by question type")
    
    parser.add_argument("--datasets", type=str,
                       help="Comma-separated list of dataset names to merge")
    
    # Input/Output options
    parser.add_argument("--input", type=str,
                       help="Input JSON file")
    
    parser.add_argument("--output", type=str,
                       help="Output JSON file")
    
    parser.add_argument("--output_dir", type=str,
                       help="Output directory for splits")
    
    parser.add_argument("--test_ratio", type=float, default=0.2,
                       help="Test set ratio for splitting")
    
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for splitting")
    
    args = parser.parse_args()
    
    try:
        processor = OmniMedVQAProcessor(args.data_root)
        
        if args.mode == "list":
            datasets = processor.list_available_datasets()
            print(f"\n📋 Available Datasets ({len(datasets)}):")
            for i, name in enumerate(datasets, 1):
                print(f"  {i:2d}. {name}")
        
        elif args.mode == "stats":
            dataset_name = None
            if args.datasets:
                dataset_name = args.datasets.split(',')[0]
            
            stats = processor.get_dataset_stats(dataset_name)
            processor.print_stats_report(stats)
        
        elif args.mode == "filter":
            if args.modality:
                data = processor.filter_by_modality(args.modality)
                print(f"🔍 Filtered {len(data)} samples by modality: {args.modality}")
            elif args.question_type:
                data = processor.filter_by_question_type(args.question_type)
                print(f"🔍 Filtered {len(data)} samples by question type: {args.question_type}")
            else:
                raise ValueError("Must specify --modality or --question_type for filtering")
            
            if args.output:
                processor.save_data(data, args.output)
        
        elif args.mode == "merge":
            if not args.datasets:
                raise ValueError("Must specify --datasets for merging")
            
            dataset_names = [name.strip() for name in args.datasets.split(',')]
            data = processor.merge_datasets(dataset_names)
            print(f"🔗 Merged {len(data)} total samples")
            
            if args.output:
                processor.save_data(data, args.output)
        
        elif args.mode == "convert":
            if not args.input:
                raise ValueError("Must specify --input file for conversion")
            
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            converted_data = processor.convert_to_inference_format(data)
            print(f"🔄 Converted {len(converted_data)} samples to inference format")
            
            if args.output:
                processor.save_data(converted_data, args.output)
        
        elif args.mode == "split":
            if not args.input:
                raise ValueError("Must specify --input file for splitting")
            
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            train_data, test_data = processor.create_train_test_split(
                data, args.test_ratio, args.random_seed
            )
            
            print(f"📊 Split into {len(train_data)} train and {len(test_data)} test samples")
            
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                processor.save_data(train_data, os.path.join(args.output_dir, "train.json"))
                processor.save_data(test_data, os.path.join(args.output_dir, "test.json"))
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

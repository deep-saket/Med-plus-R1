#!/usr/bin/env python3
"""
Med-R1 Inference Script
=======================

A comprehensive inference script for the Med-R1 model that supports:
- Local Mac MPS acceleration
- Single image inference
- Batch processing
- Multiple medical imaging modalities
- Flexible output formats

Usage:
    python inference.py --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI \
                       --image_path /path/to/medical_image.jpg \
                       --question "What abnormalities do you observe in this image?"

    python inference.py --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_CT \
                       --batch_file questions.json \
                       --output_file results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import re

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from PIL import Image
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class MedR1Inferencer:
    """
    Med-R1 Model Inferencer with MPS support for Mac
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the Med-R1 inferencer

        Args:
            model_path: Path to the model checkpoint
            device: Device to use ("auto", "cuda", "mps", "cpu")
        """
        self.model_path = model_path
        self.device = self._setup_device(device)

        print(f"🚀 Initializing Med-R1 Inferencer...")
        print(f"📍 Model Path: {model_path}")
        print(f"🖥️  Device: {self.device}")

        # Load model and processor
        self.model = self._load_model()
        self.processor = self._load_processor()

        print("✅ Model loaded successfully!")

    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_model(self):
        """Load the Med-R1 model"""
        try:
            # Try Qwen2.5-VL first (newer version)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"⚠️  Failed to load Qwen2.5-VL, trying Qwen2-VL: {e}")
            # Fallback to Qwen2-VL
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

        # Move to device if not using CUDA (which handles device_map automatically)
        if self.device != "cuda":
            model = model.to(self.device)

        model.eval()
        return model

    def _load_processor(self):
        """Load the processor"""
        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        processor.tokenizer.padding_side = "left"
        return processor

    def _validate_image_path(self, image_path: str) -> bool:
        """Validate if image path exists and is a valid image"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False

        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            print(f"❌ Invalid image file {image_path}: {e}")
            return False

    def _format_question(self, question: str, question_type: str = "open") -> str:
        """
        Format question based on type

        Args:
            question: The input question
            question_type: Type of question ("open", "multiple_choice", "yes_no")
        """
        if question_type == "multiple_choice":
            return f"{question} Provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags."
        elif question_type == "yes_no":
            return f"{question} Answer with Yes or No inside <answer>...</answer> tags."
        else:
            return question

    def _extract_answer(self, output_text: str) -> Optional[str]:
        """Extract answer from model output"""
        # Try to find answer within <answer> tags
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(answer_pattern, output_text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # If no tags found, return the full output
        return output_text.strip()

    def infer_single(
        self,
        image_path: str,
        question: str,
        question_type: str = "open",
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Perform inference on a single image-question pair

        Args:
            image_path: Path to the image file
            question: Question about the image
            question_type: Type of question ("open", "multiple_choice", "yes_no")
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature

        Returns:
            Dictionary containing inference results
        """
        if not self._validate_image_path(image_path):
            return {
                "success": False,
                "error": f"Invalid image path: {image_path}"
            }

        start_time = time.time()

        # Format the question
        formatted_question = self._format_question(question, question_type)

        # Prepare message
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": formatted_question}
            ]
        }]

        try:
            # Process input
            text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(message)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = inputs.to(self.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Extract answer
            extracted_answer = self._extract_answer(output_text)

            inference_time = time.time() - start_time

            return {
                "success": True,
                "image_path": image_path,
                "question": question,
                "formatted_question": formatted_question,
                "raw_output": output_text,
                "extracted_answer": extracted_answer,
                "inference_time": inference_time,
                "device": str(self.device),
                "model_path": self.model_path
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path,
                "question": question
            }

    def infer_batch(
        self,
        questions_data: List[Dict[str, Any]],
        batch_size: int = 1,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform batch inference

        Args:
            questions_data: List of dictionaries with 'image' and 'question' keys
            batch_size: Batch size for processing
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature

        Returns:
            List of inference results
        """
        results = []

        print(f"🔄 Processing {len(questions_data)} questions in batches of {batch_size}...")

        for i in tqdm(range(0, len(questions_data), batch_size)):
            batch_data = questions_data[i:i + batch_size]

            # Process each item in the batch individually for simplicity
            for item in batch_data:
                result = self.infer_single(
                    image_path=item['image'],
                    question=item['question'],
                    question_type=item.get('question_type', 'open'),
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature
                )
                results.append(result)

        return results


def load_questions_file(file_path: str) -> List[Dict[str, Any]]:
    """Load questions from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'questions' in data:
        return data['questions']
    else:
        raise ValueError("Unsupported file format. Expected list or dict with 'questions' key.")


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file"""
    # Calculate accuracy if ground truth is available
    total_count = len(results)
    correct_count = 0

    for result in results:
        if result.get('success') and 'ground_truth' in result:
            if result.get('extracted_answer') == result.get('ground_truth'):
                correct_count += 1

    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

    output_data = {
        'total_questions': total_count,
        'successful_inferences': sum(1 for r in results if r.get('success')),
        'accuracy': accuracy if correct_count > 0 else None,
        'results': results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {output_path}")
    if accuracy is not None:
        print(f"📊 Accuracy: {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Med-R1 Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python inference.py --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI \\
                     --image_path sample_xray.jpg \\
                     --question "What abnormalities do you observe?"

  # Batch processing
  python inference.py --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_CT \\
                     --batch_file questions.json \\
                     --output_file results.json \\
                     --batch_size 2

  # Multiple choice question
  python inference.py --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI \\
                     --image_path medical_image.jpg \\
                     --question "What is the primary finding? A) Normal B) Abnormal C) Unclear" \\
                     --question_type multiple_choice
        """
    )

    # Model and device arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the Med-R1 model checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="Device to use for inference")

    # Single inference arguments
    parser.add_argument("--image_path", type=str,
                       help="Path to a single image for inference")
    parser.add_argument("--question", type=str,
                       help="Question about the image")
    parser.add_argument("--question_type", type=str, default="open",
                       choices=["open", "multiple_choice", "yes_no"],
                       help="Type of question")

    # Batch processing arguments
    parser.add_argument("--batch_file", type=str,
                       help="JSON file containing batch questions")
    parser.add_argument("--output_file", type=str,
                       help="Output file for batch results")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true",
                       help="Use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")

    args = parser.parse_args()

    # Validate arguments
    if not args.image_path and not args.batch_file:
        parser.error("Either --image_path or --batch_file must be provided")

    if args.image_path and not args.question:
        parser.error("--question must be provided when using --image_path")

    if args.batch_file and not args.output_file:
        parser.error("--output_file must be provided when using --batch_file")

    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        print("Available model paths:")
        checkpoint_dir = Path("checkpoints/Med-R1/Qwen_2.5_3B_nothink")
        if checkpoint_dir.exists():
            for model_dir in checkpoint_dir.iterdir():
                if model_dir.is_dir():
                    print(f"  - {model_dir}")
        sys.exit(1)

    try:
        # Initialize inferencer
        inferencer = MedR1Inferencer(args.model_path, args.device)

        if args.image_path:
            # Single image inference
            print(f"\n🔍 Analyzing image: {args.image_path}")
            print(f"❓ Question: {args.question}")
            print(f"📝 Question Type: {args.question_type}")

            result = inferencer.infer_single(
                image_path=args.image_path,
                question=args.question,
                question_type=args.question_type,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature
            )

            if result['success']:
                print(f"\n✅ Inference completed in {result['inference_time']:.2f}s")
                print(f"🤖 Answer: {result['extracted_answer']}")
                print(f"\n📄 Full Response:\n{result['raw_output']}")
            else:
                print(f"\n❌ Inference failed: {result['error']}")

        else:
            # Batch processing
            print(f"\n📂 Loading questions from: {args.batch_file}")
            questions_data = load_questions_file(args.batch_file)

            results = inferencer.infer_batch(
                questions_data=questions_data,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature
            )

            save_results(results, args.output_file)

            # Print summary
            successful = sum(1 for r in results if r.get('success'))
            print(f"\n📊 Summary:")
            print(f"   Total questions: {len(results)}")
            print(f"   Successful: {successful}")
            print(f"   Failed: {len(results) - successful}")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

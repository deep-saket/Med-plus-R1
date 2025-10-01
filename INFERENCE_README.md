# Med-R1 Inference Script

A comprehensive inference script for the Med-R1 model with Mac MPS support, designed for medical image analysis across multiple modalities.

## Features

- ✅ **Mac MPS Support**: Optimized for Apple Silicon (M1/M2/M3) with Metal Performance Shaders
- 🖼️ **Multi-Modal Support**: Works with CT, MRI, X-Ray, Fundus, Dermoscopy, Ultrasound, OCT, and Microscopy images
- 🔄 **Flexible Processing**: Single image inference or batch processing
- 📊 **Multiple Question Types**: Open-ended, multiple choice, and yes/no questions
- 💾 **Comprehensive Output**: Detailed results with timing and accuracy metrics

## Installation

1. **Install Dependencies**:
```bash
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install qwen-vl-utils
pip install pillow tqdm
```

2. **For Mac MPS Support** (Apple Silicon):
```bash
# Ensure you have the latest PyTorch with MPS support
pip install --upgrade torch torchvision torchaudio
```

## Available Models

The Med-R1 repository includes specialized models for different imaging modalities:

```
checkpoints/Med-R1/Qwen_2.5_3B_nothink/
├── VQA_AI/          # General medical AI
├── VQA_CT/          # CT scans
├── VQA_Fundus/      # Fundus photography
├── VQA_LG/          # Large general model
├── VQA_MR/          # MRI scans
└── VQA_MRI/         # MRI specialized
```

## Usage

### Single Image Inference

```bash
# Basic usage
python inference.py \
    --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI \
    --image_path /path/to/medical_image.jpg \
    --question "What abnormalities do you observe in this image?"

# Multiple choice question
python inference.py \
    --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_CT \
    --image_path ct_scan.jpg \
    --question "Is there a mass lesion? A) Yes B) No C) Unclear" \
    --question_type multiple_choice

# Yes/No question
python inference.py \
    --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_MR \
    --image_path brain_mri.jpg \
    --question "Does this MRI show signs of stroke?" \
    --question_type yes_no
```

### Batch Processing

```bash
# Process multiple images
python inference.py \
    --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI \
    --batch_file sample_questions.json \
    --output_file results.json \
    --batch_size 2
```

### Advanced Options

```bash
# Use sampling for more diverse outputs
python inference.py \
    --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI \
    --image_path medical_image.jpg \
    --question "Describe the findings" \
    --do_sample \
    --temperature 0.7 \
    --max_new_tokens 1024

# Force specific device
python inference.py \
    --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI \
    --image_path medical_image.jpg \
    --question "What do you see?" \
    --device mps  # or cuda, cpu
```

## Input Format

### Single Image
- `--image_path`: Path to the medical image
- `--question`: Your question about the image
- `--question_type`: Type of question (open, multiple_choice, yes_no)

### Batch File Format (JSON)
```json
[
  {
    "image": "path/to/image1.jpg",
    "question": "What abnormalities do you observe?",
    "question_type": "open"
  },
  {
    "image": "path/to/image2.jpg",
    "question": "Is this normal? A) Yes B) No",
    "question_type": "multiple_choice",
    "ground_truth": "B"
  }
]
```

## Output Format

### Single Image Output
```
🔍 Analyzing image: chest_xray.jpg
❓ Question: What abnormalities do you observe?
📝 Question Type: open

✅ Inference completed in 2.34s
🤖 Answer: The chest X-ray shows bilateral lower lobe opacity consistent with pneumonia.

📄 Full Response:
Looking at this chest X-ray, I can observe bilateral lower lobe opacities that appear to be consistent with pneumonia. The heart size appears normal, and there is no evidence of pleural effusion or pneumothorax.
```

### Batch Output (JSON)
```json
{
  "total_questions": 5,
  "successful_inferences": 5,
  "accuracy": 80.0,
  "results": [
    {
      "success": true,
      "image_path": "sample_images/chest_xray.jpg",
      "question": "What abnormalities do you observe?",
      "raw_output": "The chest X-ray shows...",
      "extracted_answer": "bilateral lower lobe opacity",
      "inference_time": 2.34,
      "device": "mps"
    }
  ]
}
```

## Performance Optimization

### Mac MPS (Apple Silicon)
- **Automatic Detection**: The script automatically detects and uses MPS when available
- **Memory Efficient**: Uses bfloat16 precision to reduce memory usage
- **Optimized Loading**: Smart model loading for different hardware configurations

### Memory Management
```bash
# For large images or limited memory
python inference.py \
    --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI \
    --batch_file questions.json \
    --output_file results.json \
    --batch_size 1  # Reduce batch size
```

## Question Types

### Open-ended Questions
- General medical analysis
- Detailed descriptions
- Diagnostic reasoning

### Multiple Choice
- Automatically formats with answer tags
- Extracts single-letter answers
- Supports accuracy calculation

### Yes/No Questions
- Binary medical decisions
- Quick screening questions
- Presence/absence of findings

## Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)

## Troubleshooting

### Common Issues

1. **Model Not Found**:
```bash
❌ Model path not found: checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI
```
**Solution**: Check available models and use correct path

2. **MPS Not Available**:
```bash
⚠️  MPS not available, falling back to CPU
```
**Solution**: Update PyTorch or use `--device cpu`

3. **Out of Memory**:
```bash
RuntimeError: MPS backend out of memory
```
**Solution**: Reduce batch size or use `--device cpu`

4. **Image Loading Error**:
```bash
❌ Invalid image file: corrupted_image.jpg
```
**Solution**: Check image file integrity and format

### Performance Tips

- Use MPS for Apple Silicon Macs (M1/M2/M3)
- Start with batch_size=1 and increase gradually
- Use appropriate model for your image type (CT, MRI, etc.)
- For faster inference, use `--do_sample False` (greedy decoding)

## Model Selection Guide

| Image Type | Recommended Model | Use Case |
|------------|------------------|-----------|
| CT Scans | VQA_CT | Computed tomography analysis |
| MRI | VQA_MR, VQA_MRI | Magnetic resonance imaging |
| X-Ray | VQA_AI | General radiography |
| Fundus Photos | VQA_Fundus | Retinal imaging |
| General | VQA_LG | Multi-modal analysis |

## Examples

See `sample_questions.json` for example input format and various question types.

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with MPS support for Mac)
- transformers >= 4.30.0
- qwen-vl-utils
- PIL (Pillow)
- tqdm

For complete requirements, see `src/requirements.txt`.

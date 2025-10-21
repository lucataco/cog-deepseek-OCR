# DeepSeek-OCR Cog Implementation

This is a [Cog](https://github.com/replicate/cog) implementation of [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR), a powerful vision-language model for Optical Character Recognition (OCR) that can extract text from documents, charts, tables, and other images.

## Features

- **High-Quality OCR**: Extract text from various document types including PDFs, scanned documents, charts, and tables
- **Grounded OCR**: Optional bounding box detection for extracted text regions
- **Markdown Output**: Convert documents to structured markdown format
- **Multiple Task Modes**: Pre-configured settings for different use cases (Tiny, Small, Base, Large, Gundam)
- **GPU Accelerated**: Uses CUDA 12.4 with Flash Attention 2 for efficient inference
- **Offline Mode**: Runs completely offline with local model checkpoints

## Prerequisites

- [Cog](https://github.com/replicate/cog) installed
- NVIDIA GPU with CUDA support
- Docker

## Installation

1. Clone this repository:
```bash
git clone https://github.com/lucataco/cog-deepseek-OCR.git
cd deepseek-ocr
```

2. Build the Cog image:
```bash
cog build
```

**Note**: Model weights (~20GB) are automatically downloaded from Replicate's CDN on first run using `pget`, a fast parallel downloader. The weights are cached in the `checkpoints/` directory for subsequent runs.

## Usage

### Basic OCR

Extract text from an image:

```bash
cog predict -i image=@your-document.jpg
```

### Custom Prompt

Use a custom prompt for specific extraction tasks:

```bash
cog predict -i image=@document.jpg -i prompt="<image>\nExtract all table data and convert to markdown."
```

### Grounded OCR with Bounding Boxes

Get text with bounding box coordinates:

```bash
cog predict -i image=@document.jpg -i prompt="<image>\n<|grounding|>Convert the document to markdown."
```

### Task Modes

Choose from pre-configured task modes optimized for different scenarios:

- **Gundam (Recommended)**: `base_size=1024, image_size=640, crop_mode=True` - Best balance of speed and accuracy
- **Tiny**: `base_size=512, image_size=512, crop_mode=False` - Fastest, lower quality
- **Small**: `base_size=640, image_size=640, crop_mode=False` - Fast with decent quality
- **Base**: `base_size=1024, image_size=1024, crop_mode=False` - Good quality
- **Large**: `base_size=1280, image_size=1280, crop_mode=False` - Best quality, slower
- **Custom**: Specify your own `base_size`, `image_size`, and `crop_mode`

Example:

```bash
cog predict -i image=@document.jpg -i task_mode="Large"
```

### Advanced Options

```bash
cog predict \
  -i image=@document.jpg \
  -i task_mode="Custom" \
  -i base_size=1024 \
  -i image_size=640 \
  -i crop_mode=true \
  -i save_visualization=true \
  -i test_compress=true
```

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | Path | Required | Input image to perform OCR on |
| `prompt` | String | `"<image>\n<|grounding|>Convert the document to markdown."` | Custom prompt for the model |
| `task_mode` | String | `"Gundam (Recommended)"` | Preset configuration: Tiny, Small, Base, Large, Gundam, or Custom |
| `base_size` | Integer | `1024` | Base size for image processing (512-2048, only for Custom mode) |
| `image_size` | Integer | `640` | Target image size for vision encoder (512-2048, only for Custom mode) |
| `crop_mode` | Boolean | `true` | Enable crop mode for large documents (only for Custom mode) |
| `save_visualization` | Boolean | `false` | Save visualization results with bounding boxes |
| `test_compress` | Boolean | `true` | Enable compression testing for visual token optimization |

## Prompt Templates

### Standard OCR
```
<image>
Extract the text in the image.
```

### Grounded OCR with Bounding Boxes
```
<image>
<|grounding|>Convert the document to markdown.
```

### Table Extraction
```
<image>
Parse the table and convert it to markdown format.
```

### Chart/Figure Analysis
```
<image>
Parse the figure and describe its contents.
```

## Model Architecture

- **Base Model**: DeepSeek-V2 (7B parameters)
- **Vision Encoder**: Combination of SAM ViT-B and CLIP-L
- **Attention**: Flash Attention 2 for efficient inference
- **Precision**: bfloat16 for optimal GPU performance

## Performance

The model uses dynamic preprocessing with crop mode to handle large documents efficiently:

- **Gundam Mode**: ~760 visual tokens for a typical document
- **Compression Ratio**: Typically 0.6-0.8 for text-heavy documents
- **Max Tokens**: Up to 8192 new tokens for output

## Development

### Project Structure

```
deepseek-ocr/
├── cog.yaml              # Cog configuration
├── predict.py            # Prediction interface
├── requirements.txt      # Python dependencies
├── checkpoints/          # Model checkpoints directory
│   ├── __init__.py       # Package initialization
│   ├── modeling_deepseekocr.py
│   ├── modeling_deepseekv2.py
│   ├── configuration_deepseek_v2.py
│   ├── deepencoder.py
│   └── conversation.py
└── README.md
```

### Key Files

- **[predict.py](predict.py)**: Main Cog predictor interface with automatic weight downloading, warning suppression, and parameter handling
- **[checkpoints/modeling_deepseekocr.py](checkpoints/modeling_deepseekocr.py)**: Core model implementation with custom inference logic
- **[cog.yaml](cog.yaml)**: Cog configuration specifying the runtime environment and pget installation
- **[requirements.txt](requirements.txt)**: Python package dependencies

### Automatic Weight Management

The implementation uses [pget](https://github.com/replicate/pget), a fast parallel file downloader, to automatically fetch model weights on first run:

- Weights are downloaded from: `https://weights.replicate.delivery/default/deepseek-ai/DeepSeek-OCR/model.tar`
- Downloaded to: `checkpoints/` directory
- Cached for subsequent runs
- Download time: ~3-5 minutes on a typical connection

### Running Tests

```bash
# Test with a sample image
cog predict -i image=@demo.jpg

# Test different modes
cog predict -i image=@demo.jpg -i task_mode="Tiny"
cog predict -i image=@demo.jpg -i task_mode="Large"
```

## Troubleshooting

### Model Loading Issues

If you encounter model loading errors:

1. **First Run - Automatic Download**: On the first prediction, the model weights (~20GB) will be automatically downloaded from Replicate's CDN. This may take several minutes depending on your connection speed.

2. **Verify checkpoints**: After download, ensure checkpoints are present:
   ```bash
   ls -la checkpoints/
   # Should contain: config.json, model safetensors files, and Python files
   ```

3. **Manual Download**: If automatic download fails, you can manually download from Hugging Face:
   ```bash
   pip install huggingface-hub
   huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir checkpoints
   ```

4. Verify the `__init__.py` file exists in the checkpoints directory

### GPU Memory Issues

If you run out of GPU memory:

- Use a smaller `task_mode` like "Tiny" or "Small"
- Reduce `base_size` and `image_size` in Custom mode
- Disable `crop_mode` for smaller documents

### Warning Messages

The implementation includes comprehensive warning suppression for known transformers library warnings that cannot be fixed at the application level. If you see unexpected warnings, they may indicate actual issues that need attention.

## Citation

If you use this implementation, please cite the original DeepSeek-OCR paper:

```bibtex
@article{deepseek-ocr,
  title={DeepSeek-OCR: A Vision-Language Model for Optical Character Recognition},
  author={DeepSeek AI},
  year={2024},
  url={https://huggingface.co/deepseek-ai/DeepSeek-OCR}
}
```

## License

This implementation follows the license of the original DeepSeek-OCR model. Please refer to the [official repository](https://huggingface.co/deepseek-ai/DeepSeek-OCR) for licensing details.

## Acknowledgments

- [DeepSeek AI](https://www.deepseek.com/) for the original model
- [Replicate](https://replicate.com/) for the Cog framework
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library

## Support

For issues related to:
- **This Cog implementation**: Open an issue in this repository
- **The original model**: Visit the [DeepSeek-OCR repository](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **Cog framework**: Check the [Cog documentation](https://github.com/replicate/cog)

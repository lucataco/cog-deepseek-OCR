# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import torch
import tempfile
import warnings
import logging
from cog import BasePredictor, Input, Path
from transformers import AutoTokenizer

# Import the custom DeepSeek-OCR model class
from checkpoints import DeepseekOCRForCausalLM

# Suppress all warnings from transformers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Set transformers logging to error only
logging.getLogger('transformers').setLevel(logging.ERROR)

# Filter out known warnings from transformers that we cannot fix
# These warnings come from the underlying transformers library and model architecture
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the DeepSeek-OCR model into memory to make running multiple predictions efficient"""
        print("Loading DeepSeek-OCR model from local checkpoints...")

        # Configure for offline mode - no internet access needed
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Load model from local checkpoints directory
        model_path = "./checkpoints"

        # Load tokenizer in offline mode
        # Using local_files_only to prevent any network calls
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        # Configure tokenizer pad token to avoid warnings
        # Set pad token to eos token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model directly using the custom class from local files
        # No trust_remote_code needed - we're importing the code directly
        # Suppress warnings during model loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = DeepseekOCRForCausalLM.from_pretrained(
                model_path,
                _attn_implementation='flash_attention_2',
                local_files_only=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0"
            )

        # Set to eval mode
        self.model = self.model.eval()

        print("Model loaded successfully!")

    def predict(
        self,
        image: Path = Input(description="Input image to perform OCR on (supports documents, charts, tables, etc.)"),
        prompt: str = Input(
            description="Custom prompt for the model. Use '<image>\\n' prefix for standard OCR or '<image>\\n<|grounding|>' for grounded OCR with bounding boxes",
            default="<image>\n<|grounding|>Convert the document to markdown. "
        ),
        task_mode: str = Input(
            description="Preset configuration for different use cases",
            choices=["Gundam (Recommended)", "Tiny", "Small", "Base", "Large", "Custom"],
            default="Gundam (Recommended)"
        ),
        base_size: int = Input(
            description="Base size for image processing (only used when task_mode is 'Custom')",
            default=1024,
            ge=512,
            le=2048
        ),
        image_size: int = Input(
            description="Target image size for vision encoder (only used when task_mode is 'Custom')",
            default=640,
            ge=512,
            le=2048
        ),
        crop_mode: bool = Input(
            description="Enable crop mode for better handling of large documents (only used when task_mode is 'Custom')",
            default=True
        ),
        save_visualization: bool = Input(
            description="Save visualization results with bounding boxes and detected regions",
            default=False
        ),
        test_compress: bool = Input(
            description="Enable compression testing for visual token optimization",
            default=True
        ),
    ) -> str:
        """
        Run OCR inference on the input image using DeepSeek-OCR.

        Returns the extracted text/markdown content from the image.
        """

        # Configure parameters based on task_mode
        if task_mode == "Tiny":
            base_size, image_size, crop_mode = 512, 512, False
        elif task_mode == "Small":
            base_size, image_size, crop_mode = 640, 640, False
        elif task_mode == "Base":
            base_size, image_size, crop_mode = 1024, 1024, False
        elif task_mode == "Large":
            base_size, image_size, crop_mode = 1280, 1280, False
        elif task_mode == "Gundam (Recommended)":
            base_size, image_size, crop_mode = 1024, 640, True
        # else: Custom mode - use provided parameters

        print(f"Running inference with mode: {task_mode}")
        print(f"Parameters: base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}")

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as output_dir:
            # Run inference using the model's infer method
            # Suppress warnings during inference
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=str(image),
                    output_path=output_dir,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    test_compress=test_compress,
                    save_results=save_visualization
                )

            # Return the extracted text/markdown
            if isinstance(result, str):
                return result
            elif isinstance(result, dict) and 'text' in result:
                return result['text']
            else:
                return str(result)

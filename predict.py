# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import torch
import subprocess
import tempfile
import warnings
from cog import BasePredictor, Input, Path
from transformers import AutoTokenizer

# Suppress all warnings from transformers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

MODEL_PATH = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/deepseek-ai/DeepSeek-OCR/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the DeepSeek-OCR model into memory to make running multiple predictions efficient"""
        # Download weights
        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)

        from checkpoints import DeepseekOCRForCausalLM
            
        print("Loading DeepSeek-OCR model from local checkpoints...")
        # Configure for offline mode - no internet access needed
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Load tokenizer in offline mode
        # Using local_files_only to prevent any network calls
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
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
                MODEL_PATH,
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

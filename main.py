import os
import shutil
import torch
from typing import List
from model_inference import main_sample
from contextlib import asynccontextmanager
from model_pipeline import ModelPipeline
from fastapi import FastAPI
from pydantic import BaseModel
from repo.fetch_styles import fetch_style_images
import argparse


class HandwritingRequest(BaseModel):
    uname: str
    texts: List[str]
    style_code: str


# Will be updated by CLI args
MODEL_CONFIG = {
    "unet_path": None,
    "vae_folder_path": None,
    "style_encoder_path": None
}

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    print("Loading models...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = ModelPipeline(
        unet_path=MODEL_CONFIG["unet_path"],
        vae_folder_path=MODEL_CONFIG["vae_folder_path"],
        style_encoder_path=MODEL_CONFIG["style_encoder_path"],
        device=device
    )

    print("Models loaded successfully.")
    yield

    print("Cleaning temporary files...")
    if os.path.exists("out/"):
        for f in os.listdir("out/"):
            os.remove(os.path.join("out/", f))
        os.rmdir("out/")
    print("Shutdown completed.")


app = FastAPI(title="Strox Handwriting Generation API", lifespan=lifespan)

# health check endpoint
@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.post("/generate/")
async def generate_handwriting(payload: HandwritingRequest):
    global pipeline
    print("RECEIVED:", payload.texts)
    print("RECEIVED:", payload.style_code)
    
    uname = payload.uname
    texts = payload.texts
    style_code = payload.style_code
    
    # Fetch style images
    style_images = fetch_style_images(style_code)
    
    style_image_paths = []
    for idx, img in enumerate(style_images):
        file_path = f"temp_{style_code}_{idx}.png"
        img.save(file_path)
        style_image_paths.append(file_path)
    
    output_path = main_sample(
        model_pipeline=pipeline,
        text_list=texts,
        style_refs=style_image_paths,
        uname=uname,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # check and remove the tmp/uname folder
    base_tmp = "/tmp"
    user_tmp = os.path.join(base_tmp, uname)

    if os.path.isdir(user_tmp):
        shutil.rmtree(user_tmp)

    os.makedirs(user_tmp, exist_ok=True)
    
    return {
        "message": "Handwriting generation completed.",
        "output_path": output_path
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run handwriting generation API")

    parser.add_argument(
        "--unet",
        required=True,
        help="Path to UNet model (.pth)"
    )
    parser.add_argument(
        "--vae",
        required=True,
        help="Path to VAE folder"
    )
    parser.add_argument(
        "--style",
        required=True,
        help="Path to style encoder model (.pth)"
    )

    parser.add_argument(
        "--port",
        default=8080,
        type=int,
        help="Port to run FastAPI server"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Populate model paths globally
    MODEL_CONFIG["unet_path"] = args.unet
    MODEL_CONFIG["vae_folder_path"] = args.vae
    MODEL_CONFIG["style_encoder_path"] = args.style

    import uvicorn
    port = int(os.environ.get("PORT", args.port))
    uvicorn.run(app, host="0.0.0.0", port=port)

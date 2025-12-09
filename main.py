import os
import torch
from typing import List
from model_inference import main_sample
from contextlib import asynccontextmanager
from model_pipeline import ModelPipeline
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from repo.fetch_styles import fetch_style_images


app = FastAPI(title="Strox Handwrting Generation API")

class HandwritingRequest(BaseModel):
    uname: str
    texts: List[str]
    style_code: str


pipeline = None

@asynccontextmanager
def load_models():
    global pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = ModelPipeline(
        unet_path="unet_model.pth",
        vae_folder_path="vae_model/",
        style_encoder_path="mobilenetv3_iam_style_MIII-P5.pth",
        device=device
    )
    yield
    # if the out/ folder path exist delete that
    if os.path.exists("out/"):
        for f in os.listdir("out/"):
            os.remove(os.path.join("out/", f))
        os.rmdir("out/")


@app.post("/generate/")
async def generate_handwriting(payload: HandwritingRequest):
    global pipeline
    
    # Unpack values
    uname = payload.uname
    texts = payload.texts
    style_code = payload.style_code
    
    # Fetch style images using style_code
    style_images = fetch_style_images(style_code)
    
    style_image_paths = []
    for img in style_images:
        file_path = f"temp_{style_code}_{style_images.index(img)}.png"
        img.save(file_path)
        style_image_paths.append(file_path)
    
    # Run your main function
    output_path = main_sample(
        model_pipeline=pipeline,
        text_list=texts,
        style_refs=style_image_paths,
        uname=uname
    )
    
    return {
        "message": "Handwriting generation completed.",
        "output_path": output_path
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

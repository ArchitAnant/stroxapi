import torch
from typing import List
from model_inference import main_sample
from model_pipeline import ModelPipeline
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Strox Handwrting Generation API")

pipeline = None

@app.lifecycle_event("startup")
def load_models():
    global pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = ModelPipeline(
        unet_path="unet_model.pth",
        vae_folder_path="vae_model/",
        style_encoder_path="mobilenetv3_iam_style_MIII-P5.pth",
        device=device
    )

@app.post("/generate/")
async def generate_handwriting(texts: List[str], style_images: List[UploadFile]):
    global pipeline
    style_image_paths = []
    for img in style_images:
        contents = await img.read()
        with open(f"temp_{img.filename}", "wb") as f:
            f.write(contents)
        style_image_paths.append(f"temp_{img.filename}")
    
    output_path = "generated_handwriting/"
    main_sample(
        model_pipeline=pipeline,
        text_list=texts,
        style_refs=style_image_paths,
        out=output_path
    )
    return {"message": "Handwriting generation completed.", "output_path": output_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
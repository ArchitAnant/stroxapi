FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y \
    git wget libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy app code
COPY . .

# Make directories for your models
RUN mkdir -p handwriting_vae/vae \
    && mkdir -p handwriting_vae/schedular

# Download UNET model
RUN wget https://huggingface.co/konnik/DiffusionPen/resolve/main/diffusionpen_iam_model_path/models/ema_ckpt.pt \
    -O ema_ckpt.pt

# Download VAE files
RUN wget https://huggingface.co/ari-archit/stable-diffusion-handwriting/resolve/main/diffusion_pytorch_model.safetensors \
    -O handwriting_vae/vae/diffusion_pytorch_model.safetensors

RUN wget https://huggingface.co/ari-archit/stable-diffusion-handwriting/resolve/main/config.json \
    -O handwriting_vae/vae/config.json

# Download style model
RUN pip install gdown

RUN gdown 1V5Pmw9j7vkzgLbXsW8v0b41uCQ0bWJXx \
    -O mobilenetv3_iam_style_MIII-P5.pth

# Install Python libs
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# Run your server exactly like you do in Colab
CMD ["python", "main.py", \
     "--unet", "ema_ckpt.pt", \
     "--vae", "handwriting_vae/", \
     "--style", "mobilenetv3_iam_style_MIII-P5.pth"]

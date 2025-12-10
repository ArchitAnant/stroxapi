from .client import BlobClient
from PIL import Image
import io

def fetch_style_images(style_code):
    """
    fetch style images from azure blob storage of the continer 'styles'
    return: list of PIL images
    """
    style_image = []
    blob_client = BlobClient()
    container_client = blob_client.styles_container_client
    blob_list = container_client.list_blobs(name_starts_with=style_code)
    for blob in blob_list:
        blob_data = container_client.download_blob(blob.name)
        image_data = blob_data.readall()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        style_image.append(image)
    return style_image
    
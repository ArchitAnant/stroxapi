from repo.client import BlobClient
from repo.fetch_styles import fetch_style_images
import io

def test_fetch_style_images():
    # upload a dummy style image to blob storage first
    blob_client = BlobClient()
    from PIL import Image
    img = Image.new('RGB', (100, 100), color = 'green')
    blob_name = "test_style_image.png"
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        data = output.getvalue()
        blob_client.styles_container_client.upload_blob(name=blob_name, data=data, overwrite=True)
    
    # now fetch the style image using fetch_styles
    style_images = fetch_style_images("test_style_image")
    assert len(style_images) == 1
    fetched_img = style_images[0]
    assert fetched_img.size == (100, 100)
    assert fetched_img.mode == 'RGB'
    # clean up by deleting the uploaded style image
    try:
        blob_client.styles_container_client.delete_blob(blob_name)
    except Exception as e:
        print(f"Error deleting blob {blob_name}: {e}")

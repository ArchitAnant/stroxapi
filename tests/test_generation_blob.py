from repo.upload_main import upload, remove
from repo.client import BlobClient


def test_upload_generation_blob():
    blob_client = BlobClient()
    # create a dummy image list
    from PIL import Image
    img1 = Image.new('RGB', (100, 100), color = 'red')
    img2 = Image.new('RGB', (100, 100), color = 'blue')
    img_list = [img1, img2]
    node_code = "testuser_20240101_000000_1234"
    assert upload(img_list, node_code, blob_client) == True
    assert remove(node_code, 2, blob_client) == True

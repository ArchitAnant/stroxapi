from datetime import datetime
from random import randint
from typing import List
from PIL import Image
import io

def generate_node_code(uname):
    node_code = f"{uname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{randint(1000,9999)}"
    return node_code

def upload(img_list:List[Image.Image], node_code,blob_client):
    base_upload_url = None
    for idx, img in enumerate(img_list):
        blob_name = f"{node_code}_image_{idx+1}.png"
        with io.BytesIO() as output:
            img.save(output, format="PNG")
            data = output.getvalue()
            blob_client.generation_container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        blob_url = f"https://{blob_client.blob_service_client.account_name}.blob.core.windows.net/{blob_client.generation_container_client.container_name}/{blob_name}"
        base_upload_url = blob_url
    if base_upload_url is None:
        return False
    
    return True


def remove(node_code,n,blob_client):
    for idx in range(1,n+1):
        blob_name = f"{node_code}_image_{idx}.png"
        try:
            blob_client.generation_container_client.delete_blob(blob_name)
        except Exception as e:
            print(f"Error deleting blob {blob_name}: {e}")
    return True
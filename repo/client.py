from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os

load_dotenv()

class BlobClient:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("BLOB_CONTAINER_NAME")
    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

# Path to your local image
# local_file_path = "sample.jpg"

# # Blob name (what it will be called in Azure)
# blob_name = "my_uploaded_image.jpg"

# # Upload
# with open(local_file_path, "rb") as data:
#     container_client.upload_blob(name=blob_name, data=data, overwrite=True)

# print("Upload complete!")

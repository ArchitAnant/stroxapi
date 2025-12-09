from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import json
import logging
import os

load_dotenv()

class BlobClient:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("BLOB_CONTAINER_NAME")
    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

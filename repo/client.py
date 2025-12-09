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
    generation_container_name = os.getenv("GENERATION_BLOB_CONTAINER_NAME")
    styles_container_name = os.getenv("STYLES_BLOB_CONTAINER_NAME")
    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.generation_container_client = self.blob_service_client.get_container_client(self.generation_container_name)
        self.styles_container_client = self.blob_service_client.get_container_client(self.styles_container_name)
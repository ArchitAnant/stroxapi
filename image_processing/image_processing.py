from PIL import Image
import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')

def get_ocr(image_path):
  with open(image_path, 'rb') as f:
      image_bytes = f.read()

  client = genai.Client(api_key=API_KEY)
  response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
      types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg',
      ),
      'OCR this and just return the text in the image as response'
    ]
  )

  print(response.text)
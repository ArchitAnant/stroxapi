from transformers import CanineTokenizer, CanineModel

tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
model = CanineModel.from_pretrained("google/canine-c")  # no fine-tuning needed

text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
embeddings = model(**inputs).last_hidden_state  # feed this to your VAE decoder

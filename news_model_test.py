from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

model_name = "yiyanghkust/finbert-tone"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = classifier("Zomato is reducing their price")
print(result)

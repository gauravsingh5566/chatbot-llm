# chatbot-backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize FastAPI app
app = FastAPI()

# Load the Llama model and tokenizer (or another model)
model_name = "Llama-3.3-model"  # Replace with the actual model path or Hugging Face model ID
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the input structure for the user query
class Query(BaseModel):
    query: str

# Define a route to interact with the chatbot
@app.post("/chat")
async def chat(query: Query):
    # Tokenize the input query
    inputs = tokenizer(query.query, return_tensors="pt")
    
    # Generate a response from the model
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}

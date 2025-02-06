import os
import torch
from flask import Flask, request, jsonify
from transformers import LlamaTokenizer, LlamaForCausalLM
from torch import nn

# Load the tokenizer
model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.3-70B-Instruct/")
tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=False)

# Load the model manually
class LlamaForCausalLMCustom(nn.Module):
    def __init__(self):
        super(LlamaForCausalLMCustom, self).__init__()
        # Define your model structure here

    def forward(self, input_ids):
        # Define the forward pass logic here
        return model_output

# Load model weights from the checkpoint files
model = LlamaForCausalLMCustom()

checkpoint_files = [
    "consolidated.00.pth", "consolidated.01.pth", "consolidated.02.pth", 
    "consolidated.03.pth", "consolidated.04.pth", "consolidated.05.pth", 
    "consolidated.06.pth", "consolidated.07.pth"
]

model_weights = {}

# Load weights from the checkpoint files into the model
for checkpoint_file in checkpoint_files:
    checkpoint_path = os.path.join(model_path, checkpoint_file)
    checkpoint = torch.load(checkpoint_path)
    # Load the checkpoint weights into the model (this will depend on your model's architecture)
    model_weights.update(checkpoint)  # Update the model weights (use proper naming convention)

# Now, load the model weights manually
model.load_state_dict(model_weights)

# Initialize the Flask app
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")
    
    # Tokenize the input and generate a response
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=1000, num_return_sequences=1)
    
    # Decode the output and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

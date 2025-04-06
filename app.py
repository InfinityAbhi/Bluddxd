from flask import Flask, request, jsonify, send_from_directory
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = Flask(__name__)

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

@app.route("/")
def index():
    return send_from_directory(".", "blud.html")  # Serve blud.html

@app.route("/chat", methods=["POST"])
def chat():
    # Get user input from the request
    user_input = request.json.get("message", "")

    # Tokenize input and generate response
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the response as JSON
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
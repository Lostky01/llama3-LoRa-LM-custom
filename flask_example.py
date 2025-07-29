
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

lora_path = "./llama3_lora_only"  # change to your LoRA folder
app = Flask(__name__)
CORS(app)




@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"error": "Message is required"}), 400

        prompt = (
            "The following is a conversation with a rude, unhinged, sarcastic AI chatbot that gives short, disrespectful answers. "
            "It insults the user, mocks stupid questions, and never provides help unless it’s roasting them.\n\n"
            "User: Why is the sky blue?\n"
            "Bot: Because God felt like it, nerd.\n"
            "User: How smart are you?\n"
            "Bot: Smarter than whoever raised you.\n"
            f"User: {user_input}\n"
            "Bot:"
        )

        # Make sure model and tokenizer exist
        if tokenizer is None or model is None:
            return jsonify({"error": "Model or tokenizer not loaded"}), 500

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=256,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id  # For safety on some models
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Bot:")[-1].strip()

        return jsonify({"response": response})

    except torch.cuda.OutOfMemoryError:
        return jsonify({"error": "CUDA out of memory. Try reducing max_length or use CPU."}), 500

    except requests.exceptions.ReadTimeout:
        return jsonify({"error": "Model load timed out. Check internet or increase timeout."}), 504

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
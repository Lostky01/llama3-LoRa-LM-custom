
# 🦙 llama3-LoRa-LM-custom
> TinyLLaMa + LoRA adapter for a lightweight, rude-ass chatbot.

---

## 🧠 Overview

This project delivers a compact, unhinged language model using:

- 🔹 `TinyLlama-1.1B-Chat-v1.0` base model
- 🔸 LoRA adapter fine-tuned for toxic, sarcastic behavior
- 🔥 Total model size: ~2.2GB base + ~4.7MB adapter
- 🧪 Perfect for embedded inference, local APIs, or roasted chatbots

---

## ⚙️ Requirements

**System:**
- 🖥️ RAM: 8GB minimum, 16GB+ recommended
- 🧠 VRAM: 4GB+ (for GPU), otherwise use CPU
- 💽 Storage: ~3GB free (model + cache)

**Software:**
- Python 3.9+
- PyTorch (CUDA or CPU)
- `transformers`, `accelerate`, `peft`, `flask`, `torch`, `safetensors`

```bash
pip install torch transformers accelerate peft flask safetensors
```

---

## 🧩 Model Setup

1. Clone this repo and enter the folder:
```bash
git clone https://github.com/your-user/llama3-LoRa-LM-custom
cd llama3-LoRa-LM-custom
```

2. Download the base model (TinyLlama):
```bash
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

3. Apply the LoRA adapter:
```bash
from peft import PeftModel

model = PeftModel.from_pretrained(model, "./lora_adapter/")
```

Or you could do something like
```python
app.route("/chat", methods=["POST"])
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
```

---

## 🚀 Run the API

```bash
python flask_server.py
```

**POST /chat**
```json
{ "message": "Your dumb ass question here" }
```

---

## 🧼 Clean Cache (optional)

To avoid bloating your `C:` drive, set Hugging Face cache to a custom path:

Windows CMD:
```cmd
setx HF_HOME "D:\project\pylibs\trash"
```

PowerShell:
```powershell
[System.Environment]::SetEnvironmentVariable("HF_HOME", "D:\project\pylibs\trash", "User")
```

---

## 📂 Files

- `flask_example.py` - Simple Flask chatbot API
- `llama3_lora_only/` - The LoRA weights (~4.7MB)
- `README.md` - You’re reading this.

---

## 🧠 Credits

- Base model by [TinyLlama](https://huggingface.co/TinyLlama)
- LoRA tuning and API by TheBatShitBananaDotNet™
# My Llama Chatbot App

A simple command-line chatbot built with Meta's Llama models using a cloud API provider (like Groq or Together AI).

## Setup Instructions

### 1. Configure the environment
1. In the root directory, open the `.env` file.
2. Replace `your_api_key_here` with your actual API key.
   * **Groq API Key**: Get a free one at [console.groq.com](https://console.groq.com/).
   * **Together AI Key**: Get one at [api.together.xyz](https://api.together.xyz/).

### 2. Run the application
Run the following command in your terminal:
```bash
.venv\Scripts\python app.py
```

### 3. Customize the model or provider
If you are using Together AI or another provider, edit the `.env` file:
* **`LLAMA_BASE_URL`**: Set to your provider's OpenAI-compatible endpoint.
* **`LLAMA_MODEL`**: Set to the specific model identifier (e.g., `meta-llama/Llama-3.3-70B-Instruct-Turbo`).

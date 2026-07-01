import os
import sys
import shutil
from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI
import rag_helper

# Load configuration
load_dotenv()

api_key = os.getenv("LLAMA_API_KEY")
base_url = os.getenv("LLAMA_BASE_URL")
model_name = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instant")

if not api_key or api_key == "your_api_key_here":
    print("[ERROR] API Key is missing in .env! Please set it before running the web app.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)

# Configure uploads folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Copy knowledge.txt into uploads if it exists to preserve the sample facts
if os.path.exists("knowledge.txt") and not os.path.exists(os.path.join(UPLOAD_FOLDER, "knowledge.txt")):
    shutil.copy("knowledge.txt", os.path.join(UPLOAD_FOLDER, "knowledge.txt"))

# Initialize OpenAI client pointing to Llama provider
client = OpenAI(api_key=api_key, base_url=base_url)

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".txt", ".pdf"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/files", methods=["GET"])
def list_files():
    """List all uploaded files with details."""
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        for name in os.listdir(UPLOAD_FOLDER):
            path = os.path.join(UPLOAD_FOLDER, name)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                files.append({
                    "name": name,
                    "size_bytes": size,
                    "size_display": f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
                })
    return jsonify({"files": files})

@app.route("/api/files/<filename>/content", methods=["GET"])
def get_file_content(filename):
    """Retrieve content of a specific file."""
    # Ensure file is inside the upload folder (security sanitization)
    filename = secure_filename(filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
        
    try:
        content = rag_helper.extract_text_from_file(path)
        return jsonify({"name": filename, "content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    """Delete an uploaded file."""
    filename = secure_filename(filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
        
    try:
        os.remove(path)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle uploading of a file (TXT or PDF)."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            file.save(path)
            return jsonify({"success": True, "name": filename})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "Invalid file type. Only .txt and .pdf are allowed."}), 400

@app.route("/api/chat", methods=["POST"])
def chat():
    """Receive chat messages, query the RAG index, and stream the response back."""
    data = request.json or {}
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "No messages found"}), 400

    # Get last user query to search RAG index
    last_user_msg = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
    query = last_user_msg["content"] if last_user_msg else ""

    # Search document chunks using RAG helper
    chunks = rag_helper.retrieve_relevant_chunks(query, uploads_dir=UPLOAD_FOLDER, top_n=5)
    
    if chunks:
        # Build prompt context
        context_str = "\n\n".join([f"--- FROM FILE: {c['source']} ---\n{c['content']}" for c in chunks])
        system_instruction = (
            "You are a helpful, professional AI assistant.\n"
            "You have access to the following relevant document contexts uploaded by the user.\n"
            "Answer the user's questions based STRICTLY on the document context provided below.\n"
            "Identify the source filename when referencing facts.\n"
            "If the answer cannot be found in the document context, state politely that the "
            "information is not in the uploaded documents.\n\n"
            f"--- DOCUMENT CONTEXT ---\n{context_str}\n------------------------"
        )
    else:
        system_instruction = (
            "You are a helpful, professional AI assistant.\n"
            "No document context is currently uploaded. State politely that there are no "
            "documents loaded in the assistant's memory, and instruct the user to upload "
            "files in the sidebar to get started."
        )

    # Rebuild message thread ensuring system prompt is at the top
    cleaned_messages = [msg for msg in messages if msg.get("role") != "system"]
    cleaned_messages.insert(0, {"role": "system", "content": system_instruction})

    def generate():
        try:
            response_stream = client.chat.completions.create(
                model=model_name,
                messages=cleaned_messages,
                stream=True
            )
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {content}\n\n"
        except Exception as e:
            yield f"data: [Error: {str(e)}]\n\n"

    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    print(f"Starting Llama Web Assistant on http://127.0.0.1:5000")
    print(f"Using Model: {model_name}")
    app.run(debug=True, port=5000)

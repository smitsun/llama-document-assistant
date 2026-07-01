import os
import re
import math
from pypdf import PdfReader

def extract_text_from_file(file_path):
    """Extract all text from a text file or PDF."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    elif ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    return ""

def chunk_text(text, filename, chunk_size=800, overlap=150):
    """Split text into overlapping chunks of clean characters."""
    # Clean multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_data = text[start:end]
        chunks.append({
            "content": chunk_data,
            "source": filename
        })
        start += (chunk_size - overlap)
    return chunks

def tokenize(text):
    """Normalize and split text into a list of alphanumeric words."""
    return re.findall(r"\w+", text.lower())

def retrieve_relevant_chunks(query, uploads_dir="uploads", top_n=5):
    """Index all documents in the uploads directory and retrieve the top-N relevant chunks using simple TF-IDF ranking."""
    if not os.path.exists(uploads_dir):
        return []
    
    # 1. Load and chunk all files
    all_chunks = []
    for filename in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir, filename)
        if os.path.isfile(file_path):
            text = extract_text_from_file(file_path)
            chunks = chunk_text(text, filename)
            all_chunks.extend(chunks)
            
    if not all_chunks:
        return []
        
    # 2. Tokenize query
    query_tokens = tokenize(query)
    if not query_tokens:
        # Fallback to first N chunks if no query words
        return all_chunks[:top_n]
        
    # 3. Simple TF-IDF Implementation
    N = len(all_chunks)
    
    # Count how many chunks contain each query token
    df = {}
    for token in query_tokens:
        count = sum(1 for chunk in all_chunks if token in tokenize(chunk["content"]))
        df[token] = count
        
    # Calculate IDF for each query token
    idf = {}
    for token in query_tokens:
        df_val = df.get(token, 0)
        # Smoothed IDF: log((N - df + 0.5) / (df + 0.5) + 1.0)
        idf[token] = math.log((N - df_val + 0.5) / (df_val + 0.5) + 1.0)
        
    # Score each chunk
    scored_chunks = []
    for chunk in all_chunks:
        chunk_tokens = tokenize(chunk["content"])
        score = 0.0
        for token in query_tokens:
            tf = chunk_tokens.count(token)
            if tf > 0:
                # TF weight = 1 + log(tf)
                tf_weight = 1 + math.log(tf)
                score += tf_weight * idf[token]
        scored_chunks.append((score, chunk))
        
    # Sort by score descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Filter for chunks that have a non-zero matching score
    matched = [chunk for score, chunk in scored_chunks if score > 0]
    if matched:
        return matched[:top_n]
        
    # Fallback to first N chunks if no documents matched
    return all_chunks[:top_n]

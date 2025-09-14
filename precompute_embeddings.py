"""
Pre-compute embeddings for instant app loading
This script creates embeddings once and saves them to disk
"""

import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Dict

def create_chunks_with_metadata(docs: List[Dict]) -> List[Dict]:
    """Create chunks from documents with metadata"""
    chunks = []
    
    for doc in docs:
        if not doc or 'content' not in doc:
            continue
            
        content = doc['content']
        title = doc.get('title', 'Untitled')
        url = doc.get('url', 'No URL')
        
        # Split into sentences
        sentences = content.split('. ')
        
        # Create overlapping chunks
        chunk_size = 3  # sentences per chunk
        overlap = 1
        
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = '. '.join(chunk_sentences)
            
            if len(chunk_text.strip()) > 50:  # Skip very short chunks
                chunks.append({
                    'text': chunk_text,
                    'source_title': title,
                    'source_url': url,
                    'chunk_id': f"{title}_{i}"
                })
    
    return chunks

def precompute_embeddings():
    """Pre-compute and save embeddings"""
    print("🚀 Starting embedding pre-computation...")
    
    # Load documents
    if not os.path.exists('atlan_knowledge_base.json'):
        print("❌ atlan_knowledge_base.json not found!")
        return
    
    with open('atlan_knowledge_base.json', 'r') as f:
        docs = json.load(f)
    
    print(f"📚 Loaded {len(docs)} documents")
    
    # Create chunks
    chunks = create_chunks_with_metadata(docs)
    print(f"📝 Created {len(chunks)} text chunks")
    
    # Initialize embedding model
    print("🤖 Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings
    print("⚡ Computing embeddings...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Save everything
    embeddings_dir = 'embeddings_cache'
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Save embeddings as numpy array
    np.save(f'{embeddings_dir}/embeddings.npy', embeddings)
    
    # Save chunks metadata
    with open(f'{embeddings_dir}/chunks.json', 'w') as f:
        json.dump(chunks, f, indent=2)
    
    # Save model info
    model_info = {
        'model_name': 'all-MiniLM-L6-v2',
        'num_chunks': len(chunks),
        'embedding_dim': embeddings.shape[1]
    }
    
    with open(f'{embeddings_dir}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Embeddings saved to {embeddings_dir}/")
    print(f"   - {len(chunks)} chunks")
    print(f"   - Embedding dimension: {embeddings.shape[1]}")
    print(f"   - File size: {os.path.getsize(f'{embeddings_dir}/embeddings.npy') / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    precompute_embeddings()
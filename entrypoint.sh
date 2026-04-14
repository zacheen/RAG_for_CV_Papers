#!/bin/bash

# Start the Ollama server at port 11434
echo "Starting the Ollama Server"
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama server to start..."
sleep 5

# Pull LLaMA model
echo "Pulling LLaMA 3.2 model..."
ollama pull llama3.2

# Verify data exists
if [ ! -d "/root/data/chroma_db" ] || [ -z "$(ls -A /root/data/chroma_db 2>/dev/null)" ]; then
    echo "WARNING: No ChromaDB data found at /root/data/chroma_db"
    echo "Upload your locally-built data/ folder before using the app."
    echo "See docs/development/gcp-deployment.md for instructions."
fi

# Start the streamlit server, blocking exit. Disable source watching to avoid
# optional transformers image modules being imported during watcher scans.
echo "Starting the Streamlit server"
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.fileWatcherType none

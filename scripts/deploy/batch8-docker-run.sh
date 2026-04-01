#!/bin/bash
# Batch 8 [SSH] - Build & run Docker with GPU

cd ~/cv-paper-rag
docker rm -f cv-rag 2>/dev/null
docker build -t cv-paper-rag .
docker run -d --name cv-rag --gpus all -p 8501:8501 -p 11434:11434 -v $(pwd)/data:/root/data cv-paper-rag
docker logs -f cv-rag

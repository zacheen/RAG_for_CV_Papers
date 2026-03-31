# GCP Compute Engine Deployment Guide

## Overview

The deployment is split into two phases:
1. **Local** — Run the ingestion pipeline (crawl arXiv, build ChromaDB)
2. **GCP** — Upload the pre-built database, run LLM + Streamlit

This saves GPU costs since ingestion doesn't need a GPU.

---

## Phase 1: Local Ingestion (on your machine)

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run ingestion

```bash
# Download 800 papers and build the vector database
python scripts/ingest.py --max-papers 800

# Or with a topic filter
python scripts/ingest.py --query "object detection" --max-papers 800
```

This creates:
- `data/pdfs/` — downloaded PDF files
- `data/chroma_db/` — the vector database (this is what gets uploaded)

### Verify

```bash
python -c "from src.processing.embedder import get_collection_stats; print(get_collection_stats())"
```

Should show `total_chunks >= 10000`.

---

## Phase 2: GCP Deployment

### Step 1: Create the VM

```bash
gcloud compute instances create cv-paper-rag \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --boot-disk-size=200GB \
    --image-family=common-cu123-debian-11 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

**Notes:**
- `n1-standard-8` provides 8 vCPUs and 30GB RAM (sufficient for LLaMA 3.2)
- `nvidia-tesla-t4` is the most cost-effective GPU option
- Deep Learning VM image comes with NVIDIA drivers pre-installed

### Step 2: SSH into the VM

```bash
gcloud compute ssh cv-paper-rag --zone=us-central1-a
```

### Step 3: Install Docker + NVIDIA Container Toolkit

If not already installed on the Deep Learning VM:

```bash
# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Step 4: Clone the repo

```bash
git clone <YOUR_REPO_URL> cv-paper-rag
cd cv-paper-rag
```

### Step 5: Upload pre-built data from local machine

Run this **from your local machine** (not the VM):

```bash
# Upload the entire data/ folder to the VM
gcloud compute scp --recurse ./data cv-paper-rag:~/cv-paper-rag/data \
    --zone=us-central1-a
```

If the data folder is large, you can compress first:

```bash
# Local: compress
tar -czf data.tar.gz data/

# Upload compressed archive
gcloud compute scp data.tar.gz cv-paper-rag:~/cv-paper-rag/ --zone=us-central1-a

# On VM: decompress
gcloud compute ssh cv-paper-rag --zone=us-central1-a --command="cd ~/cv-paper-rag && tar -xzf data.tar.gz"
```

### Step 6: Build and run with GPU

```bash
# SSH into VM
gcloud compute ssh cv-paper-rag --zone=us-central1-a

cd cv-paper-rag
docker build -t cv-paper-rag .
docker run -d \
    --name cv-rag \
    --gpus all \
    -p 8501:8501 \
    -p 11434:11434 \
    -v $(pwd)/data:/root/data \
    cv-paper-rag
```

The container will:
1. Start Ollama server (with GPU acceleration)
2. Pull LLaMA 3.2 model (~4.7GB, first run only)
3. Start Streamlit on port 8501

Monitor progress:
```bash
docker logs -f cv-rag
```

### Step 7: Open Firewall for Port 8501

```bash
gcloud compute firewall-rules create allow-streamlit \
    --allow=tcp:8501 \
    --target-tags=http-server \
    --description="Allow Streamlit access"

gcloud compute instances add-tags cv-paper-rag \
    --zone=us-central1-a \
    --tags=http-server
```

### Step 8: Access the App

Get the external IP:
```bash
gcloud compute instances describe cv-paper-rag \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Open in browser: `http://<EXTERNAL_IP>:8501`

---

## Cost Estimates

| Resource | Approximate Cost |
|----------|-----------------|
| n1-standard-8 | ~$0.38/hr |
| NVIDIA T4 GPU | ~$0.35/hr |
| 200GB SSD | ~$34/month |
| **Total (running)** | **~$0.73/hr** |

**Tip:** Stop the VM when not demoing to save costs:
```bash
gcloud compute instances stop cv-paper-rag --zone=us-central1-a
gcloud compute instances start cv-paper-rag --zone=us-central1-a
```

## Troubleshooting

- **GPU not detected:** Run `nvidia-smi` inside the VM to verify drivers.
- **Ollama OOM:** LLaMA 3.2 needs ~5GB VRAM. T4 has 16GB, so this should be fine.
- **No data warning on startup:** You forgot to upload `data/`. Run the `gcloud compute scp` step.
- **Port not accessible:** Verify firewall rule and that the VM has an external IP.

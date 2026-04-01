# GCP Compute Engine Deployment Guide

## Overview

The deployment is split into two phases:
1. **Local (Windows)** — Run the ingestion pipeline (crawl arXiv, build ChromaDB)
2. **GCP** — Upload the pre-built database, run LLM + Streamlit on GPU VM

> **Convention:** Each batch is numbered. `[PS]` = run in PowerShell on Windows. `[SSH]` = run in SSH on the VM.
>
> Batch scripts are in `scripts/deploy/`. PS batches (0-6) run first, then SSH into VM and run batches 7-8. **No switching back and forth.**

| Batch | Where | Script | What |
|-------|-------|--------|------|
| 0 | PS | `batch0-local-ingest.ps1` | Local ingestion |
| 1 | PS | `batch1-create-vm.ps1` | Create VM |
| 2 | PS | `batch2-upload.ps1` | Upload code + chroma_db |
| 3 | PS | `batch3-verify-upload.ps1` | Verify upload |
| 4 | PS | `batch4-firewall.ps1` | Open firewall |
| 5 | PS | `batch5-get-url.ps1` | Get app URL |
| 6 | PS | `batch6-ssh.ps1` | SSH into VM |
| 7 | SSH | `batch7-vm-setup.sh` | Install Docker + verify GPU |
| 8 | SSH | `batch8-docker-run.sh` | Build + start Docker |

---

## Phase 1: Local Ingestion

### Batch 0 — Install & Ingest [PS]

```powershell
.\scripts\deploy\batch0-local-ingest.ps1
```

Installs dependencies, downloads 800 arXiv CS.CV papers, builds ChromaDB.
Output: `data/chroma_db/` (~700MB), `total_chunks >= 10000`.

---

## Phase 2: GCP Deployment

### Batch 1 — Create VM [PS]

```powershell
.\scripts\deploy\batch1-create-vm.ps1
```

Creates `g2-standard-8` VM with NVIDIA L4 GPU, CUDA 12.8, Ubuntu 22.04.

> If zone is full (`ZONE_RESOURCE_POOL_EXHAUSTED`), edit `$ZONE` in the script and try: `us-east1-b`, `europe-west4-a`, `asia-east1-a`

### Batch 2 — Upload project to VM [PS]

```powershell
.\scripts\deploy\batch2-upload.ps1
```

Uploads `src/`, `scripts/`, `docs/`, root files, and `data/chroma_db/` to VM. Does NOT upload `data/pdfs/` (not needed on GCP).

> **Windows PowerShell notes:**
> - Do NOT use `~` in remote path — use `/home/User/` instead
> - Remote directory must exist before uploading
> - Edit `$VM` and `$ZONE` at top of script if VM name or zone differs

### Batch 3 — Verify upload [PS]

```powershell
.\scripts\deploy\batch3-verify-upload.ps1
```

Checks that `src/`, `scripts/`, and `data/chroma_db/` exist on VM.

### Batch 4 — Open firewall [PS]

```powershell
.\scripts\deploy\batch4-firewall.ps1
```

Opens port 8501 for Streamlit access. Only needs to run once.

### Batch 5 — Get URL [PS]

```powershell
.\scripts\deploy\batch5-get-url.ps1
```

Prints the app URL. Save it — you'll open it in browser after Batch 8.

### Batch 6 — SSH into VM [PS]

```powershell
.\scripts\deploy\batch6-ssh.ps1
```

Connects to the VM. Stay here for Batch 7 and 8.

### Batch 7 — Install Docker + verify GPU [SSH]

```bash
bash ~/cv-paper-rag/scripts/deploy/batch7-vm-setup.sh
```

Installs Docker + NVIDIA Container Toolkit. Last line should show NVIDIA L4 inside Docker.

### Batch 8 — Build & start Docker [SSH]

```bash
bash ~/cv-paper-rag/scripts/deploy/batch8-docker-run.sh
```

Builds Docker image, starts container with GPU. Wait for `Starting the Streamlit server`, then open the URL from Batch 5 in your browser.

---

## Quick Reference: Rebuild after code changes

### Batch R1 — Upload updated code [PS]

```powershell
$VM = "cv-paper-rag"
$ZONE = "us-west1-a"
$PROJECT_DIR = "D:\dont_move\Northeastern_University\CS6120\final_project"

gcloud compute scp --recurse "$PROJECT_DIR\src" ${VM}:/home/User/cv-paper-rag/ --zone=$ZONE
gcloud compute scp "$PROJECT_DIR\app.py" "$PROJECT_DIR\requirements.txt" "$PROJECT_DIR\Dockerfile" "$PROJECT_DIR\entrypoint.sh" ${VM}:/home/User/cv-paper-rag/ --zone=$ZONE
```

### Batch R2 — Rebuild & restart [SSH]

```bash
cd ~/cv-paper-rag
docker rm -f cv-rag
docker build -t cv-paper-rag .
docker run -d --name cv-rag --gpus all -p 8501:8501 -p 11434:11434 -v $(pwd)/data:/root/data cv-paper-rag
docker logs -f cv-rag
```

---

## VM Management [PS]

```powershell
# Stop VM (saves cost)
gcloud compute instances stop cv-paper-rag --zone=us-west1-a

# Start VM
gcloud compute instances start cv-paper-rag --zone=us-west1-a

# Delete VM entirely
gcloud compute instances delete cv-paper-rag --zone=us-west1-a
```

---

## Cost Estimates

| Resource | Approximate Cost |
|----------|-----------------|
| g2-standard-8 + NVIDIA L4 | ~$0.56/hr |
| 200GB SSD | ~$34/month |
| **Total (running)** | **~$0.56/hr** |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ZONE_RESOURCE_POOL_EXHAUSTED` | Try another zone in Batch 1 |
| `nvidia-smi` fails on VM | Driver not installed: `sudo apt install nvidia-driver-570` |
| Docker can't see GPU | Install NVIDIA Container Toolkit (Batch 7) |
| `exec ./entrypoint.sh: no such file` | Windows CRLF issue. On VM: `dos2unix ~/cv-paper-rag/entrypoint.sh`, then Batch R2 |
| `ModuleNotFoundError: src.config` | `src/` is empty. Re-upload with Batch R1, verify with Batch 3 |
| Ollama uses CPU (slow) | Check `docker exec cv-rag ollama ps`. Ensure `--gpus all` in Batch 8 |
| No data warning on startup | Upload `chroma_db` (Batch 2) |
| Port 8501 not accessible | Run Batch 4 (firewall) |
| PowerShell comma parsing | Wrap in quotes: `--accelerator="type=nvidia-l4,count=1"` |
| Nested `scripts/scripts/` on VM | scp destination should end with `/` not `/scripts`. Fixed in current batch2. |

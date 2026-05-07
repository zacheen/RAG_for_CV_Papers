# Batch 2 [PS] - Upload project + ChromaDB to VM
# Run from PowerShell on your local machine

$VM = "cv-paper-rag"
$ZONE = "us-west1-a"
$PROJECT_DIR = "D:\dont_move\Northeastern_University\CS6120\final_project"

# Create target directory on VM (only top-level, let scp create subdirs)
# gcloud compute ssh $VM --zone=$ZONE --command="rm -rf ~/cv-paper-rag && mkdir -p ~/cv-paper-rag/data"

# Upload code (scp --recurse creates src/, scripts/, docs/ automatically)
gcloud compute scp --recurse "$PROJECT_DIR\src" ${VM}:/home/User/cv-paper-rag/ --zone=$ZONE
gcloud compute scp --recurse "$PROJECT_DIR\scripts" ${VM}:/home/User/cv-paper-rag/ --zone=$ZONE
gcloud compute scp --recurse "$PROJECT_DIR\docs" ${VM}:/home/User/cv-paper-rag/ --zone=$ZONE
gcloud compute scp "$PROJECT_DIR\app.py" "$PROJECT_DIR\requirements.txt" "$PROJECT_DIR\Dockerfile" "$PROJECT_DIR\entrypoint.sh" "$PROJECT_DIR\CLAUDE.md" ${VM}:/home/User/cv-paper-rag/ --zone=$ZONE

# Upload ChromaDB only (not PDFs)
# gcloud compute ssh $VM --zone=$ZONE --command="mkdir -p /home/User/cv-paper-rag/data/chroma_db"
# gcloud compute scp --recurse "$PROJECT_DIR\data\chroma_db" ${VM}:/home/User/cv-paper-rag/data/ --zone=$ZONE

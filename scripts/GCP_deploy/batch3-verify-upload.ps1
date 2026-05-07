# Batch 3 [PS] - Verify upload

$VM = "cv-paper-rag"
$ZONE = "us-west1-a"

gcloud compute ssh $VM --zone=$ZONE --command="ls ~/cv-paper-rag && echo '---' && ls ~/cv-paper-rag/src/ && echo '---' && du -sh ~/cv-paper-rag/data/chroma_db"

# Batch 6 [PS] - SSH into VM (stay here for batch 7 & 8)

$VM = "cv-paper-rag"
$ZONE = "us-west1-a"

gcloud compute ssh $VM --zone=$ZONE

# Batch 5 [PS] - Get external IP

$VM = "cv-paper-rag"
$ZONE = "us-west1-a"

$IP = gcloud compute instances describe $VM --zone=$ZONE --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
Write-Host ""
Write-Host "App URL: http://${IP}:8501"

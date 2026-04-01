# Batch 4 [PS] - Open firewall for Streamlit

$VM = "cv-paper-rag"
$ZONE = "us-west1-a"

gcloud compute firewall-rules create allow-streamlit --allow=tcp:8501 --target-tags=http-server --description="Allow Streamlit access"
gcloud compute instances add-tags $VM --zone=$ZONE --tags=http-server

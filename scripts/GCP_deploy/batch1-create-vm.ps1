# Batch 1 [PS] - Create GCP VM
# If zone is full, change us-west1-a to another zone (us-east1-b, europe-west4-a, etc.)

gcloud compute instances create cv-paper-rag --zone=us-west1-a --machine-type=g2-standard-8 --accelerator="type=nvidia-l4,count=1" --boot-disk-size=200GB --image-family=common-cu128-ubuntu-2204-nvidia-570 --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --metadata="install-nvidia-driver=True"

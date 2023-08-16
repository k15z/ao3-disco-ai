# Push code and data to gcloud
docker build --platform linux/amd64 . --tag ao3-disco-ai
docker tag ao3-disco-ai us-west1-docker.pkg.dev/ao3-disco/ao3-disco-docker/ao3-disco-ai
docker push us-west1-docker.pkg.dev/ao3-disco/ao3-disco-docker/ao3-disco-ai
gcloud storage rsync data gs://ao3-disco-ai/data --recursive --gzip-in-flight-all --checksums-only

# Submit jobs in gcloud batch
python gcloud/submit.py # Modify this to configure your experiments

# Pull tensorboard results back down
gcloud storage rsync gs://ao3-disco-ai/lightning_logs lightning_logs --recursive --gzip-in-flight-all --checksums-only

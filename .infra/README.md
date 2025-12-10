# Infra

## Create k8s cluster

```sh
export PROJECT_ID=redislabs-redisvpc-dev-238506
gcloud container clusters create-auto redis-ai-workshop-dec-2025 \
    --location=europe-west1 \
    --project=$PROJECT_ID
```

## Install Ollama

The default configuration uses NVIDIA L4 GPUs. Make sure your GKE Autopilot cluster has access to GPU resources.

```sh
helm repo add otwld https://helm.otwld.com/
helm repo update
helm install ollama otwld/ollama -f ./helm/ollama/values.yaml
```

### Get Public IP Address

The Ollama service is exposed via LoadBalancer. Get the external IP:

```sh
# Wait for external IP to be assigned (may take 1-2 minutes)
kubectl get svc ollama -w

# Or get it directly once assigned
export OLLAMA_IP=$(kubectl get svc ollama -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Ollama is available at: http://${OLLAMA_IP}:11434"
```

### Test Ollama

Once you have the external IP, test it:

```sh
# List available models
curl http://${OLLAMA_IP}:11434/api/tags

# Generate a response
curl http://${OLLAMA_IP}:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

# Or use the ollama CLI (if installed locally)
export OLLAMA_HOST=http://${OLLAMA_IP}:11434
ollama list
ollama run llama3.2:3b "Why is the sky blue?"
```

### Restrict Access to Specific IP

To restrict access to only your current IP address:

```sh
# Get your current IP
MY_IP=$(curl -s https://api.ipify.org)
echo "Your IP: ${MY_IP}"

# Patch the service to restrict access
kubectl patch svc ollama -p "{\"spec\":{\"loadBalancerSourceRanges\":[\"${MY_IP}/32\"]}}"

# Verify the restriction
kubectl get svc ollama -o jsonpath='{.spec.loadBalancerSourceRanges}'
```

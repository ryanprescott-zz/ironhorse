# N8N Orchestrator Docker Setup

This directory contains Docker configuration for running n8n as an orchestrator for AI Toolkit components.

## Overview

The n8n orchestrator is deployed using the official n8n Docker image and configured to work with AI Toolkit components.

## Configuration

### Environment Variables

The following environment variables are configured in docker-compose.yml:

- `N8N_BASIC_AUTH_ACTIVE`: Enable basic authentication (true)
- `N8N_BASIC_AUTH_USER`: Admin username (default: admin)
- `N8N_BASIC_AUTH_PASSWORD`: Admin password (default: admin)
- `N8N_HOST`: Host to bind to (0.0.0.0)
- `N8N_PORT`: Port to run on (5678)
- `N8N_PROTOCOL`: Protocol to use (http)
- `WEBHOOK_URL`: Webhook URL for n8n

### Volumes

- `n8n-data`: Persistent storage for n8n workflows and data
- `/workflows`: Read-only mount of workflow templates from `integrations/orchestrators/n8n/pipelines`

## Usage

### Starting n8n

From the project root:

```bash
docker-compose up -d n8n
```

### Accessing n8n

1. Open your browser to http://localhost:5678
2. Login with credentials:
   - Username: admin
   - Password: admin

### Importing Workflows

1. Navigate to the n8n UI
2. Click "Workflows" → "Import from File"
3. Select a workflow JSON file from `/workflows` directory
4. Save and activate the workflow

## Network Configuration

n8n runs on the `network` Docker network, allowing it to communicate with component services using their service names:

- `http://docling-parser:26000`
- `http://langchain-splitter:26001`

## Security Considerations

**IMPORTANT**: The default configuration uses basic authentication with default credentials. For production use:

1. Change the default username and password
2. Use HTTPS instead of HTTP
3. Configure proper authentication (OAuth, LDAP, etc.)
4. Set up SSL/TLS certificates
5. Use secrets management for sensitive data

## Customization

### Using a Custom n8n Image

If you need custom n8n nodes or configurations:

1. Create a Dockerfile in this directory:

```dockerfile
FROM n8nio/n8n:latest

# Install custom nodes
RUN npm install -g n8n-nodes-custom-package

# Copy custom configuration
COPY n8n-config.json /home/node/.n8n/config
```

2. Update docker-compose.yml to build from this Dockerfile:

```yaml
n8n:
  build:
    context: .
    dockerfile: platform/docker/orchestrators/n8n/Dockerfile
```

### Persistent Data

N8N data is persisted in the `n8n-data` Docker volume. To back up:

```bash
docker run --rm -v n8n-data:/data -v $(pwd):/backup alpine tar czf /backup/n8n-backup.tar.gz /data
```

To restore:

```bash
docker run --rm -v n8n-data:/data -v $(pwd):/backup alpine tar xzf /backup/n8n-backup.tar.gz -C /
```

## Troubleshooting

### Can't Connect to Components

1. Verify all services are running:
   ```bash
   docker-compose ps
   ```

2. Check network connectivity:
   ```bash
   docker-compose exec n8n ping docling-parser
   ```

3. Verify component health:
   ```bash
   curl http://localhost:26000/health
   curl http://localhost:26001/health
   ```

### Workflows Not Executing

1. Check n8n logs:
   ```bash
   docker-compose logs n8n
   ```

2. Verify workflow is active in the n8n UI

3. Check component logs for errors:
   ```bash
   docker-compose logs docling-parser
   docker-compose logs langchain-splitter
   ```

## Additional Resources

- [N8N Documentation](https://docs.n8n.io/)
- [N8N Docker Guide](https://docs.n8n.io/hosting/installation/docker/)
- [AI Toolkit Workflow Examples](../../../../integrations/orchestrators/n8n/pipelines/)

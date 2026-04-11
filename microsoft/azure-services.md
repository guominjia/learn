# Azure OpenAI Documentation

## Quick Links

- [Azure Portal](https://portal.azure.com)
- [Azure AI Foundry Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/overview)
- [Azure AI Studio](https://ai.azure.com/?cid=learnDocs)

## Understanding Models vs Deployments

### Key Difference

In Azure OpenAI, there's an important distinction between **available models** and **actual deployments**:

- **`client.models.list()`** returns the supported base models that your Azure OpenAI resource has access to (e.g., `gpt-4`, `gpt-4o`, `gpt-5`, etc.)
- **Deployments** are the actual instances you've created to interact with these models

> **Important**: You interact with models through deployments, not the base models directly.

## Listing Your Deployments

There are three methods to view your Azure OpenAI deployments:

### 1. Azure Portal (Recommended)
- Navigate to your Azure OpenAI resource
- Select **Deployments** under the "Management" section
- View all active deployments with their configurations

### 2. Azure Management SDK (Python)
```python
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient

# Use CognitiveServicesManagementClient
# Call client.deployments.list() with:
# - subscription_id
# - resource_group_name  
# - account_name
```

### 3. REST API
Use Azure's REST API to programmatically retrieve deployment information for your Azure OpenAI resource.

### Summary
- **`client.models.list()`** → Shows what you *could* deploy
- **Portal/SDK/REST API** → Shows what you *have* deployed

## Deployment Limitations

### One Model Per Deployment Rule

❌ **Not Possible**: You cannot create a single deployment (e.g., "GPT4TO5") that uses both GPT-4 and GPT-5 simultaneously.

### Reasons:

1. **Single Model Association**: Each Azure OpenAI deployment is linked to one specific model version (e.g., `gpt-4-0613`, `gpt-5-reasoning`)

2. **Unique Deployment Names**: Deployment names serve as unique identifiers for specific model instances within your Azure resource

### Best Practice
Create separate deployments for each model you want to use, with descriptive names that indicate the model and purpose. 
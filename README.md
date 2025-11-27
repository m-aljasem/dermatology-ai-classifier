# Dermatology AI Classifier

Convolutional neural network for **classifying 7 types of skin lesions** from dermatoscopic images (HAM10000).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 👤 Author

- **Name**: Mohamad AlJasem, MD MPH MSc  
- **Email**: [mohamad@aljasem.eu.org](mailto:mohamad@aljasem.eu.org)  
- **GitHub**: [github.com/m-aljasem](https://github.com/m-aljasem)  
- **Website**: [aljasem.eu.org](https://aljasem.eu.org)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Classes](#-classes)
- [Exported Weights](#-exported-weights)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

This project trains a **CNN classifier** on the **HAM10000** dataset to differentiate between **7 skin lesion types**, including melanoma and benign lesions.

It provides:

- A custom CNN architecture
- A Streamlit app for image‑based diagnosis assistance
- Exported weights for deployment

> ⚠️ **Not a diagnostic tool** – for research & education only.

---

## ✨ Features

- 7‑class softmax classifier
- Input resolution: **224×224 RGB**
- Data augmentation (via training script, once wired)
- Confidence scores for each lesion type

---

## 🛠 Tech Stack

- Python 3.8+
- TensorFlow / Keras
- Streamlit

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Dev tools:

```bash
pip install -r requirements-dev.txt
```

---

## 🚀 Quick Start

### 1️⃣ Train the Model

```bash
cd dermatology-ai-classifier
python src/train.py
```

Once data loading is implemented, this will save best weights to:

```text
models/skin_cancer_model.h5
```

### 2️⃣ Run the Streamlit App

```bash
cd dermatology-ai-classifier
streamlit run app.py
```

Upload a dermatoscopic image and receive:
- Predicted lesion type
- Confidence score

---

## 🧑‍💻 Usage

### 🌐 Web App

```bash
streamlit run app.py
```

The app:

- Builds the CNN model
- Loads `models/skin_cancer_model.h5` if present
- Outputs class name + clinical description + confidence

### 🧬 Programmatic Usage

```python
from src.model import build_skin_cancer_model
import numpy as np

model = build_skin_cancer_model()
model.load_weights("models/skin_cancer_model.h5")  # after training

# img_preprocessed: (1, 224, 224, 3) in [0,1]
pred = model.predict(img_preprocessed, verbose=0)[0]
class_idx = np.argmax(pred)
confidence = pred[class_idx]
```

---

## 🗂 Project Structure

```text
dermatology-ai-classifier/
├── app.py                    # Streamlit app
├── config/
├── data/                     # HAM10000 metadata + images
├── docs/
├── experiments/
├── models/                   # Saved weights (skin_cancer_model.h5)
├── notebooks/
├── scripts/
├── src/
│   ├── __init__.py
│   └── model.py              # build_skin_cancer_model()
└── tests/
```

---

## 🧬 Classes

- **akiec** – Actinic keratoses and intraepithelial carcinoma  
- **bcc** – Basal cell carcinoma  
- **bkl** – Benign keratosis‑like lesions  
- **df** – Dermatofibroma  
- **mel** – Melanoma  
- **nv** – Melanocytic nevi  
- **vasc** – Vascular lesions  

---

## 📦 Exported Weights

- Training script saves to:

```text
../models/skin_cancer_model.h5
```

- Streamlit app loads from:

```text
models/skin_cancer_model.h5
```

You can ship the `models/` folder with any deployment.

---

## 📄 License

Licensed under the **MIT License**.  
See `LICENSE` for details.

---

## 🏥 Disclaimer

> This model is intended for **research and educational use only**.  
> It must **not** be used for clinical diagnosis or patient management.




## 🌐 RESTful API

The project includes a FastAPI server for programmatic access to the model. This allows you to integrate predictions into your own applications, web services, or scripts.

### Installation

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

### Starting the API Server

Start the API server using one of these methods:

**Method 1: Direct Python execution**
```bash
python api.py
```

**Method 2: Using uvicorn directly**
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Method 3: Production mode (no auto-reload)**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API information |
| `/health` | GET | Health check endpoint (checks if model is loaded) |
| `/model/info` | GET | Get detailed model information |
| `/predict` | POST | Make a prediction |

### Interactive API Documentation

Once the server is running, you can access interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API explorer with "Try it out" feature
- **ReDoc**: `http://localhost:8000/redoc` - Beautiful, responsive API documentation

### Using the API

#### Health Check

```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
# Output: {"status": "healthy", "model_loaded": true}
```

#### Get Model Information

```python
import requests

response = requests.get("http://localhost:8000/model/info")
print(response.json())
# Output: Model type, input shape, classes, etc.
```

#### Make Predictions

# Example: Skin Lesion Classification
import requests

# Upload skin lesion image
with open("skin_lesion.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    result = response.json()

print(f"Lesion Type: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All class probabilities: {result['probabilities']}")

### Using cURL

You can also use cURL to interact with the API:

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get Model Info:**
```bash
curl http://localhost:8000/model/info
```

**Make Prediction (for image-based models):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_image.jpg"
```

**Make Prediction (for JSON-based models like Alzheimer's):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"gender": 1.0, "age": 75.0, "education": 14.0, "ses": 2.0, "mmse": 27.0, "etiv": 1490.0, "nwbv": 0.73, "asf": 1.20}'
```

### Error Handling

The API returns appropriate HTTP status codes:

- `200 OK` - Successful request
- `400 Bad Request` - Invalid input data
- `503 Service Unavailable` - Model not loaded (train the model first)
- `500 Internal Server Error` - Server error during prediction

Example error handling:

```python
import requests

try:
    response = requests.post("http://localhost:8000/predict", json=data)
    response.raise_for_status()  # Raises exception for bad status codes
    result = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
```

### API Response Format

**Successful Prediction Response:**
```json
{
    "prediction": "Demented",
    "confidence": 0.85,
    "probabilities": {
        "Nondemented": 0.15,
        "Demented": 0.85
    }
}
```

**Error Response:**
```json
{
    "detail": "Model not loaded. Please train the model first."
}
```

### Deployment

For production deployment, consider:

1. **Using a production ASGI server**: Use `uvicorn` with multiple workers or `gunicorn` with uvicorn workers
2. **Adding authentication**: Implement API keys or OAuth2
3. **Rate limiting**: Add rate limiting to prevent abuse
4. **Logging**: Configure proper logging for monitoring
5. **HTTPS**: Use HTTPS in production with SSL certificates

Example production command:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

## 🔌 MCP Server

The project includes a Model Context Protocol (MCP) server that exposes the model as tools for AI assistants and other MCP-compatible clients. This allows AI assistants like Claude, ChatGPT, or custom MCP clients to interact with your model.

### What is MCP?

Model Context Protocol (MCP) is a standardized protocol for AI assistants to interact with external tools and services. It enables AI assistants to:
- Call your model for predictions
- Get model information
- Check model health status

### Installation

The MCP server requires the MCP SDK:

```bash
pip install mcp
```

### Starting the MCP Server

Start the MCP server:

```bash
python mcp_server.py
```

The server runs as a stdio-based server, communicating via standard input/output. It's designed to be used with MCP clients.

### MCP Tools

The server exposes the following tools:

| Tool | Description | Parameters |
|------|-------------|------------|
| `predict` | Make a prediction using the model | `input` (string): Input data as JSON string or file path |
| `model_info` | Get information about the loaded model | None |
| `health_check` | Check if the model is loaded and ready | None |

### Using MCP with Python Client

```python
from mcp import ClientSession, StdioServerParameters
import asyncio
import json

async def main():
    # Connect to MCP server
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"],
            env=None
        )
    ) as session:
        # Initialize the session
        await session.initialize()
        
        # List available tools
        tools = await session.list_tools()
        print("Available tools:", [tool.name for tool in tools])
        
        # Health check
        health_result = await session.call_tool(
            "health_check",
            {}
        )
        print("Health:", health_result.content[0].text)
        
        # Get model info
        model_info = await session.call_tool(
            "model_info",
            {}
        )
        print("Model Info:", model_info.content[0].text)
        
        # Make prediction
        # For image-based models, provide base64 encoded image or file path
        prediction_input = json.dumps({
            "file_path": "test_image.jpg"
        })
        
        prediction_result = await session.call_tool(
            "predict",
            {"input": prediction_input}
        )
        print("Prediction:", prediction_result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())
```

### Using MCP with Claude Desktop

To use with Claude Desktop, add this to your MCP configuration file:

```json
{
  "mcpServers": {
    "skin-cancer": {
      "command": "python",
      "args": ["/home/m-aljasem/projects/ai-projects/skin-cancer/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/home/m-aljasem/projects/ai-projects/skin-cancer"
      }
    }
  }
}
```

### MCP Tool Responses

**Health Check Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Model Info Response:**
```json
{
  "model_type": "TensorFlow/Keras",
  "model_path": "models/model.h5",
  "classes": ["Class1", "Class2"],
  "description": "Model description"
}
```

**Prediction Response:**
```json
{
  "prediction": "Class1",
  "confidence": 0.95,
  "probabilities": {
    "Class1": 0.95,
    "Class2": 0.05
  }
}
```

### Error Handling

The MCP server returns error messages in JSON format:

```json
{
  "error": "Model not loaded. Please train the model first."
}
```

### Integration Examples

**Example 1: Batch Predictions**
```python
import asyncio
from mcp import ClientSession, StdioServerParameters

async def batch_predict(file_paths):
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
    ) as session:
        await session.initialize()
        
        results = []
        for file_path in file_paths:
            result = await session.call_tool(
                "predict",
                {"input": json.dumps({"file_path": file_path})}
            )
            results.append(json.loads(result.content[0].text))
        
        return results

# Usage
predictions = asyncio.run(batch_predict([
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
]))
```

**Example 2: Model Monitoring**
```python
import asyncio
from mcp import ClientSession, StdioServerParameters
import time

async def monitor_model():
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
    ) as session:
        await session.initialize()
        
        while True:
            health = await session.call_tool("health_check", {})
            print(f"[{time.strftime('%H:%M:%S')}] Health: {health.content[0].text}")
            await asyncio.sleep(60)  # Check every minute

# Run monitoring
asyncio.run(monitor_model())
```

### Troubleshooting

**Issue: MCP server not starting**
- Ensure `mcp` package is installed: `pip install mcp`
- Check that the model file exists in the `models/` directory
- Verify Python path and dependencies

**Issue: Tool calls failing**
- Check that the model is trained and weights are saved
- Verify input format matches expected format
- Check server logs for detailed error messages

**Issue: Connection errors**
- Ensure the MCP server process is running
- Check that stdio communication is working
- Verify environment variables if needed


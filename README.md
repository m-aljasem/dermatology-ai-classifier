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

The project includes a FastAPI server for programmatic access to the model.

### Starting the API Server

```bash
python api.py
# or
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `GET /model/info` - Get model information
- `POST /predict` - Make a prediction

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction (example for image-based models)
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

## 🔌 MCP Server

The project includes a Model Context Protocol (MCP) server for integration with AI assistants.

### Starting the MCP Server

```bash
python mcp_server.py
```

### MCP Tools

The server exposes the following tools:

- `predict` - Make a prediction using the model
- `model_info` - Get information about the loaded model
- `health_check` - Check if the model is loaded and ready

### MCP Client Integration

To use with an MCP client:

```python
from mcp import ClientSession, StdioServerParameters
import asyncio

async def main():
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
    ) as session:
        # List tools
        tools = await session.list_tools()
        print(tools)
        
        # Call tool
        result = await session.call_tool(
            "health_check",
            {}
        )
        print(result)

asyncio.run(main())
```


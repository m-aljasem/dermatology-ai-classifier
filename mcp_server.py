"""
MCP Server for Skin Lesion Classification

Model Context Protocol (MCP) server that exposes the model as tools
for AI assistants and other MCP clients.
"""

import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf

# MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("⚠️  MCP SDK not installed. Install with: pip install mcp")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

app = Server("skin-cancer-mcp-server")

# Paths
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "skin_cancer_model.h5"

# Global model
model = None
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
INPUT_SHAPE = (224, 224, 3)

def load_model():
    """Load TensorFlow model."""
    global model
    try:
        if MODEL_PATH.exists():
            model = tf.keras.models.load_model(str(MODEL_PATH))
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️  Model not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="predict",
            description="Make a prediction using the Skin Lesion Classification model. Input should be a file path to an image or base64 encoded image.",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "File path to image or base64 encoded image data"
                    }
                },
                "required": ["input"]
            }
        ),
        Tool(
            name="model_info",
            description="Get information about the loaded model",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="health_check",
            description="Check if the model is loaded and ready",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    if name == "health_check":
        model_loaded = model is not None
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "healthy" if model_loaded else "degraded",
                "model_loaded": model_loaded
            }, indent=2)
        )]
    
    elif name == "model_info":
        if model is None:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Model not loaded"
                }, indent=2)
            )]
        
        info = {
            "model_type": "TensorFlow/Keras",
            "model_path": str(MODEL_PATH),
            "classes": CLASSES,
            "input_shape": INPUT_SHAPE,
            "description": "Skin Lesion Classification"
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(info, indent=2)
        )]
    
    elif name == "predict":
        if model is None:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Model not loaded. Please train the model first."
                }, indent=2)
            )]
        
        try:
            input_data = arguments.get("input", "")
            
            # Handle file path or base64 encoded image
            if Path(input_data).exists():
                image = Image.open(input_data)
            elif input_data.startswith("data:image"):
                # Base64 encoded image
                header, encoded = input_data.split(",", 1)
                image_data = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(image_data))
            else:
                # Try as base64 string
                try:
                    image_data = base64.b64decode(input_data)
                    image = Image.open(io.BytesIO(image_data))
                except:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": "Invalid input. Provide file path or base64 encoded image."
                        }, indent=2)
                    )]
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize and preprocess
            image = image.resize(INPUT_SHAPE[:2])
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, 0)
            
            # Predict
            pred = model.predict(img_array, verbose=0)
            class_idx = int(np.argmax(pred[0]))
            confidence = float(pred[0][class_idx])
            class_name = CLASSES[class_idx]
            
            probabilities = {cls: float(prob) for cls, prob in zip(CLASSES, pred[0])}
            
            result = {
                "prediction": class_name,
                "class_index": class_idx,
                "confidence": confidence,
                "probabilities": probabilities
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Prediction error: {str(e)}"
                }, indent=2)
            )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Unknown tool: {name}"
            }, indent=2)
        )]

async def main():
    """Main entry point."""
    # Load model
    load_model()
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())

# Model Setup Guide

The trained model (`skin_cancer_model.keras`) is **not included** in the GitHub repository due to file size constraints (~50-200 MB). This guide explains how to obtain and set up the model.

## 🚀 Quick Setup

### Option 1: Download Script (Easiest)

```bash
# Download from GitHub Releases automatically
python scripts/download_model.py
```

This will download the model from GitHub Releases and place it in `models/skin_cancer_model.keras`.

### Option 2: Manual Download from GitHub Releases

1. Go to [GitHub Releases](https://github.com/m-aljasem/dermatology-ai-classifier/releases)
2. Download `skin_cancer_model.keras` from the latest release
3. Create the `models/` directory if it doesn't exist:
   ```bash
   mkdir -p models
   ```
4. Place the downloaded file in `models/skin_cancer_model.keras`

### Option 3: Copy from Kaggle Export

If you have the model file from Kaggle:

```bash
# Copy from Kaggle export directory
python scripts/download_model.py --source kaggle --path /path/to/best_model.keras

# Or manually:
mkdir -p models
cp /path/to/best_model.keras models/skin_cancer_model.keras
```

### Option 4: Train Your Own Model

Train the model yourself using the Kaggle notebook:

1. Open `notebooks/02_model_training.ipynb` on Kaggle
2. Run all cells to train the model
3. Download `best_model.keras` or `final_model.keras` from Kaggle outputs
4. Copy it to `models/skin_cancer_model.keras`

## ✅ Verify Setup

After downloading/copying the model, verify it's in place:

```bash
# Check if model exists
ls -lh models/skin_cancer_model.keras

# Should show file size (~50-200 MB)
```

## 🧪 Test the Setup

Run the Streamlit app to verify everything works:

```bash
streamlit run app.py
```

The app will:
- ✅ Show "Model loaded" in the sidebar if the model is found
- ❌ Show "Model not found" with download instructions if missing

## 📁 Expected File Structure

After setup, your project should have:

```
skin-cancer/
├── models/
│   ├── skin_cancer_model.keras  ← Trained model (50-200 MB)
│   ├── label_mapping.json       ← Class mappings
│   └── model_config.json        ← Model configuration
├── app.py
└── ...
```

## 🔧 Troubleshooting

### Model file not found

**Error:** `FileNotFoundError: models/skin_cancer_model.keras`

**Solution:**
1. Verify the file exists: `ls models/skin_cancer_model.keras`
2. Check file permissions: `chmod 644 models/skin_cancer_model.keras`
3. Ensure you're in the project root directory

### Download script fails

**Error:** `Error downloading model: HTTP Error 404`

**Solution:**
- The GitHub release might not exist yet
- Use Option 2 (manual download) or Option 3 (Kaggle export)
- Or train your own model (Option 4)

### Model loading errors

**Error:** `ValueError: Unknown layer: ...`

**Solution:**
- Ensure you're using the correct TensorFlow version (see `requirements.txt`)
- The model might be incompatible with your TensorFlow version
- Try retraining with your current TensorFlow version

## 📝 For Developers

### Updating the Download Script

If you change the GitHub release URL, update `scripts/download_model.py`:

```python
GITHUB_RELEASE_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/latest/download/skin_cancer_model.keras"
```

### Adding Model to Git LFS (Alternative)

If you want to track the model in Git:

```bash
# Install Git LFS
git lfs install

# Track .keras files
git lfs track "*.keras"
git lfs track "models/*.keras"

# Add model
git add models/skin_cancer_model.keras .gitattributes
git commit -m "Add trained model (Git LFS)"
git push
```

**Note:** Git LFS has storage limits on free GitHub accounts (1 GB storage, 1 GB bandwidth/month).

## 🎯 Best Practices

1. **For Development:** Use the download script or copy from Kaggle
2. **For Production:** Host model on cloud storage (S3, Google Cloud Storage) or use model hosting services
3. **For Sharing:** Use GitHub Releases for easy distribution
4. **For CI/CD:** Download model as part of deployment pipeline

## 📚 Related Files

- `scripts/download_model.py` - Automated download script
- `KAGGLE_EXPORT_GUIDE.md` - Guide for exporting from Kaggle
- `README.md` - Main project documentation


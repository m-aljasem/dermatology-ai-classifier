# Streamlit Cloud Deployment Guide

This guide explains how to deploy your Dermatology AI Classifier app to Streamlit Cloud.

## ✅ Prerequisites

Before deploying, ensure:

1. ✅ Your repository is on GitHub
2. ✅ Model file is added via Git LFS (see [SETUP_GIT_LFS.md](SETUP_GIT_LFS.md))
3. ✅ `requirements.txt` is up to date
4. ✅ `app.py` is in the root directory

## 🚀 Quick Deployment Steps

### Step 1: Push Your Code to GitHub

```bash
# Make sure everything is committed
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository: `m-aljasem/dermatology-ai-classifier`
5. Select branch: `main` (or your default branch)
6. Main file path: `app.py`
7. Click **"Deploy!"**

### Step 3: Wait for Deployment

Streamlit Cloud will:
- ✅ Clone your repository
- ✅ Install Git LFS automatically
- ✅ Pull LFS files (your model)
- ✅ Install dependencies from `requirements.txt`
- ✅ Launch your app

**Deployment typically takes 2-5 minutes.**

## 🔍 What Streamlit Cloud Does Automatically

Streamlit Cloud automatically:

1. **Git LFS Support**: Installs Git LFS and pulls LFS files
2. **Dependency Installation**: Runs `pip install -r requirements.txt`
3. **App Launch**: Runs `streamlit run app.py`
4. **Auto-updates**: Redeploys on every push to your main branch

## ⚠️ Important Considerations

### Model File Size

- **Git LFS Storage**: Your model file (~50-200 MB) is stored via Git LFS
- **GitHub LFS Limits**: 
  - Free: 1 GB storage, 1 GB bandwidth/month
  - Your model: ~50-200 MB (well within limits)
- **Streamlit Cloud**: Downloads the model during deployment

### Memory Requirements

- **TensorFlow**: Requires significant memory (~2-4 GB RAM)
- **Streamlit Cloud Free Tier**: 1 GB RAM (may be insufficient)
- **Recommendation**: Consider Streamlit Cloud Pro for better performance

### Startup Time

- **First Deployment**: 5-10 minutes (downloads model, installs dependencies)
- **Subsequent Deployments**: 2-5 minutes (cached dependencies)

## 🔧 Troubleshooting

### Deployment Fails: "Model not found"

**Problem:** Model file not available after deployment

**Solutions:**

1. **Verify Git LFS is set up:**
   ```bash
   git lfs ls-files
   # Should show: models/skin_cancer_model.keras
   ```

2. **Ensure model is committed:**
   ```bash
   git add models/skin_cancer_model.keras
   git commit -m "Add model file"
   git push origin main
   ```

3. **Check .gitattributes:**
   ```bash
   cat .gitattributes
   # Should include: *.keras filter=lfs diff=lfs merge=lfs -text
   ```

### Deployment Fails: "Out of memory"

**Problem:** App crashes due to insufficient memory

**Solutions:**

1. **Upgrade to Streamlit Cloud Pro** (more RAM)
2. **Optimize model loading** (lazy loading)
3. **Use model quantization** (reduce model size)

### Slow Startup Time

**Problem:** App takes too long to start

**Solutions:**

1. **Pre-warm the model** (load on startup)
2. **Use caching** (`@st.cache_resource` for model)
3. **Optimize dependencies** (remove unused packages)

### Dependencies Installation Fails

**Problem:** Some packages fail to install

**Solutions:**

1. **Check requirements.txt** for version conflicts
2. **Pin specific versions** instead of ranges
3. **Remove optional dependencies** if not needed

## 📝 Optimizing for Streamlit Cloud

### 1. Add Model Caching

Update `app.py` to cache the model:

```python
@st.cache_resource
def load_model():
    """Load and cache the model."""
    if WEIGHTS_PATH.exists():
        return tf.keras.models.load_model(str(WEIGHTS_PATH))
    return None

# Use cached model
model = load_model()
```

### 2. Optimize Requirements

Create a minimal `requirements.txt` for Streamlit Cloud:

```txt
streamlit>=1.28.0
tensorflow>=2.13.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python>=4.8.0
scipy>=1.11.0
```

Remove optional dependencies if not needed:
- `kaggle` (not needed for deployment)
- `fastapi`, `uvicorn` (not needed for Streamlit app)
- `mcp` (not needed for Streamlit app)
- `shap`, `lime` (only if explainability is used)

### 3. Add Health Check

Add a health check endpoint (optional):

```python
# In app.py
if st.sidebar.button("Health Check"):
    if WEIGHTS_PATH.exists():
        st.success("✅ Model file found")
        st.success("✅ All systems operational")
    else:
        st.error("❌ Model file missing")
```

## 🎯 Deployment Checklist

Before deploying, verify:

- [ ] Repository is public (or you have Streamlit Cloud Pro)
- [ ] `app.py` is in root directory
- [ ] `requirements.txt` exists and is up to date
- [ ] Model file is committed via Git LFS
- [ ] `.gitattributes` is configured
- [ ] `.streamlit/config.toml` exists (optional)
- [ ] All dependencies are compatible

## 📊 Monitoring Your Deployment

After deployment:

1. **Check Logs**: View deployment logs in Streamlit Cloud dashboard
2. **Test App**: Visit your app URL and test functionality
3. **Monitor Usage**: Check resource usage in dashboard
4. **Set Alerts**: Configure alerts for errors or high usage

## 🔗 Useful Links

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [Streamlit Cloud Status](https://status.streamlit.io/)

## 💡 Pro Tips

1. **Use Environment Variables**: Store sensitive configs in Streamlit Cloud secrets
2. **Enable Auto-refresh**: Set `runner.fastReruns = true` in config
3. **Monitor Performance**: Use Streamlit Cloud analytics
4. **Set Resource Limits**: Configure memory/CPU limits if needed

---

**Ready to deploy?** Follow the Quick Deployment Steps above! 🚀


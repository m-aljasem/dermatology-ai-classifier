# Setting Up Git LFS for Model Files

This guide explains how to add the trained model to the repository using Git LFS (Large File Storage), so users can use it immediately without downloading.

## 🚀 Quick Setup

### Step 1: Install Git LFS

**On Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# Or download from: https://git-lfs.github.com/
```

**On macOS:**
```bash
brew install git-lfs
```

**On Windows:**
Download installer from: https://git-lfs.github.com/

### Step 2: Initialize Git LFS in Your Repository

```bash
# Navigate to your project directory
cd /path/to/skin-cancer

# Initialize Git LFS
git lfs install

# Verify .gitattributes is set up (already created)
cat .gitattributes
```

### Step 3: Add Your Model File

```bash
# Copy your model from Kaggle export
cp /path/to/best_model.keras models/skin_cancer_model.keras

# Add the model file (Git LFS will automatically track it)
git add models/skin_cancer_model.keras

# Verify it's being tracked by LFS
git lfs ls-files

# Commit
git commit -m "Add trained model via Git LFS"

# Push (this will upload the LFS file)
git push origin main
```

## ✅ Verification

After pushing, verify:

1. **Check LFS tracking:**
   ```bash
   git lfs ls-files
   # Should show: models/skin_cancer_model.keras
   ```

2. **Clone on another machine:**
   ```bash
   git clone https://github.com/m-aljasem/dermatology-ai-classifier.git
   cd dermatology-ai-classifier
   git lfs pull  # Download LFS files
   ls -lh models/skin_cancer_model.keras  # Should show the file
   ```

## 📋 What Happens

- **For you (developer):** Model is stored in Git LFS, pushed to GitHub
- **For users:** When they clone the repo, they get the model automatically
- **Git LFS:** Handles large files efficiently (stores pointers in Git, actual files separately)

## ⚠️ Important Notes

1. **Git LFS Storage Limits:**
   - GitHub Free: 1 GB storage, 1 GB bandwidth/month
   - Model file: ~50-200 MB (well within limits)

2. **First-time Setup:**
   - Users need Git LFS installed: `git lfs install`
   - Then clone normally: `git clone ...`
   - LFS files download automatically

3. **If LFS Not Installed:**
   - Users will see a pointer file instead of the actual model
   - They'll need to install Git LFS and run `git lfs pull`

## 🔧 Troubleshooting

### Model file shows as pointer file

**Problem:** File shows as small (~100 bytes) instead of large (~50-200 MB)

**Solution:**
```bash
git lfs pull
```

### Git LFS not installed

**Error:** `git: 'lfs' is not a git command`

**Solution:** Install Git LFS (see Step 1 above)

### LFS file not tracking

**Problem:** Model file not being tracked by LFS

**Solution:**
```bash
# Remove from cache
git rm --cached models/skin_cancer_model.keras

# Re-add (should use LFS now)
git add models/skin_cancer_model.keras

# Verify
git lfs ls-files
```

## 📚 Alternative: Auto-Download on First Run

If you prefer not to use Git LFS, you can modify the app to auto-download the model on first run. See `scripts/download_model.py` for implementation.

## 🎯 Recommended Approach

**Use Git LFS** - It's the standard solution for large files in Git repositories and provides the best user experience (model available immediately after clone).


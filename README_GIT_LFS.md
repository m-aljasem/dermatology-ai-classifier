# Quick Start: Using the Model with Git LFS

The trained model is included in this repository using **Git LFS** (Large File Storage), so you can use it immediately after cloning!

## 🚀 Quick Setup (3 Steps)

```bash
# 1. Clone the repository
git clone https://github.com/m-aljasem/dermatology-ai-classifier.git
cd dermatology-ai-classifier

# 2. Install Git LFS (one-time setup)
git lfs install

# 3. Download the model file
git lfs pull

# 4. Run the app
streamlit run app.py
```

That's it! The model is now available and the app will work.

## 📋 Detailed Steps

### Step 1: Install Git LFS

**Linux:**
```bash
sudo apt-get install git-lfs
```

**macOS:**
```bash
brew install git-lfs
```

**Windows:**
Download from: https://git-lfs.github.com/

### Step 2: Clone Repository

```bash
git clone https://github.com/m-aljasem/dermatology-ai-classifier.git
cd dermatology-ai-classifier
```

### Step 3: Initialize Git LFS (One-time)

```bash
git lfs install
```

This only needs to be done once per machine.

### Step 4: Download Model File

```bash
git lfs pull
```

This downloads the `models/skin_cancer_model.keras` file (~50-200 MB).

### Step 5: Verify Model is Available

```bash
ls -lh models/skin_cancer_model.keras
```

Should show the file size (~50-200 MB).

### Step 6: Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ✅ Verification

After running `git lfs pull`, verify:

```bash
# Check LFS files
git lfs ls-files

# Should show:
# models/skin_cancer_model.keras
```

## 🔧 Troubleshooting

### "git: 'lfs' is not a git command"

**Solution:** Install Git LFS (see Step 1 above)

### Model file shows as small (~100 bytes)

**Problem:** LFS file not downloaded, only pointer file present

**Solution:**
```bash
git lfs pull
```

### "This repository is configured for Git LFS but 'git-lfs' was not found"

**Solution:** Install Git LFS and run `git lfs install`

## 💡 Why Git LFS?

- ✅ Model included in repository (no separate download needed)
- ✅ Works immediately after clone + `git lfs pull`
- ✅ Standard solution for large files in Git
- ✅ Efficient storage (only downloads when needed)

## 📚 More Information

- Full setup guide: [SETUP_GIT_LFS.md](SETUP_GIT_LFS.md)
- Git LFS documentation: https://git-lfs.github.com/


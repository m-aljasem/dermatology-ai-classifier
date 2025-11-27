# Quick Start: Adding Kaggle Export Files

## 🚀 Quick Commands

After downloading files from Kaggle, run these commands:

```bash
# 1. Copy config files to models/
cp label_mapping.json models/
cp model_config.json models/

# 2. Copy visualizations to docs/
cp training_history.png docs/
cp confusion_matrix_normalized.png docs/
cp confusion_matrix_counts.png docs/
cp model_architecture.png docs/ 2>/dev/null || true

# 3. Copy training history CSV
cp training_history.csv docs/

# 4. Verify files are in place
ls -lh models/*.json docs/*.png docs/*.csv

# 5. Add to git (small files only)
git add models/label_mapping.json models/model_config.json
git add docs/training_history.png docs/training_history.csv
git add docs/confusion_matrix_*.png docs/model_architecture.png 2>/dev/null || true

# 6. Commit
git commit -m "Add training artifacts from Kaggle export"

# 7. Push
git push origin main
```

## 📋 Files Checklist

### ✅ Add These (Small Files):
- [ ] `label_mapping.json` → `models/`
- [ ] `model_config.json` → `models/`
- [ ] `training_history.csv` → `docs/`
- [ ] `training_history.png` → `docs/`
- [ ] `confusion_matrix_normalized.png` → `docs/`
- [ ] `confusion_matrix_counts.png` → `docs/`
- [ ] `model_architecture.png` → `docs/` (if available)

### ❌ Do NOT Add These (Large Files):
- [ ] `best_model.keras` (~50-200 MB) - Use Git LFS or GitHub Releases
- [ ] `final_model.keras` (~50-200 MB) - Use Git LFS or GitHub Releases  
- [ ] `export.zip` (~100-400 MB) - Use GitHub Releases

## 📦 For Model Files

**Option 1: GitHub Releases (Easiest)**
1. Go to your GitHub repo → Releases → Create new release
2. Upload `best_model.keras` as a release asset
3. Update README with download link

**Option 2: Git LFS**
```bash
git lfs install
git lfs track "*.keras"
cp best_model.keras models/skin_cancer_model.keras
git add models/skin_cancer_model.keras .gitattributes
git commit -m "Add trained model (Git LFS)"
```

See `KAGGLE_EXPORT_GUIDE.md` for detailed instructions.


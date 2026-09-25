Breast Tumor Classification with Random Forest + Pattern Segmentation

Individual Project — UCI Breast Cancer Wisconsin (Diagnostic)

Required deliverables
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocess.ipynb`
3. `notebooks/03_train_rf_classifier.ipynb`
4. `notebooks/04_evaluate.ipynb`
5. `notebooks/05_clustering.ipynb`
6. `models/rf_model.joblib`
7. `src/inference.py`

Supporting files
- `data/wdbc.data`
- `data/wdbc.names`
- `requirements.txt`
- `outputs/` contains report figures.

Method
- Target: M=1, B=0
- Split: 70/15/15 stratified Train/Validation/Test
- Random seed: 42
- Supervised model: Random Forest only
- Hyperparameter tuning: RandomizedSearchCV
- Decision threshold: 0.20, selected using validation data
- Clustering: StandardScaler + K-means + hierarchical clustering
Reproducibility
Run the notebooks in order. The model notebook saves `models/rf_model.joblib`; the evaluation notebook reports the held-out test metrics; the clustering notebook creates the cluster analysis and plots.

This is an educational/demonstration project and must not be presented as medical advice or a clinical diagnostic tool.

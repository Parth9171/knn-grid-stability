# KNN-Based Electrical Grid Stability Classification

This project predicts the stability class (`stable` / `unstable`) of simulated electrical-grid operating conditions using K-Nearest Neighbour (KNN).

## Project structure
- `app.py` — interactive Streamlit GUI
- `ml_backend.py` — shared model/data functions used by the GUI and notebook
- `notebooks/knn_grid_stability.ipynb` — reproducible ML workflow
- `data/` — downloaded dataset is saved here on first run
- `project_report.md` and `Project_Report.docx` — project report
- `requirements.txt` — dependencies

## Dataset source
Official source: https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data  
Citation: Arzamasov, V. (2018). *Electrical Grid Stability Simulated Data*. UCI Machine Learning Repository. https://doi.org/10.24432/C5PG66  
License: CC BY 4.0.

The project downloads the dataset via `ucimlrepo` when it is not present locally. The official dataset contains 10,000 simulated observations. The target is `stabf`; the 11 predictors are `tau1`–`tau4`, `p2`–`p4`, and `g1`–`g4`. The `p1` column is documented as non-predictive and `stab` is excluded to avoid target leakage.

## Run on Windows
1. Install Python 3.10+.
2. Extract the ZIP and open a terminal in the extracted folder.
3. Create an environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Start the GUI:
   ```bash
   streamlit run app.py
   ```
6. Run the notebook:
   ```bash
   jupyter notebook
   ```
   Open `notebooks/knn_grid_stability.ipynb` and run all cells.

## Evaluation integrity
The notebook selects K using cross-validation on the training partition, then evaluates the selected pipeline on a held-out test partition. Do not claim a 90%+ result until the notebook has been run and the actual test accuracy is visible in the output. This is a simulated dataset and is not intended for real-time grid operation or protection.

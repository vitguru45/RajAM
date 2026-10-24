# RajAM — Hardness & Microstructure Predictor for Laser-AM Steels

Streamlit app (Raj3DLabs family) by Dr. Gururaja S, CIMR, VIT Vellore.

**Alloys:** SS316L · 18Ni-300 maraging · 17-4PH · H13  **Processes:** LPBF/SLM and L-DED

## Run locally (Windows / PyCharm terminal)
```
pip install -r requirements.txt
streamlit run app.py
```
First load takes ~30 s (it calibrates and runs leave-one-out validation once, then caches).

## Deploy on Streamlit Community Cloud
Push this folder (app.py, physics.py, models.py, data/, requirements.txt) to a GitHub repo
(e.g. vitguru45/RajAM) → share.streamlit.io → New app → main file `app.py`.

## Files
| File | Purpose |
|---|---|
| `app.py` | UI: inputs, predictions, process map, suggestions/inverse design, melt pool, accuracy, references, counter |
| `physics.py` | Analytical models (Rosenthal, King keyhole, Tang LoF, balling, cell spacing, Hall-Petch, aging/tempering kinetics) |
| `models.py` | Dataset loading, calibration, Random Forest, Gradient Boosting, physics-informed GPR, LOO-CV |
| `data/rajam_dataset.csv` | 42 experimental records from published papers (sources & DOIs in the file) |

## Notes
* Visit/prediction counter and usage log are JSON/CSV files beside app.py. On Streamlit Cloud they reset whenever the app restarts or redeploys.
* Extend the data: add rows to `data/rajam_dataset.csv` (same columns) or upload a CSV in the app's "Data" tab.
* 316L has 33 tabulated points; maraging / H13 / 17-4PH have only a few literature anchors — treat those predictions as estimates.

# ❤️ HeartBeat AI — AI-Assisted ECG Heartbeat Screening

**Domain:** Health Operations · **Event:** AICON'26 Build With AI

> Screening-support prototype. **Not a medical diagnostic device.**

🔗 **Live demo:** _paste your streamlit.app link here_
🔗 **Code:** _paste your GitHub repository link here_

## 1. Problem
Heart-rhythm problems (arrhythmias) are often found late. Reading an ECG needs a cardiologist, and many clinics and rural health centres do not have one. Doctors also spend time reading long recordings in which most beats are normal.

## 2. Solution
HeartBeat AI classifies **every heartbeat** in an ECG recording as *Normal* or *Possible arrhythmia* and shows the result on a dashboard. It acts as a **triage helper**: the clinician can look first at the segments the AI flagged instead of reading the whole recording.

**Problem → Data → AI → Output → Impact**

| Stage | What we did |
|---|---|
| Data | MIT-BIH Arrhythmia Database (real ECGs, beats labelled by cardiologists) |
| AI | Random Forest classifier (80 trees, max depth 18) on 0.6 s windows around each beat (216 samples, z-score normalised) |
| Output | Gradio dashboard: real ECG, per-beat prediction, rhythm score, flagged-beat count, confidence |
| Impact | Faster first-level screening and prioritisation of recordings for review |

## 3. How it works
1. Download MIT-BIH records with `wfdb` (lead MLII).
2. Cut a 0.6 s window around each annotated beat (0.2 s before, 0.4 s after) and normalise it.
3. Label beats Normal (N, L, R, e, j) or Possible arrhythmia (A, a, J, S, V, E, F, !).
4. Train a Random Forest (`class_weight="balanced"`, 80 trees, max depth 18) on 22 patients. The deployed model is this light 2 MB version; the Colab notebook also contains a larger 200-tree experiment.
5. Test on 19 **different** patients never seen in training (standard inter-patient split, paced records excluded).
6. The dashboard runs the saved model on every beat; a beat with probability ≥ 0.5 is flagged.

## 4. Results (19 unseen patients, 43,826 beats)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Possible arrhythmia | 0.48 | 0.56 | 0.52 |

Overall accuracy: **87.6%** (deployed model).
Abnormal beats are the hard class: the model catches about half of them and also raises false alarms.

## 5. Limitations (honest)
- Abnormal-beat recall is ~56% and precision ~48%: about half of abnormal beats are caught and there are false alarms. This is the main weakness.
- Beat positions come from the dataset annotations. A live system would need an R-peak detector.
- Only one ECG lead (MLII) and a binary Normal/Abnormal label are used.
- Not clinically validated; not a diagnostic device.

## 6. Next steps
Add RR-interval (timing between beats) features, try a 1D-CNN, add an R-peak detector, and connect to wearable ECG signals.

## 7. SDG & industry relevance
Supports **SDG 3 – Good Health and Well-being**. Relevant to clinics, diagnostic centres and telemedicine for triage and for saving specialist time.

## 8. Tools, data and disclosure
- **Dataset (public):** MIT-BIH Arrhythmia Database from PhysioNet (Open Data Commons Attribution License).
  - Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng Med Biol 20(3):45–50 (2001).
  - Goldberger AL et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 101(23):e215–e220 (2000).
- **Libraries:** Python, wfdb, NumPy, scikit-learn, joblib, matplotlib, Streamlit (deployed dashboard), Gradio (Colab prototype).
- **AI coding assistant:** an AI assistant (Claude) was used to help write and debug code. We trained and evaluated the model ourselves and can explain every part of it.
- _Add one line here about anything that existed before the event (if any)._

## 9. Run it yourself
**Colab:** open `HeartBeat_AI_final.ipynb`, run all cells.
**Locally / Streamlit:** keep `streamlit_app.py`, `requirements.txt`, `heartbeat_model.pkl`, `metrics.json` and `.streamlit/config.toml` in one folder, then `pip install -r requirements.txt` and `streamlit run streamlit_app.py`. (First start downloads the demo patients.)

## 10. Files
`streamlit_app.py` · `requirements.txt` · `heartbeat_model.pkl` · `metrics.json` · `.streamlit/config.toml` · `HeartBeat_AI_final.ipynb`

## Team
_Add team name and members._

# SHL--audio-scoring-pipeline
# Audio Scoring AI: Dual-Pipeline Feature Extraction

## Project Overview
This repository contains a machine learning pipeline designed to automatically grade the grammar and fluency of spoken audio files. 

Faced with a highly constrained dataset (only 409 training samples), training a deep neural network directly on raw audio waveforms would lead to severe overfitting. To solve this, I engineered a **Dual-Pipeline Feature Extraction** architecture. By extracting highly interpretable numerical features using state-of-the-art NLP and acoustic libraries, the data is perfectly formatted for an XGBoost Regressor.

## Architecture

### 1. The NLP Pipeline (Grammar)
* **OpenAI Faster-Whisper:** Utilized GPU-accelerated automatic speech recognition (ASR) to transcribe the raw `.wav` files into text.
* **LanguageTool:** Processed the transcripts through an NLP grammar engine to extract explicit error counts, word counts, and text-based error rates.

### 2. The Acoustic Pipeline (Fluency)
* **Librosa:** Mathematically analyzed the raw audio waveforms to isolate active speaking durations from dead air. 
* Extracted specific fluency metrics including the speaker's `silence_ratio` (hesitation) and `speaking_rate` (words per second).

### 3. The Predictive Model
* **Algorithm:** XGBoost Regressor
* **Validation:** 5-Fold Cross Validation
* **Interpretability:** Because the model relies on engineered features rather than a black-box deep learning approach, it allows for clear feature importance mapping. The model successfully demonstrates that it weighs grammatical accuracy alongside acoustic hesitation to predict the final Mean Opinion Score (MOS).

## Libraries Used
`pandas`, `numpy`, `xgboost`, `faster-whisper`, `language-tool-python`, `librosa`, `scikit-learn`, `seaborn`, `matplotlib`


## 📂 Repository Structure

* `data/` - Contains the pre-computed prediction outputs (the CSV files). **These are the required inputs for the final blending script.**
* `notebook5faa0bc8b1.ipynb` - The main notebook containing the feature extraction and model training pipelines.
* `blendnotbeook.ipynb` - The final ensembling script that blends the independent model predictions.

## ⚙️ How to Run the Ensemble

To generate the final blended submission:
1. Ensure the `data/` folder contains your two prediction CSV files.
2. Run `blendnotbeook.ipynb`. 
3. The script will automatically locate the CSVs, apply the weighted average, and output the final `submission_final_blend.csv`.

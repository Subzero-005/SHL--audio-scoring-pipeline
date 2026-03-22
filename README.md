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

# SHL Audio Grammar Scoring Challenge 🎙️🤖

**Final Leaderboard Rank:** 28th (Top 25%)  
**Best Evaluation Score (RMSE):** 0.829  

An end-to-end Machine Learning pipeline designed to predict the Mean Opinion Score (MOS) of spoken audio based on grammar, fluency, and acoustic clarity. This project tackles the challenge of predicting human-assigned scores on a highly constrained dataset (409 training samples) by combining interpretable handcrafted NLP features with latent deep learning acoustic embeddings.

## 🧠 Architecture Overview

To prevent overfitting on the small dataset and maximize performance, the project was developed in three distinct evolutionary stages, ultimately resulting in a weighted hybrid ensemble.

### Stage 1: Dual-Pipeline Feature Engineering (Score: 0.886)
An interpretable, rules-based approach that mimics how a human evaluates speech.
* **NLP/Lexical Pipeline:** Transcribed audio using GPU-accelerated `faster-whisper`. Engineered text features including grammar error rates (via `language-tool-python`), lexical diversity, and filler word ratios (um, uh, like).
* **Acoustic/Fluency Pipeline:** Processed raw waveforms using `librosa` to extract speaking rate (words per second), silence-to-speech ratios, pause intervals, and RMS energy variance.
* **Model:** A heavily regularized **XGBoost Regressor** (5-Fold CV) trained on these 9 core features, ensuring high interpretability and robust resistance to outliers.

### Stage 2: Deep Learning Foundation Model (Score: 0.857)
Transitioned to a latent feature extraction architecture to capture subtle acoustic patterns (tone, confidence, accent) that manual feature engineering misses.
* **Extractor:** Passed raw 16kHz audio through the pre-trained **Wav2Vec 2.0 Base** transformer (`facebook/wav2vec2-base`), extracting the 768-dimensional contextualized hidden states.
* **Model:** Replaced XGBoost with a **Ridge Regressor** (`alpha=50.0`). Because the feature space (768) exceeded the number of training samples (409), strict L2 regularization was required to combat the Curse of Dimensionality and prevent severe overfitting.

### Stage 3: The Final Ensemble Blend (Score: 0.829)
Attempting to train a single model on both the 9 NLP features and the 768 DL features simultaneously resulted in "feature drowning" (Score: 0.875). 

To solve this, I implemented an **Independent Prediction Blending** strategy. By allowing the Deep Learning model (which understands tone) and the XGBoost model (which enforces strict grammatical rules) to predict independently, I averaged their outputs to cancel out individual model biases.
* **Final Weights:** `0.70 * (Wav2Vec Predictions) + 0.30 * (XGBoost NLP Predictions)`

## 📂 Repository Structure

* `data/` - Contains the pre-computed prediction outputs (the CSV files). **These are the required inputs for the final blending script.**
* `notebook5faa0bc8b1.ipynb` - The main notebook containing the feature extraction and model training pipelines.
* `blend_notebook.ipynb` - The final ensembling script that blends the independent model predictions.

## ⚙️ How to Run the Ensemble

To generate the final blended submission:
1. Ensure the `data/` folder contains your two prediction CSV files.
2. Run `blend_notebook.ipynb`. 
3. The script will automatically locate the CSVs, apply the weighted average, and output the final `submission_final_blend.csv`.

**🏆 Performance Note:** Attempting to train a single model on all features simultaneously results in "feature drowning" with a score of **0.875**. Running this ensemble notebook alone to independently blend the models successfully bypasses this issue, providing the final optimized score of **0.829**.

## 🛠️ Tech Stack
* **Deep Learning & Audio:** PyTorch, Transformers (HuggingFace), Wav2Vec 2.0, Librosa, Faster-Whisper.
* **Machine Learning:** XGBoost, Scikit-Learn (Ridge, KFold Cross-Validation).
* **Data Processing:** Pandas, NumPy.

## 💡 Key Learnings & Takeaways
1. **The limits of concatenation:** Blindly concatenating dense neural embeddings with sparse handcrafted features often degrades model performance due to scale imbalances.
2. **Handling Data Scarcity:** When fine-tuning a massive transformer is computationally prohibitive or risks overfitting due to low N, utilizing the model purely as a fixed feature extractor paired with a heavily regularized linear model is highly effective.
3. **The power of Ensembling:** Averaging the predictions of fundamentally different model architectures (Tree-based vs. Distance-based) almost always yields a superior, more generalized result.

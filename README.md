# Music Preference Prediction

A machine learning project exploring whether personal music preferences can be predicted from custom audio features extracted directly from audio files.

## Project Goal

The objective of this project is to investigate how well audio characteristics alone can explain and predict my personal song ratings.

The project combines audio signal processing, feature engineering, database management, and machine learning to build a personalized music preference model.

## Dataset

* 125 manually rated songs
* Rating scale: 1–10
* Features extracted directly from local MP3 files
* SQLite database for feature storage and rating management

To analyze your own music collection, place MP3 files in:

```text
data/songs_mp3/
```

Audio files are not included in this repository.

## Extracted Features

### Rhythmic Features

* BPM (Tempo)
* Timing Deviation
* Asymmetry Bias
* Asymmetry Index

### Dynamic Features

* Dynamic Variation Score

### Spectral Features

* Sub Bass
* Bass
* Low Mid
* Mid
* High Mid
* Presence
* Brilliance
* Spectral Spread
* Spectral Balance

## Machine Learning Models

The following models were evaluated:

* Ridge Regression (linear baseline)
* Random Forest Regression
* Extra Trees Regression

Model performance was evaluated using **5-Fold Cross Validation**.

## Results

### Best Performing Model: Extra Trees Regression

| Metric               | Value |
| -------------------- | ----: |
| Cross Validation R²  |  0.27 |
| Cross Validation MAE |  1.63 |

The Extra Trees model achieved the best predictive performance and outperformed both the linear baseline and the Random Forest model.

## Feature Importance

The Extra Trees model identified several spectral characteristics as the most influential predictors.

![Feature Importance](images/feature_importance.png)

## Actual vs Predicted Ratings

The plot below compares the predicted ratings with the manually assigned ratings.

![Actual vs Predicted](images/actual_vs_predicted.png)

## Technology Stack

* Python
* SQLite
* Pandas
* NumPy
* Librosa
* Scikit-Learn
* Matplotlib

## Repository Structure

```text
data/
├── songs.db
└── songs_mp3/

images/
├── actual_vs_predicted.png
└── feature_importance.png

src/
├── extract_features.py
├── init_db.py
└── train_model.py
```

## Future Improvements

Potential future extensions include:

* Harmonic and chroma-based features
* Key and mode detection (major/minor)
* Larger and more diverse song datasets
* Additional machine learning models and hyperparameter optimization

```
```

# Music Preference Prediction

Predicting personal music preferences using custom audio features and machine learning.

## Project Goal

The goal of this project is to investigate how well audio characteristics alone can predict my personal music preferences.

## Dataset

* 101 manually rated songs
* Rating scale: 1–10
* Audio features extracted from local MP3 files using Python

To analyze your own music collection, place MP3 files in:

```text
data/songs_mp3/
```

Audio files are not included in this repository.

## Features

The extracted feature set includes:

* BPM (tempo)
* Dynamic Variation Score
* Sub Bass
* Bass
* Low Mid
* Mid
* High Mid
* Presence
* Brilliance
* Spectral Spread
* Spectral Balance
* Timing Deviation
* Asymmetry Bias
* Asymmetry Index

## Models

The following machine learning models were evaluated:

* Ridge Regression
* Random Forest Regression

Evaluation was performed using 5-fold cross validation.

## Results

| Metric               | Value |
| -------------------- | ----: |
| Cross Validation R²  |  0.29 |
| Cross Validation MAE |  1.58 |

The Random Forest model significantly outperformed the linear baseline, indicating that music preference depends on non-linear interactions between audio features.

## Feature Importance

![Feature Importance](images/feature_importance.png)

## Actual vs Predicted Ratings

![Actual vs Predicted](images/actual_vs_predicted.png)

## Technologies

* Python
* SQLite
* Pandas
* Scikit-Learn
* Librosa
* Matplotlib

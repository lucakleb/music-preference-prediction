# Music Preference Prediction

Predicting personal music preferences using custom audio features and machine learning.

Place your local `.mp3` files in:

data/songs_mp3/

The audio files are not included in this repository.

## Dataset

- 101 manually rated songs
- Rating scale: 1–10

## Features

- BPM
- Dynamic Variation Score
- Spectral Features
- Timing Deviation
- Asymmetry Metrics

## Models

- Ridge Regression
- Random Forest Regression

## Results

| Metric | Value |
|----------|----------:|
| CV R² | 0.29 |
| CV MAE | 1.58 |

## Feature Importance

![Feature Importance](images/feature_importance.png)

## Actual vs Predicted

![Actual vs Predicted](images/actual_vs_predicted.png)
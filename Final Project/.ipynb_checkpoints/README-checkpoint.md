# Satellite Image Land-Use Classifier & Temporal Change Detector

This repository contains a complete computer vision pipeline to classify land-use types from satellite imagery and detect temporal changes using embeddings.

## Datasets
- **EuroSAT**: 27,000 images, 10 classes (used for training and validation)
- **UC Merced Land Use**: 2,100 images, 21 classes (used as a holdout test set)

## Project Structure
- `data_setup.py`: Downloads datasets, performs spatial block splitting, and creates class distribution plots.
- `models.py`: Defines the Baseline 3-layer CNN and the ResNet-18 transfer learning models.
- `train_classifier.py`: Training script for models with two-phase fine-tuning, generates ablation tables and checkpoints.
- `change_detector.py`: Extracts embeddings, computes cosine similarity, generates ROC curve, and outputs change heatmaps.
- `app.py`: Streamlit dashboard for interactive classification and change detection.
- `notebooks/`: Jupyter notebooks containing error analysis and spatial leakage experiments.

## Setup Instructions
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run data setup to download and prepare the datasets:
   ```bash
   python data_setup.py
   ```
3. Train the models:
   ```bash
   python train_classifier.py
   ```
4. Run the change detector evaluation:
   ```bash
   python change_detector.py
   ```
5. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

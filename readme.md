# Skin Disease Classification with Explainability

This project focuses on building an interpretable computer vision model for classifying skin diseases using the HAM10000 dataset, with an emphasis on model explainability.

## Project Overview

Skin disease diagnosis through computer vision has enormous potential, but for medical applications, mere classification accuracy is insufficient. Doctors and patients need to understand *why* a model makes specific predictions. This project implements and compares various explainability techniques to make the classification model more transparent and trustworthy.

## Dataset

The HAM10000 dataset ("Human Against Machine with 10,000 training images") contains dermatoscopic images of common pigmented skin lesions across seven categories:

1. Actinic keratoses and intraepithelial carcinoma (akiec)
2. Basal cell carcinoma (bcc)
3. Benign keratosis-like lesions (bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic nevi (nv)
7. Vascular lesions (vasc)

## Project Structure

The project includes the following key components:

1. **Data Loading and Preprocessing**
   - Dataset download and organization
   - Data exploration and visualization
   - Data augmentation to handle class imbalance
   - Train/validation/test splitting

2. **Model Architecture**
   - Transfer learning with ResNet50
   - Fine-tuning for skin disease classification
   - Training with appropriate loss functions and optimizers

3. **Explainability Techniques**
   - GradCAM and its variants (GradCAM++, XGradCAM, etc.)
   - LIME (Local Interpretable Model-agnostic Explanations)
   - SHAP (SHapley Additive exPlanations)
   - Integrated Gradients
   - Occlusion Sensitivity

4. **Explainability Experiments**
   - Comparison of different techniques
   - Advanced visualizations
   - Metrics to evaluate explanation quality

5. **Deployment**
   - Model export for inference
   - Interactive web interface with Gradio
   - Integrated explainability in deployment

## Files Description

- `skin_disease_classification.py`: Main script containing data loading, model building, training, and evaluation
- `explainability_module.py`: Advanced explainability techniques and visualization functions
- `deployment_script.py`: Web interface for model deployment with integrated explainability
- `inference.py`: Standalone script for model inference

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- torchvision
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- pytorch-grad-cam
- lime
- shap
- captum
- gradio (for deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/skin-disease-classification.git
cd skin-disease-classification

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

1. Data preparation and model training:
```bash
python skin_disease_classification.py
```

2. Run the web interface:
```bash
python deployment_script.py
```

## Explainability Techniques

This project implements and compares various explainability techniques:

### GradCAM
GradCAM produces a coarse localization map highlighting important regions in the image for predicting the concept. It uses the gradients of the target class flowing into the final convolutional layer.

### LIME
LIME (Local Interpretable Model-agnostic Explanations) explains the predictions by approximating the model locally with an interpretable model. It perturbs the input and observes how the predictions change.

### SHAP
SHAP (SHapley Additive exPlanations) assigns each feature an importance value for a particular prediction based on game theory.

### Integrated Gradients
Integrated Gradients attributes the prediction to input features by accumulating gradients along a path from a baseline to the input.

### Occlusion Sensitivity
This technique systematically occludes different portions of the input image and monitors the change in prediction to identify important regions.

## Improving Explainability

The project includes several experiments to enhance explainability:

1. **Combined Visualization**: Presenting multiple explainability techniques side-by-side
2. **Comparative Analysis**: Evaluating the consistency between different techniques
3. **Multi-scale Explanations**: Generating explanations at various levels of granularity
4. **Class Discrimination**: Comparing explanations between target and alternative classes

## Model Deployment

The model is deployed using Gradio, providing an intuitive web interface where users can:

1. Upload images of skin lesions
2. Get predictions with confidence scores
3. View various explanations for the predictions
4. Compare different explainability techniques

## Results

The project evaluates both the classification performance (accuracy, precision, recall, F1-score) and the quality of explanations (localization, consistency, and clinical relevance).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The HAM10000 dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- PyTorch and torchvision teams
- Explainability libraries: pytorch-grad-cam, lime, shap, captum

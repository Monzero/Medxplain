import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
from tqdm import tqdm
import zipfile
import requests
from io import BytesIO

# For deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
import torch.nn.functional as F

# For model evaluation
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# For explainability
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import lime
from lime import lime_image
import shap

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths
DATA_DIR = 'data/HAM10000/'
METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMAGES_PATH = os.path.join(DATA_DIR, 'images')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)

# Function to download and extract HAM10000 dataset
def download_ham10000():
    """
    Download and extract the HAM10000 dataset if it doesn't exist locally
    """
    # Check if the dataset exists
    if os.path.exists(METADATA_PATH) and len(os.listdir(IMAGES_PATH)) > 0:
        print("Dataset already exists locally. Skipping download.")
        return
    
    print("Downloading HAM10000 dataset...")
    
    # URLs for the dataset (these are example URLs and might need to be updated)
    metadata_url = "https://dataverse.harvard.edu/api/access/datafile/3172592"
    images_part1_url = "https://dataverse.harvard.edu/api/access/datafile/3172593"
    images_part2_url = "https://dataverse.harvard.edu/api/access/datafile/3172594"
    
    # Download metadata
    response = requests.get(metadata_url)
    if response.status_code == 200:
        with open(METADATA_PATH, 'wb') as f:
            f.write(response.content)
        print("Metadata downloaded successfully.")
    else:
        print(f"Failed to download metadata. Status code: {response.status_code}")
        return
    
    # Download and extract images part 1
    response = requests.get(images_part1_url)
    if response.status_code == 200:
        z = zipfile.ZipFile(BytesIO(response.content))
        z.extractall(DATA_DIR)
        print("Images part 1 downloaded and extracted successfully.")
    else:
        print(f"Failed to download images part 1. Status code: {response.status_code}")
    
    # Download and extract images part 2
    response = requests.get(images_part2_url)
    if response.status_code == 200:
        z = zipfile.ZipFile(BytesIO(response.content))
        z.extractall(DATA_DIR)
        print("Images part 2 downloaded and extracted successfully.")
    else:
        print(f"Failed to download images part 2. Status code: {response.status_code}")

# Download the dataset
download_ham10000()

# 1. Data Loading and Preprocessing
# --------------------------------

class HAM10000Dataset(Dataset):
    """
    HAM10000 Dataset Class
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(IMAGES_PATH, self.df.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = self.df.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_data():
    """
    Load and preprocess the HAM10000 dataset
    """
    # Read metadata
    df = pd.read_csv(METADATA_PATH)
    
    # Display dataset information
    print("Dataset Information:")
    print(f"Total number of images: {len(df)}")
    print(f"Number of classes: {df['dx'].nunique()}")
    print(f"Class distribution: \n{df['dx'].value_counts()}")
    
    # Encode class labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['dx'])
    class_names = le.classes_
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='dx', data=df)
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Create datasets
    train_dataset = HAM10000Dataset(train_df[['image_id', 'label']], transform=train_transform)
    val_dataset = HAM10000Dataset(val_df[['image_id', 'label']], transform=val_transform)
    test_dataset = HAM10000Dataset(test_df[['image_id', 'label']], transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, class_names, test_df

# 2. Model Building
# ----------------

class SkinLesionModel(nn.Module):
    """
    CNN model for skin lesion classification
    """
    def __init__(self, num_classes):
        super(SkinLesionModel, self).__init__()
        # Load a pre-trained ResNet50 model
        self.model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
            
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def get_gradcam_layer(self):
        """Return the target layer for GradCAM"""
        return self.model.layer4[-1]

def train_model(model, train_loader, val_loader, num_epochs=20):
    """
    Train the model
    """
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training statistics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation statistics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.savefig('training_history.png')
    plt.close()
    
    return model, history

def evaluate_model(model, test_loader, class_names):
    """
    Evaluate the model on the test set
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print("Classification Report:")
    print(df_report)
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return df_report, cm, all_preds, all_labels

# 3. Explainability Techniques
# ---------------------------

def get_gradcam_visualization(model, img_tensor, target_class=None):
    """
    Generate Grad-CAM visualization for the given image
    """
    # Prepare the input
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Create a GradCAM object
    cam = GradCAM(model=model, target_layers=[model.get_gradcam_layer()])
    
    # Define target
    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Convert tensor to numpy for visualization
    rgb_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    
    # Overlay CAM on original image
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return cam_image, grayscale_cam

def get_lime_explanation(model, img_tensor, class_names):
    """
    Generate LIME explanation for the given image
    """
    # Create the LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Convert the PyTorch tensor to a numpy array for LIME
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # Define the prediction function for LIME
    def batch_predict(images):
        batch = torch.stack(tuple(torch.from_numpy(i.transpose(2, 0, 1)) for i in images))
        batch = batch.to(device)
        model.eval()
        output = model(batch)
        return output.detach().cpu().numpy()
    
    # Generate explanation
    explanation = explainer.explain_instance(
        img_np, 
        batch_predict, 
        top_labels=5, 
        hide_color=0, 
        num_samples=1000
    )
    
    # Get the top predicted class
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        _, pred_class = torch.max(output, 1)
        pred_class = pred_class.item()
    
    # Get the explanation for the predicted class
    temp, mask = explanation.get_image_and_mask(
        pred_class, 
        positive_only=True, 
        num_features=5, 
        hide_rest=True
    )
    
    # Create the visualization
    lime_img = mark_boundaries(temp, mask)
    
    return lime_img, explanation, pred_class

def get_shap_explanation(model, img_tensor, background_imgs):
    """
    Generate SHAP explanation for the given image
    """
    # Convert background images to a tensor
    background = torch.stack(background_imgs).to(device)
    
    # Define a function to get model outputs
    def model_output(images):
        model.eval()
        return model(images)
    
    # Create the explainer
    explainer = shap.DeepExplainer(model_output, background)
    
    # Get SHAP values
    input_tensor = img_tensor.unsqueeze(0).to(device)
    shap_values = explainer.shap_values(input_tensor)
    
    # Prepare the image for visualization
    img_np = img_tensor.cpu().numpy()
    
    # Compute absolute sum of SHAP values across channels for each pixel
    shap_abs_sum = np.abs(np.array(shap_values)).sum(axis=1).sum(axis=1)
    
    return shap_values, shap_abs_sum, img_np

# 4. Experiments to Improve Explainability
# ---------------------------------------

def combined_explainability(model, img_tensor, class_names, background_imgs):
    """
    Combine different explainability methods for a more comprehensive visualization
    """
    # Get individual explanations
    gradcam_img, gradcam_map = get_gradcam_visualization(model, img_tensor)
    lime_img, lime_exp, pred_class = get_lime_explanation(model, img_tensor, class_names)
    shap_values, shap_abs_sum, img_np = get_shap_explanation(model, img_tensor, background_imgs)
    
    # Create a composite visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Grad-CAM
    axes[0, 1].imshow(gradcam_img)
    axes[0, 1].set_title('Grad-CAM Visualization')
    axes[0, 1].axis('off')
    
    # LIME
    axes[1, 0].imshow(lime_img)
    axes[1, 0].set_title('LIME Explanation')
    axes[1, 0].axis('off')
    
    # SHAP
    shap_img = shap_values[pred_class][0].transpose(1, 2, 0)
    abs_shap_img = np.abs(shap_img).sum(axis=2)
    # Normalize for visualization
    abs_shap_img = (abs_shap_img - abs_shap_img.min()) / (abs_shap_img.max() - abs_shap_img.min() + 1e-10)
    axes[1, 1].imshow(abs_shap_img, cmap='hot')
    axes[1, 1].set_title('SHAP Importance')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    return fig

def evaluate_explainability(model, test_loader, class_names, test_df):
    """
    Evaluate different explainability methods on test images
    """
    model.eval()
    
    # Get a batch of images for background (for SHAP)
    background_imgs = []
    for images, _ in test_loader:
        background_imgs = list(images[:10])
        break
    
    # Select a few test images for visualization
    sample_indices = np.random.choice(len(test_df), 5, replace=False)
    sample_images = []
    
    for i, idx in enumerate(sample_indices):
        img_path = os.path.join(IMAGES_PATH, test_df.iloc[idx]['image_id'] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img)
        
        # Get true class
        true_class = test_df.iloc[idx]['label']
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(device))
            _, pred_class = torch.max(output, 1)
            pred_class = pred_class.item()
        
        # Generate combined explanation
        fig = combined_explainability(model, img_tensor, class_names, background_imgs)
        
        # Add class information
        fig.suptitle(f"True: {class_names[true_class]} | Predicted: {class_names[pred_class]}", fontsize=16)
        
        # Save the visualization
        plt.savefig(f'explanation_sample_{i+1}.png')
        plt.close(fig)
        
        sample_images.append((img_tensor, true_class, pred_class))
    
    print("Saved explainability visualizations for 5 sample images.")
    
    # Experiment: Compare different Grad-CAM variants
    # Select one sample image
    img_tensor, true_class, _ = sample_images[0]
    
    # Define different Grad-CAM variants
    cam_algorithms = {
        'GradCAM': GradCAM,
        'GradCAM++': GradCAMPlusPlus,
        'ScoreCAM': ScoreCAM,
        'XGradCAM': XGradCAM,
        'AblationCAM': AblationCAM
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original image
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Generate visualizations for each method
    for i, (name, algorithm) in enumerate(cam_algorithms.items(), 1):
        try:
            # Create the CAM object
            cam = algorithm(model=model, target_layers=[model.get_gradcam_layer()])
            
            # Generate CAM
            input_tensor = img_tensor.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(true_class)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Overlay CAM on original image
            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            # Add to plot
            axes[i].imshow(cam_image)
            axes[i].set_title(name)
            axes[i].axis('off')
        except Exception as e:
            print(f"Error with {name}: {e}")
            axes[i].text(0.5, 0.5, f"Error: {name}", ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_comparison.png')
    plt.close()
    
    return sample_images

# 5. Model Export and Deployment
# ----------------------------

def export_model(model, class_names):
    """
    Export the model for deployment
    """
    # Save the model architecture and weights
    torch.save(model.state_dict(), 'model_weights.pth')
    torch.save(model, 'full_model.pth')
    
    # Save as TorchScript model for deployment
    model.eval()
    example = torch.rand(1, 3, 224, 224).to(device)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save('model_scripted.pt')
    
    # Save class names
    np.save('class_names.npy', class_names)
    
    print("Model exported successfully.")
    
    # Create a simple inference function for deployment
    def inference(img_path, model_path='model_scripted.pt', class_names_path='class_names.npy'):
        # Load model
        model = torch.jit.load(model_path)
        model.eval()
        
        # Load class names
        class_names = np.load(class_names_path, allow_pickle=True)
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            pred_class = predicted.item()
        
        return {
            'predicted_class': class_names[pred_class],
            'confidence': probabilities[0][pred_class].item(),
            'probabilities': {class_names[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }
    
    # Save the inference function code
    with open('inference.py', 'w') as f:
        f.write("""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def inference(img_path, model_path='model_scripted.pt', class_names_path='class_names.npy'):
    # Load model
    model = torch.jit.load(model_path)
    model.eval()
    
    # Load class names
    class_names = np.load(class_names_path, allow_pickle=True)
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        pred_class = predicted.item()
    
    return {
        'predicted_class': class_names[pred_class],
        'confidence': probabilities[0][pred_class].item(),
        'probabilities': {class_names[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    }

# Example usage:
# result = inference('path/to/your/image.jpg')
# print(result)
""")
    
    print("Inference script created: inference.py")

# Main function to run the pipeline
def main():
    """
    Main function to run the entire pipeline
    """
    # 1. Load and preprocess data
    print("1. Loading and preprocessing data...")
    train_loader, val_loader, test_loader, class_names, test_df = load_data()
    
    # 2. Build and train the model
    print("\n2. Building and training the model...")
    model = SkinLesionModel(num_classes=len(class_names)).to(device)
    model, history = train_model(model, train_loader, val_loader, num_epochs=10)
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 3. Evaluate the model
    print("\n3. Evaluating the model...")
    report, cm, all_preds, all_labels = evaluate_model(model, test_loader, class_names)
    
    # 4. Apply explainability techniques
    print("\n4. Applying explainability techniques...")
    sample_images = evaluate_explainability(model, test_loader, class_names, test_df)
    
    # 5. Export the model for deployment
    print("\n5. Exporting the model for deployment...")
    export_model(model, class_names)
    
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()

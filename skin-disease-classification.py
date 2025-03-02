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

# For deep learning with TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For model evaluation
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Check for GPU
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", len(tf.config.list_physical_devices('GPU')) > 0)
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU Devices:", tf.config.list_physical_devices('GPU'))

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
    
    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Add path column to dataframes
    train_df['path'] = train_df['image_id'].apply(lambda x: os.path.join(IMAGES_PATH, x + '.jpg'))
    val_df['path'] = val_df['image_id'].apply(lambda x: os.path.join(IMAGES_PATH, x + '.jpg'))
    test_df['path'] = test_df['image_id'].apply(lambda x: os.path.join(IMAGES_PATH, x + '.jpg'))
    
    return train_df, val_df, test_df, class_names

def create_data_generators(train_df, val_df, test_df, batch_size=32, img_size=(224, 224)):
    """
    Create data generators for training, validation and test sets
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='path',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

# 2. Model Building
# ----------------

def build_model(num_classes, img_size=(224, 224)):
    """
    Build a model using transfer learning with ResNet50
    """
    # Load pre-trained ResNet50 without top layer
    base_model = applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(*img_size, 3)
    )
    
    # Freeze early layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, val_generator, epochs=20):
    """
    Train the model with appropriate callbacks
    """
    # Callbacks
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.savefig('training_history.png')
    plt.close()
    
    return history, model

def evaluate_model(model, test_generator, class_names):
    """
    Evaluate the model on the test set
    """
    # Predict on test data
    test_generator.reset()
    predictions = model.predict(
        test_generator,
        steps=len(test_generator),
        verbose=1
    )
    
    # Get predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes
    true_classes = test_generator.classes
    
    # Calculate classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print("Classification Report:")
    print(df_report)
    
    # Plot confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return df_report, cm, true_classes, predicted_classes, predictions

# 3. Explainability Techniques
# ---------------------------

def get_gradcam(model, img_array, pred_index=None, layer_name=None):
    """
    Generate Grad-CAM for a specific image
    
    Args:
        model: A TensorFlow model
        img_array: Input image as numpy array (1, height, width, channels)
        pred_index: Index of the class to generate CAM for, if None uses the predicted class
        layer_name: Name of the layer to use for CAM, if None uses the last conv layer
        
    Returns:
        Original image and heatmap overlay
    """
    # Get the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                layer_name = layer.name
                break
    
    # Create a model that maps the input image to the activations
    # of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Then, we compute the gradient of the predicted class with respect to
    # the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # This is the gradient of the predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the predicted class
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # For visualization purpose, we normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize the heatmap to match the original image size
    img = img_array[0]
    img = (img * 255).astype(np.uint8)  # Convert to 0-255 range
    
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = np.array(Image.fromarray(heatmap_resized).resize(
        (img.shape[1], img.shape[0]),
        resample=Image.BICUBIC
    ))
    
    # Apply colormap to heatmap
    heatmap_colored = np.uint8(plt.cm.jet(heatmap_resized)[..., :3] * 255)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap_colored * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return img, superimposed_img, heatmap_resized

def load_and_preprocess_image(img_path, img_size=(224, 224)):
    """
    Load and preprocess an image for model prediction
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    return img_array

def lime_explanation(model, img_array, class_names, img_size=(224, 224), num_samples=1000):
    """
    Generate LIME explanation for the image
    
    Note: This is a wrapper around LIME for TensorFlow models
    """
    # Import LIME
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    
    # Create explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Define prediction function
    def predict_fn(images):
        # Reshape and preprocess images for model
        processed = []
        for img in images:
            img = np.expand_dims(img, axis=0)
            processed.append(img)
        
        # Stack all images into a batch
        batch = np.vstack(processed)
        
        # Get model predictions
        preds = model.predict(batch)
        return preds
    
    # Get explanation
    explanation = explainer.explain_instance(
        img_array[0].astype('double'), 
        predict_fn,
        top_labels=5, 
        hide_color=0, 
        num_samples=num_samples
    )
    
    # Get predicted class
    pred = model.predict(img_array)
    pred_class = np.argmax(pred[0])
    
    # Get explanation for predicted class
    temp, mask = explanation.get_image_and_mask(
        pred_class,
        positive_only=True,
        num_features=5,
        hide_rest=True
    )
    
    # Create visualization
    lime_img = mark_boundaries(temp, mask)
    
    # Get explanation with negative features
    temp_neg, mask_neg = explanation.get_image_and_mask(
        pred_class,
        positive_only=False,
        negative_only=True,
        num_features=5,
        hide_rest=True
    )
    
    lime_img_neg = mark_boundaries(temp_neg, mask_neg, color=(1, 0, 0))
    
    return lime_img, lime_img_neg, explanation, pred_class, pred[0][pred_class]

def shap_explanation(model, img_array, background_images):
    """
    Generate SHAP values for the image
    
    Args:
        model: TensorFlow model
        img_array: Image to explain (1, height, width, channels)
        background_images: Background images for SHAP explainer
        
    Returns:
        SHAP visualization and values
    """
    # Import SHAP
    import shap
    
    # Create a background dataset (subset of the training data)
    background = np.vstack(background_images)
    
    # Create explainer
    explainer = shap.DeepExplainer(model, background)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(img_array)
    
    # Get predicted class
    pred = model.predict(img_array)
    pred_class = np.argmax(pred[0])
    
    # Prepare visualization
    # SHAP returns a list of arrays, one per class
    shap_for_pred_class = shap_values[pred_class][0]
    
    # Compute absolute sum across channels for importance
    abs_shap_values = np.abs(shap_for_pred_class).sum(axis=2)
    
    # Normalize for visualization
    max_val = np.max(abs_shap_values)
    if max_val > 0:
        abs_shap_norm = abs_shap_values / max_val
    else:
        abs_shap_norm = abs_shap_values
    
    # Create heatmap
    heatmap = plt.cm.jet(abs_shap_norm)[:, :, :3]
    
    # Blend with original image
    original_img = img_array[0]
    shap_img = 0.7 * original_img + 0.3 * heatmap
    
    return shap_img, abs_shap_norm, shap_values, pred_class, pred[0][pred_class]

# 4. Experiments to Improve Explainability
# ---------------------------------------

def combined_explainability(model, img_path, class_names, background_images=None, img_size=(224, 224)):
    """
    Combine different explainability methods for a comprehensive view
    """
    # Load and preprocess the image
    img_array = load_and_preprocess_image(img_path, img_size)
    
    # Get model prediction
    pred = model.predict(img_array)
    pred_class = np.argmax(pred[0])
    confidence = pred[0][pred_class]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Original image
    orig_img = img_array[0]
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. GradCAM visualization
    _, gradcam_img, _ = get_gradcam(model, img_array)
    axes[0, 1].imshow(gradcam_img / 255.0)  # Normalize back to 0-1
    axes[0, 1].set_title('GradCAM Visualization')
    axes[0, 1].axis('off')
    
    # 3. Predicted class probabilities
    class_probs = pred[0]
    sorted_idx = np.argsort(class_probs)[::-1]
    top_classes = [class_names[i] for i in sorted_idx[:5]]
    top_probs = [class_probs[i] for i in sorted_idx[:5]]
    
    y_pos = np.arange(len(top_classes))
    axes[0, 2].barh(y_pos, top_probs)
    axes[0, 2].set_yticks(y_pos)
    axes[0, 2].set_yticklabels(top_classes)
    axes[0, 2].set_title('Top Class Probabilities')
    axes[0, 2].set_xlim(0, 1)
    
    # 4. LIME explanation (positive features)
    try:
        lime_img, lime_img_neg, _, _, _ = lime_explanation(model, img_array, class_names)
        axes[1, 0].imshow(lime_img)
        axes[1, 0].set_title('LIME (Positive Features)')
        axes[1, 0].axis('off')
        
        # 5. LIME explanation (negative features)
        axes[1, 1].imshow(lime_img_neg)
        axes[1, 1].set_title('LIME (Negative Features)')
        axes[1, 1].axis('off')
    except Exception as e:
        print(f"Error with LIME: {e}")
        axes[1, 0].text(0.5, 0.5, 'LIME error', ha='center', va='center')
        axes[1, 0].axis('off')
        axes[1, 1].text(0.5, 0.5, 'LIME error', ha='center', va='center')
        axes[1, 1].axis('off')
    
    # 6. SHAP explanation (if background images are provided)
    if background_images is not None:
        try:
            shap_img, _, _, _, _ = shap_explanation(model, img_array, background_images)
            axes[1, 2].imshow(shap_img)
            axes[1, 2].set_title('SHAP Explanation')
            axes[1, 2].axis('off')
        except Exception as e:
            print(f"Error with SHAP: {e}")
            axes[1, 2].text(0.5, 0.5, 'SHAP error', ha='center', va='center')
            axes[1, 2].axis('off')
    else:
        axes[1, 2].text(0.5, 0.5, 'Background images required for SHAP', ha='center', va='center')
        axes[1, 2].axis('off')
    
    # Add prediction information as title
    fig.suptitle(f"Prediction: {class_names[pred_class]} (Confidence: {confidence:.2f})", fontsize=16)
    
    return fig

def evaluate_explainability_methods(model, test_df, class_names, num_samples=5):
    """
    Evaluate different explainability methods on sample images
    """
    # Sample random images from test set
    sampled_indices = np.random.choice(len(test_df), num_samples, replace=False)
    
    # Create a set of background images for SHAP
    background_images = []
    for i in range(min(10, len(test_df))):
        img_path = test_df.iloc[i]['path']
        img_array = load_and_preprocess_image(img_path)
        background_images.append(img_array[0])
    
    # Generate explanations for each sampled image
    for idx, i in enumerate(sampled_indices):
        img_path = test_df.iloc[i]['path']
        true_class = test_df.iloc[i]['label']
        
        # Create combined visualization
        fig = combined_explainability(model, img_path, class_names, background_images)
        
        # Add true class information
        plt.figtext(0.5, 0.01, f"True Class: {class_names[true_class]}", ha='center', fontsize=12)
        
        # Save the figure
        plt.savefig(f'explanation_sample_{idx+1}.png', bbox_inches='tight')
        plt.close(fig)
    
    print(f"Generated explanations for {num_samples} sample images.")
    
    # Compare different variants of GradCAM
    img_path = test_df.iloc[sampled_indices[0]]['path']
    img_array = load_and_preprocess_image(img_path)
    
    # Compare GradCAM for different layers
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(img_array[0])
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Find conv layers to visualize
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            conv_layers.append(layer.name)
        elif hasattr(layer, 'layers'):  # For models with nested layers like ResNet
            for inner_layer in layer.layers:
                if isinstance(inner_layer, keras.layers.Conv2D):
                    conv_layers.append(inner_layer.name)
    
    # Select up to 5 layers evenly spaced throughout the network
    if len(conv_layers) > 5:
        indices = np.linspace(0, len(conv_layers)-1, 5, dtype=int)
        conv_layers = [conv_layers[i] for i in indices]
    
    # Generate GradCAM for each layer
    for i, layer_name in enumerate(conv_layers[:5]):  # Limit to 5 layers
        row, col = (i // 3) + 1, i % 3
        if row < 2 and col < 3:  # Ensure we don't exceed the grid
            try:
                _, gradcam_img, _ = get_gradcam(model, img_array, layer_name=layer_name)
                axes[row, col].imshow(gradcam_img / 255.0)
                axes[row, col].set_title(f'GradCAM: {layer_name.split("/")[-1]}')
                axes[row, col].axis('off')
            except Exception as e:
                print(f"Error with GradCAM for layer {layer_name}: {e}")
                axes[row, col].text(0.5, 0.5, f'Error: {layer_name}', ha='center', va='center')
                axes[row, col].axis('off')
    
    # Fill any unused subplots
    for i in range(len(conv_layers[:5]) + 1, 6):
        row, col = (i // 3) + 1, i % 3
        if row < 2 and col < 3:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_layer_comparison.png', bbox_inches='tight')
    plt.close()
    
    return sampled_indices

# 5. Model Export and Deployment
# ----------------------------

def export_model(model, class_names):
    """
    Export the model for deployment
    """
    # Save model in TensorFlow SavedModel format
    model.save('saved_model')
    
    # Save class names
    np.save('class_names.npy', class_names)
    
    # Create a TFLite version for mobile deployment
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Model exported in both SavedModel and TFLite formats.")
    
    # Save a model summary
    with open('model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print("Model summary saved.")
    
    # Create a simple inference script
    with open('inference.py', 'w') as f:
        f.write("""
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def load_model_and_classes():
    # Load the model
    model = tf.keras.models.load_model('saved_model')
    
    # Load class names
    class_names = np.load('class_names.npy', allow_pickle=True)
    
    return model, class_names

def preprocess_image(img_path, img_size=(224, 224)):
    # Load image
    img = Image.open(img_path).resize(img_size)
    
    # Convert to array and add batch dimension
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(model, img_array, class_names):
    # Get predictions
    predictions = model.predict(img_array)
    
    # Get top predicted class
    pred_class_idx = np.argmax(predictions[0])
    pred_class = class_names[pred_class_idx]
    confidence = predictions[0][pred_class_idx]
    
    # Get top 3 predictions
    top3_idx = np.argsort(predictions[0])[::-1][:3]
    top3_classes = [(class_names[i], predictions[0][i]) for i in top3_idx]
    
    return pred_class, confidence, top3_classes

def get_gradcam(model, img_array, layer_name=None):
    # Find the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                layer_name = layer.name
                break
    
    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Calculate gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight channels by gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to match image size
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = np.array(Image.fromarray(heatmap_resized).resize(
        (img_array.shape[2], img_array.shape[1]),
        resample=Image.BICUBIC
    ))
    
    # Create colored heatmap
    heatmap_colored = np.uint8(plt.cm.jet(heatmap_resized)[..., :3] * 255)
    
    # Superimpose heatmap on original image
    img = (img_array[0] * 255).astype(np.uint8)
    superimposed = heatmap_colored * 0.4 + img
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return superimposed

def inference(img_path):
    # Load model and classes
    model, class_names = load_model_and_classes()
    
    # Process image
    img_array = preprocess_image(img_path)
    
    # Get prediction
    pred_class, confidence, top3_classes = predict(model, img_array, class_names)
    
    # Generate GradCAM explanation
    gradcam_img = get_gradcam(model, img_array)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(img_array[0])
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # GradCAM
    ax2.imshow(gradcam_img / 255.0)
    ax2.set_title('GradCAM Explanation')
    ax2.axis('off')
    
    # Add prediction information
    plt.suptitle(f"Prediction: {pred_class} (Confidence: {confidence:.2f})", fontsize=16)
    
    # Print top 3 predictions
    print("Top 3 predictions:")
    for i, (cls, prob) in enumerate(top3_classes):
        print(f"{i+1}. {cls}: {prob:.4f}")
    
    plt.show()
    
    return {
        'class': pred_class,
        'confidence': float(confidence),
        'top3': [(cls, float(prob)) for cls, prob in top3_classes]
    }

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        result = inference(img_path)
        print(f"Predicted class: {result['class']} with {result['confidence']:.2%} confidence")
    else:
        print("Please provide an image path as argument")
""")
    
    print("Inference script created.")
    
    return

# Main function to run the pipeline
def main():
    """
    Main function to run the entire pipeline
    """
    # 1. Load and preprocess data
    print("1. Loading and preprocessing data...")
    train_df, val_df, test_df, class_names = load_data()
    
    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators(train_df, val_df, test_df)
    
    # 2. Build and train the model
    print("\n2. Building and training the model...")
    model = build_model(len(class_names))
    history, model = train_model(model, train_generator, val_generator, epochs=10)
    
    # 3. Evaluate the model
    print("\n3. Evaluating the model...")
    report, cm, true_classes, pred_classes, predictions = evaluate_model(model, test_generator, class_names)
    
    # 4. Apply explainability techniques
    print("\n4. Applying explainability techniques...")
    evaluate_explainability_methods(model, test_df, class_names, num_samples=5)
    
    # 5. Export the model for deployment
    print("\n5. Exporting the model for deployment...")
    export_model(model, class_names)
    
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()

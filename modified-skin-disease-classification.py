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
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Recall, Precision, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

# For model evaluation
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import itertools

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
DATA_DIR = '/Users/monilshah/Documents/02_NWU/11_MSDS_462_CV/99_Group_project/data/'
METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMAGES_PATH = os.path.join(DATA_DIR, 'HAM10000_images')

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

# Class mapping for better interpretability
classes = {
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    1: ('bcc', 'Basal cell carcinoma'),
    2: ('bkl', 'Benign keratosis-like lesions'),
    3: ('df', 'Dermatofibroma'),
    4: ('nv', 'Melanocytic nevi'),
    5: ('vasc', 'Pyogenic granulomas and hemorrhage'),
    6: ('mel', 'Melanoma')
}

# 1. Data Loading and Preprocessing
# --------------------------------

def load_and_explore_data():
    """
    Load and explore the HAM10000 dataset
    """
    # Read metadata
    df = pd.read_csv(METADATA_PATH)
    
    # Display dataset information
    print("Dataset Information:")
    print(f"Total number of images: {len(df)}")
    print(f"Number of classes: {df['dx'].nunique()}")
    print(f"Class distribution: \n{df['dx'].value_counts()}")
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='dx', data=df)
    plt.xlabel('Disease', size=12)
    plt.ylabel('Frequency', size=12)
    plt.title('Frequency Distribution of Classes', size=16)
    plt.xticks(rotation=45)
    plt.savefig('class_distribution_original.png')
    plt.close()
    
    # Visualize gender distribution
    plt.figure(figsize=(8, 8))
    plt.pie(df['sex'].value_counts(), labels=df['sex'].value_counts().index, autopct="%.1f%%")
    plt.title('Gender of Patient', size=16)
    plt.savefig('gender_distribution.png')
    plt.close()
    
    # Histogram of age
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'])
    plt.title('Histogram of Age of Patients', size=16)
    plt.savefig('age_histogram.png')
    plt.close()
    
    # Location of disease over gender
    value = df[['localization', 'sex']].value_counts().to_frame()
    value.reset_index(level=[1, 0], inplace=True)
    temp = value.rename(columns={'localization': 'location', 0: 'count'})
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='location', y='count', hue='sex', data=temp)
    plt.title('Location of Disease over Gender', size=16)
    plt.xlabel('Location', size=12)
    plt.ylabel('Frequency/Count', size=12)
    plt.xticks(rotation=90)
    plt.savefig('location_gender.png')
    plt.close()
    
    return df

def preprocess_data(df, image_size=(32, 32), use_csv=False):
    """
    Preprocess the dataset for training
    """
    if use_csv:
        # If using CSV format (similar to HAM10000 ViT implementation)
        csv_path = os.path.join(DATA_DIR, 'hmnist_28_28_RGB.csv')
        print(f"Checking for CSV data at: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"CSV file not found at: {csv_path}")
            print("Falling back to image-based loading...")
            use_csv = False
        else:
            try:
                data = pd.read_csv(csv_path)
                print("CSV data loaded successfully.")
                
                # Extract labels and features
                y = data['label']
                x = data.drop(columns=['label'])
                
                # Reshape features to image format
                x = np.array(x).reshape(-1, 28, 28, 3)
                
                # Resize images to target size
                x = tf.image.resize(x, image_size).numpy()
                
                # Normalize the images
                x = (x - np.mean(x)) / np.std(x)
                
                # Apply oversampling to balance classes
                oversample = RandomOverSampler()
                x_flat = x.reshape(x.shape[0], -1)  # Flatten for oversampling
                x_flat, y = oversample.fit_resample(x_flat, y)
                x = x_flat.reshape(-1, *image_size, 3)  # Reshape back to images
                
                # Convert labels to one-hot encoding
                y = to_categorical(y)
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                
                print(f"Training set: {X_train.shape}, {y_train.shape}")
                print(f"Validation set: {X_val.shape}, {y_val.shape}")
                print(f"Test set: {X_test.shape}, {y_test.shape}")
                
                return X_train, X_val, X_test, y_train, y_val, y_test
                
            except Exception as e:
                print(f"Error loading CSV data: {e}")
                print("Falling back to image-based loading...")
                use_csv = False
    
    if not use_csv:
        # Process using image files and dataframe
        # Encode class labels
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['dx'])
        class_mapping = dict(zip(range(len(le.classes_)), le.classes_))
        print("Class mapping:", class_mapping)
        
        # Check folder structure to locate images
        print("\nChecking directory structure...")
        try:
            # First check if images are in expected directory
            img_dirs = [IMAGES_PATH]
            
            # Check for other possible locations
            data_subfolders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
            print(f"Found directories in data folder: {data_subfolders}")
            
            for folder in data_subfolders:
                possible_img_dir = os.path.join(DATA_DIR, folder)
                img_dirs.append(possible_img_dir)
            
            # Look for any folder with images
            for directory in img_dirs:
                file_count = len([f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])
                print(f"Directory {directory} contains {file_count} image files")
                
                # If this directory has images, use it as our image directory
                if file_count > 0:
                    IMAGES_PATH_ACTUAL = directory
                    print(f"Using {IMAGES_PATH_ACTUAL} as the image directory")
                    
                    # Check actual image extensions
                    extensions = set()
                    for f in os.listdir(directory)[:100]:  # Check first 100 files
                        if '.' in f:
                            ext = f.split('.')[-1].lower()
                            if ext in ['jpg', 'jpeg', 'png']:
                                extensions.add(ext)
                    
                    if extensions:
                        print(f"Found image extensions: {extensions}")
                        primary_ext = list(extensions)[0]  # Use first found extension
                    else:
                        print("No common image extensions found, defaulting to .jpg")
                        primary_ext = 'jpg'
                    break
            else:
                # If no suitable directory found
                print("WARNING: No directory with images found. Using default path.")
                IMAGES_PATH_ACTUAL = IMAGES_PATH
                primary_ext = 'jpg'
                
        except Exception as e:
            print(f"Error checking directories: {e}")
            IMAGES_PATH_ACTUAL = IMAGES_PATH
            primary_ext = 'jpg'
        
        # Add path column to dataframe
        def create_path(img_id):
            return os.path.join(IMAGES_PATH_ACTUAL, f"{img_id}.{primary_ext}")
        
        # Apply oversampling to balance classes
        print("Applying oversampling to balance classes...")
        features = df[['image_id', 'dx', 'label']].copy()
        X = features[['image_id', 'dx']].values
        y = features['label'].values
        
        oversample = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversample.fit_resample(X, y)
        
        # Create a new balanced dataframe
        balanced_df = pd.DataFrame({
            'image_id': [x[0] for x in X_resampled],
            'dx': [x[1] for x in X_resampled],
            'label': y_resampled
        })
        
        # Add path column to balanced dataframe
        balanced_df['path'] = balanced_df['image_id'].apply(create_path)
        
        # Visualize class distribution after oversampling
        plt.figure(figsize=(10, 6))
        sns.countplot(x='dx', data=balanced_df)
        plt.xlabel('Disease', size=12)
        plt.ylabel('Frequency', size=12)
        plt.title('Frequency Distribution of Classes After Oversampling', size=16)
        plt.xticks(rotation=45)
        plt.savefig('class_distribution_oversampled.png')
        plt.close()
        
        # Split into train, validation, and test sets
        train_df, temp_df = train_test_split(balanced_df, test_size=0.3, random_state=42, stratify=balanced_df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
        
        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")
        
        # Create data generators
        train_generator, val_generator, test_generator = create_data_generators(train_df, val_df, test_df, img_size=image_size)
        
        return train_generator, val_generator, test_generator, train_df, val_df, test_df

def create_data_generators(train_df, val_df, test_df, batch_size=32, img_size=(32, 32)):
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
        class_mode='raw',  # Changed to categorical for one-hot encoding
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='raw',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='path',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='raw',
        shuffle=False
    )
    
    print(f"Number of training batches: {len(train_generator)}")
    print(f"Number of validation batches: {len(val_generator)}")
    print(f"Number of test batches: {len(test_generator)}")
    
    return train_generator, val_generator, test_generator

# 2. Vision Transformer Model
# --------------------------

def mlp(x, hidden_units, dropout_rate):
    """
    Implements the MLP block of the Vision Transformer
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_vit_classifier(
    input_shape=(32, 32, 3),
    patch_size=4,
    num_patches=64,
    projection_dim=64,
    num_heads=4,
    transformer_layers=8,
    mlp_head_units=[2048, 1024],
    num_classes=7,
):
    """
    Create a Vision Transformer (ViT) model for classification
    
    Args:
        input_shape: Shape of input images
        patch_size: Size of patches to be extracted from the images
        num_patches: Number of patches (input_shape[0]/patch_size * input_shape[1]/patch_size)
        projection_dim: Projection dimension for the patches
        num_heads: Number of attention heads
        transformer_layers: Number of transformer layers
        mlp_head_units: Hidden units for the final MLP head
        num_classes: Number of output classes
        
    Returns:
        ViT model
    """
    inputs = layers.Input(shape=input_shape)

    # Patch embedding
    patches = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size, padding="VALID")(inputs)
    patches = layers.Reshape((num_patches, projection_dim))(patches)

    # Position embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    encoded_patches = patches + position_embedding

    # Transformer blocks
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    # MLP head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    logits = layers.Dense(num_classes)(features)
    outputs = layers.Activation("softmax")(logits)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_model(model, train_data, val_data, epochs=30, batch_size=128, use_generators=True):
    """
    Train the model with early stopping and model checkpointing
    
    Args:
        model: The model to train
        train_data: Training data (generator or tuple of arrays)
        val_data: Validation data (generator or tuple of arrays)
        epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        use_generators: Whether the data is provided as generators
        
    Returns:
        Training history
    """
    # Callbacks
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True
    )
    
    callback_early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        mode='max',
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )
    
    callbacks = [callback_checkpoint, callback_early_stopping, reduce_lr]
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy', Recall(), Precision(), AUC(),
                TruePositives(), TrueNegatives(), FalseNegatives(), FalsePositives()]
    )
    
    # Train the model
    if use_generators:
        # If data is provided as generators
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # If data is provided as arrays
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    # Save the final model
    model.save('final_model.keras')
    
    # Save training history
    pd.DataFrame.from_dict(history.history).to_csv('training_history.csv', index=False)
    
    # Plot training curves
    plot_training_curves(history)
    
    return history

def plot_training_curves(history):
    """
    Plot accuracy and loss curves from training history
    """
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

# 3. Model Evaluation
# -----------------

def evaluate_model(model, test_data, class_names, use_generator=True):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained model
        test_data: Test data (generator or tuple of arrays)
        class_names: Names of the classes
        use_generator: Whether test_data is a generator
        
    Returns:
        Evaluation results
    """
    print("Evaluating model on test data...")
    
    if use_generator:
        # If test_data is a generator
        test_generator = test_data
        
        # Get the steps needed to go through all test data
        steps = len(test_generator)
        
        # Get predictions
        y_pred_prob = model.predict(test_generator, steps=steps, verbose=1)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Get true labels (this is tricky with generators)
        # Reset the generator to ensure we get all samples in original order
        test_generator.reset()
        y_true = np.zeros(len(test_generator.labels))
        
        # For categorical class mode, convert one-hot back to class indices
        batch_index = 0
        for i in range(steps):
            _, y_batch = next(test_generator)
            batch_size = y_batch.shape[0]
            # For categorical class mode, convert one-hot back to class indices
            if y_batch.shape[1] > 1:  # If one-hot encoded
                y_batch = np.argmax(y_batch, axis=1)
            y_true[batch_index:batch_index+batch_size] = y_batch
            batch_index += batch_size
    else:
        # If test_data is a tuple of arrays
        X_test, y_test = test_data
        
        # Get predictions
        y_pred_prob = model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # If y_test is one-hot encoded, convert to class indices
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
    
    # Calculate metrics
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=[name for idx, (name, _) in classes.items()])
    print(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names=[name for idx, (name, _) in classes.items()])
    
    # Calculate precision, recall, f1-score
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, class_names,
                        normalize=False,
                        title='Confusion Matrix',
                        cmap=plt.cm.Blues):
    """
    Plot confusion matrix with nice formatting
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.savefig('confusion_matrix.png')
    plt.close()

# 4. Explainability Techniques
# ---------------------------

def get_gradcam(model, img_array, pred_index=None):
    """
    Generate Grad-CAM for a specific image
    
    Args:
        model: TensorFlow model
        img_array: Input image as numpy array (1, height, width, channels)
        pred_index: Index of the class to generate CAM for
        
    Returns:
        Original image and heatmap overlay
    """
    # Get the last convolutional layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    # Create a model that maps the input image to the activations
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )
    
    # Compute the gradient of the predicted class with respect to
    # the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Gradient of the output with respect to the conv layer output
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps by importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to match original image size
    img = img_array[0]
    img = (img * 255).astype(np.uint8)
    
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = np.array(Image.fromarray(heatmap_resized).resize(
        (img.shape[1], img.shape[0]),
        resample=Image.BICUBIC
    ))
    
    # Apply colormap
    heatmap_colored = np.uint8(plt.cm.jet(heatmap_resized)[..., :3] * 255)
    
    # Superimpose heatmap on original image
    superimposed_img = heatmap_colored * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return img, superimposed_img, heatmap_resized

def load_and_preprocess_image(img_path, img_size=(32, 32)):
    """
    Load and preprocess a single image for model prediction
    """
    img = Image.open(img_path).resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def visualize_model_predictions(model, test_df, class_names, num_samples=5):
    """
    Visualize model predictions and GradCAM for sample images
    """
    # Sample random images from test set
    if len(test_df) > num_samples:
        sampled_indices = np.random.choice(len(test_df), num_samples, replace=False)
    else:
        sampled_indices = range(len(test_df))
    
    # Create a figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i, idx in enumerate(sampled_indices):
        img_path = test_df.iloc[idx]['path']
        true_label = test_df.iloc[idx]['label']
        true_class = class_names[true_label][0]
        
        # Load and preprocess image
        img_array = load_and_preprocess_image(img_path)
        
        # Get prediction
        prediction = model.predict(img_array)
        pred_label = np.argmax(prediction[0])
        pred_class = class_names[pred_label][0]
        confidence = prediction[0][pred_label]
        
        # Get GradCAM visualization
        orig_img, gradcam_img, _ = get_gradcam(model, img_array, pred_label)
        
        # Display original image
        axes[i, 0].imshow(orig_img / 255.0)
        axes[i, 0].set_title(f'True: {true_class}')
        axes[i, 0].axis('off')
        
        # Display GradCAM visualization
        axes[i, 1].imshow(gradcam_img / 255.0)
        axes[i, 1].set_title(f'Pred: {pred_class} ({confidence:.2f})')
        axes[i, 1].axis('off')
        
        # Display class probabilities
        top_k = 3
        top_indices = np.argsort(prediction[0])[::-1][:top_k]
        top_classes = [class_names[idx][0] for idx in top_indices]
        top_probs = [prediction[0][idx] for idx in top_indices]
        
        bars = axes[i, 2].barh(range(top_k), top_probs)
        axes[i, 2].set_yticks(range(top_k))
        axes[i, 2].set_yticklabels(top_classes)
        axes[i, 2].set_xlim(0, 1)
        axes[i, 2].set_title('Top Predictions')
        
        # Color the bars based on correctness
        for j, bar in enumerate(bars):
            if top_indices[j] == true_label:
                bar.set_color('green')
            else:
                bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    plt.close()
    
    return sampled_indices

# 5. Export ViT model to TFLite
# ---------------------------

def export_to_tflite(model, quantize=True):
    """
    Export the model to TFLite format with optional quantization
    
    Args:
        model: Trained model
        quantize: Whether to apply quantization
        
    Returns:
        Saved TFLite model path
    """
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Apply post-training quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert model
    tflite_model = converter.convert()
    
    # Save the model
    tflite_filename = "model_quantized.tflite" if quantize else "model.tflite"
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
    
    # Get file size
    model_size = len(tflite_model) / 1024  # Size in KB
    print(f"{tflite_filename} size = {model_size:.2f} KB")
    
    return tflite_filename

def evaluate_tflite_model(tflite_model_path, test_data, class_names, num_samples=None):
    """
    Evaluate the TFLite model on test data
    
    Args:
        tflite_model_path: Path to the TFLite model
        test_data: Test data as (X_test, y_test) or a generator
        class_names: Names of the classes
        num_samples: Number of samples to evaluate (None for all)
        
    Returns:
        Evaluation metrics
    """
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Extract test data
    if isinstance(test_data, tuple):
        # If test_data is a tuple of arrays
        X_test, y_test = test_data
        
        # If y_test is one-hot encoded, convert to class indices
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
        
        # Limit number of samples if specified
        if num_samples is not None and num_samples < len(X_test):
            indices = np.random.choice(len(X_test), num_samples, replace=False)
            X_test = X_test[indices]
            y_true = y_true[indices]
        
        # Prepare predictions array
        y_pred = np.zeros(len(X_test), dtype=np.int32)
        y_pred_prob = np.zeros((len(X_test), len(class_names)), dtype=np.float32)
        
        # Perform inference
        print(f"Running TFLite inference on {len(X_test)} samples...")
        for i, sample in enumerate(X_test):
            # Ensure sample has batch dimension and correct dtype
            input_data = np.expand_dims(sample, axis=0).astype(np.float32)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Store prediction
            y_pred[i] = np.argmax(output_data[0])
            y_pred_prob[i] = output_data[0]
            
            # Print progress
            if i % 100 == 0:
                print(f"Processed {i}/{len(X_test)} samples...")
    
    else:
        # If test_data is a generator (more complex)
        test_generator = test_data
        
        # Determine number of samples to evaluate
        total_samples = len(test_generator.labels)
        if num_samples is not None and num_samples < total_samples:
            total_samples = num_samples
        
        # Prepare arrays for results
        y_pred = np.zeros(total_samples, dtype=np.int32)
        y_pred_prob = np.zeros((total_samples, len(class_names)), dtype=np.float32)
        y_true = np.zeros(total_samples, dtype=np.int32)
        
        # Reset the generator
        test_generator.reset()
        
        # Run inference
        print(f"Running TFLite inference on {total_samples} samples...")
        for i in range(total_samples):
            # Get a single image and label
            x, y = next(test_generator)
            
            # If generator returns batches, just take the first sample
            if len(x.shape) > 3:
                x = x[0]
                y = y[0]
            
            # Ensure image has batch dimension
            input_data = np.expand_dims(x, axis=0).astype(np.float32)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Store prediction
            y_pred[i] = np.argmax(output_data[0])
            y_pred_prob[i] = output_data[0]
            
            # If y is one-hot encoded, convert to class index
            if len(y.shape) > 0 and y.shape[0] > 1:
                y_true[i] = np.argmax(y)
            else:
                y_true[i] = y
            
            # Print progress
            if i % 100 == 0:
                print(f"Processed {i}/{total_samples} samples...")
    
    # Calculate metrics
    print("\nTFLite Model Classification Report:")
    report = classification_report(y_true, y_pred, target_names=[name for idx, (name, _) in classes.items()])
    print(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names=[name for idx, (name, _) in classes.items()], 
                        title='TFLite Model Confusion Matrix')
    
    # Calculate precision, recall, f1-score
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"TFLite Precision (macro): {precision:.4f}")
    print(f"TFLite Recall (macro): {recall:.4f}")
    print(f"TFLite F1 Score (macro): {f1:.4f}")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Main function to run the entire pipeline
def main():
    """
    Main function to run the complete skin disease classification pipeline using Vision Transformer
    """
    print("Starting Skin Disease Classification with Vision Transformer")
    
    # 1. Load and explore data
    print("\n1. Loading and exploring data...")
    df = load_and_explore_data()
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    # Check if CSV file exists, but default to image-based processing
    # since we know the CSV file is likely not available
    csv_path = os.path.join(DATA_DIR, 'hmnist_28_28_RGB.csv')
    if os.path.exists(csv_path):
        print(f"Found CSV data file at: {csv_path}")
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, image_size=(32, 32), use_csv=True)
        use_generators = False
    else:
        print("CSV file not found. Using image-based preprocessing...")
        train_generator, val_generator, test_generator, train_df, val_df, test_df = preprocess_data(df, image_size=(64, 64), use_csv=False)
        use_generators = True
    
    # 3. Build ViT model
    print("\n3. Building Vision Transformer model...")
    input_shape = (64, 64, 3)
    num_classes = 7
    
    # Calculate num_patches based on input size and patch size
    patch_size = 4
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    model = create_vit_classifier(
        input_shape=input_shape,
        patch_size=patch_size,
        num_patches=num_patches,
        projection_dim=64,
        num_heads=4,
        transformer_layers=8,
        mlp_head_units=[2048, 1024],
        num_classes=num_classes
    )
    
    # Print model summary
    model.summary()
    
    # 4. Train the model
    print("\n4. Training the model...")
    if use_generators:
        history = train_model(model, train_generator, val_generator, epochs=30, use_generators=True)
    else:
        history = train_model(model, (X_train, y_train), (X_val, y_val), epochs=30, batch_size=128, use_generators=False)
    
    # 5. Evaluate the model
    print("\n5. Evaluating the model...")
    if use_generators:
        class_names = [name for idx, (name, _) in classes.items()]
        results = evaluate_model(model, test_generator, class_names, use_generator=True)
        
        # Visualize model predictions
        print("\nVisualizing model predictions...")
        visualize_model_predictions(model, test_df, classes, num_samples=5)
    else:
        class_names = [name for idx, (name, _) in classes.items()]
        results = evaluate_model(model, (X_test, y_test), class_names, use_generator=False)
    
    # 6. Export model to TFLite
    print("\n6. Exporting model to TFLite with quantization...")
    tflite_model_path = export_to_tflite(model, quantize=True)
    
    # 7. Evaluate TFLite model
    print("\n7. Evaluating TFLite model...")
    if use_generators:
        tflite_results = evaluate_tflite_model(tflite_model_path, test_generator, class_names, num_samples=100)
    else:
        tflite_results = evaluate_tflite_model(tflite_model_path, (X_test, y_test), class_names, num_samples=100)
    
    print("\nSkin Disease Classification with Vision Transformer completed successfully!")

if __name__ == "__main__":
    # Execute the main function
    main()
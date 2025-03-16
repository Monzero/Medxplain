"""
Integration Guide for Advanced Preprocessing in Skin Disease Classification

This guide shows how to integrate the advanced preprocessing module with the main skin disease
classification pipeline to improve performance.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# Import the preprocessing module
from advanced_preprocessing import SkinLesionPreprocessor
from unet_segmentation_model import UNetSegmentationTrainer


def integrate_preprocessing(train_df, val_df, test_df, class_names, batch_size=32, 
                           img_size=(224, 224), preprocessing_options=None):
    """
    Integrate advanced preprocessing with the data generators.
    
    Args:
        train_df, val_df, test_df: DataFrames with image paths and labels
        class_names: List of class names
        batch_size: Batch size for training
        img_size: Target image size
        preprocessing_options: Dictionary of preprocessing options
        
    Returns:
        Preprocessed data generators and preprocessor
    """
    # Set default preprocessing options if not provided
    if preprocessing_options is None:
        preprocessing_options = {
            'denoising': True,           # Apply wavelet denoising
            'enhance_contrast': True,    # Apply CLAHE for contrast enhancement
            'segmentation': False,       # Apply lesion segmentation
            'background_removal': False, # Apply background removal
            'mixup': False,              # Apply mixup augmentation during training
            'cutmix': False,             # Apply cutmix augmentation during training
            'elastic': False,            # Apply elastic transforms
            'augmentation_strength': 'moderate' # 'light', 'moderate', or 'strong'
        }
    
    # Initialize the preprocessor
    preprocessor = SkinLesionPreprocessor(img_size=img_size)
    
    # Load segmentation model if segmentation is enabled
    if preprocessing_options.get('segmentation', False):
        if os.path.exists('models/unet_segmentation.h5'):
            unet_trainer = UNetSegmentationTrainer(img_size=img_size)
            unet_trainer.load_model('models/unet_segmentation.h5')
            preprocessor.segmentation_model = unet_trainer.model
            print("Segmentation model loaded successfully")
        else:
            print("Warning: Segmentation model not found. Segmentation will be disabled.")
            preprocessing_options['segmentation'] = False
    
    # Create a custom image preprocessor function for ImageDataGenerator
    def preprocess_image(img):
        """Preprocess a single image with the advanced pipeline"""
        processed_img = preprocessor.preprocess_pipeline(
            img,
            denoising=preprocessing_options.get('denoising', True),
            enhance_contrast=preprocessing_options.get('enhance_contrast', True),
            segment=preprocessing_options.get('segmentation', False)
        )
        
        # Apply background removal if enabled
        if preprocessing_options.get('background_removal', False):
            processed_img = preprocessor.remove_background(processed_img)
            
        return processed_img
    
    # Define augmentation strength options
    augmentation_config = {
        'light': {
            'rotation_range': 10,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': True,
            'brightness_range': [0.9, 1.1]
        },
        'moderate': {
            'rotation_range': 20,
            'width_shift_range': 0.15,
            'height_shift_range': 0.15,
            'shear_range': 0.15,
            'zoom_range': 0.15,
            'horizontal_flip': True,
            'vertical_flip': True,
            'brightness_range': [0.85, 1.15]
        },
        'strong': {
            'rotation_range': 30,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'vertical_flip': True,
            'brightness_range': [0.8, 1.2]
        }
    }
    
    # Get augmentation settings based on strength
    strength = preprocessing_options.get('augmentation_strength', 'moderate')
    aug_settings = augmentation_config[strength]
    
    # Create data generators with preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_image,
        **aug_settings
    )
    
    # No augmentation for validation and test, just preprocessing
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_image
    )
    
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_image
    )
    
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
    
    # If mixup or cutmix is enabled, create a custom training generator
    if preprocessing_options.get('mixup', False) or preprocessing_options.get('cutmix', False):
        train_generator = MixupCutmixGenerator(
            train_generator,
            preprocessor,
            len(class_names),
            use_mixup=preprocessing_options.get('mixup', False),
            use_cutmix=preprocessing_options.get('cutmix', False),
            alpha=0.2  # Mixup/cutmix strength parameter
        )
    
    return train_generator, val_generator, test_generator, preprocessor


class MixupCutmixGenerator:
    """
    Generator wrapper for mixup and cutmix augmentation.
    """
    def __init__(self, generator, preprocessor, num_classes, 
                 use_mixup=True, use_cutmix=True, alpha=0.2, prob=0.5):
        """
        Initialize the generator.
        
        Args:
            generator: Base generator
            preprocessor: SkinLesionPreprocessor instance
            num_classes: Number of classes
            use_mixup: Whether to use mixup
            use_cutmix: Whether to use cutmix
            alpha: Mixup/cutmix strength parameter
            prob: Probability of applying mixup/cutmix to a batch
        """
        self.generator = generator
        self.preprocessor = preprocessor
        self.num_classes = num_classes
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.alpha = alpha
        self.prob = prob
        self.batch_size = generator.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Get next batch with mixup/cutmix applied"""
        # Get a batch from the base generator
        X, y = next(self.generator)
        
        # Convert sparse labels to one-hot encoding
        y_onehot = tf.one_hot(y, self.num_classes).numpy()
        
        # Randomly decide whether to apply mixup or cutmix
        if np.random.random() < self.prob:
            if self.use_mixup and (not self.use_cutmix or np.random.random() < 0.5):
                # Apply mixup
                X, y_onehot = self.preprocessor.apply_mixup_batch(X, y_onehot, alpha=self.alpha)
            elif self.use_cutmix:
                # Apply cutmix
                X, y_onehot = self.preprocessor.apply_cutmix_batch(X, y_onehot, alpha=self.alpha)
        
        return X, y_onehot
    
    def __len__(self):
        """Return the number of batches in an epoch"""
        return len(self.generator)


def train_with_advanced_preprocessing(data_dir, output_dir='models', img_size=(224, 224), 
                                     preprocessing_options=None, train_options=None):
    """
    Train a skin disease classification model with advanced preprocessing.
    
    Args:
        data_dir: Directory containing HAM10000 dataset
        output_dir: Directory to save model and results
        img_size: Target image size
        preprocessing_options: Preprocessing options dictionary
        train_options: Training options dictionary
        
    Returns:
        Trained model and history
    """
    # Set default training options if not provided
    if train_options is None:
        train_options = {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.0001,
            'model_type': 'resnet50',  # 'resnet50', 'efficientnet', etc.
            'validation_split': 0.2,
            'test_split': 0.1
        }
    
    # Load and preprocess data
    print("Loading data...")
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)
    
    # Add path column
    images_dir = find_images_dir(data_dir)
    df['path'] = df['image_id'].apply(lambda x: os.path.join(images_dir, x + '.jpg'))
    
    # Encode class labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['dx'])
    class_names = le.classes_
    
    # Split data
    train_df, temp_df = train_test_split(
        df, test_size=train_options['validation_split'] + train_options['test_split'], 
        stratify=df['label'], random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=train_options['test_split'] / (train_options['validation_split'] + train_options['test_split']),
        stratify=temp_df['label'], random_state=42
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Create generators with advanced preprocessing
    train_generator, val_generator, test_generator, preprocessor = integrate_preprocessing(
        train_df, val_df, test_df, class_names,
        batch_size=train_options['batch_size'],
        img_size=img_size,
        preprocessing_options=preprocessing_options
    )
    
    # Build model
    model = build_model(
        num_classes=len(class_names),
        img_size=img_size,
        model_type=train_options['model_type'],
        learning_rate=train_options['learning_rate']
    )
    
    # Create callbacks
    callbacks = create_callbacks(output_dir)
    
    # Train model
    print("Training model with advanced preprocessing...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=train_options['epochs'],
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate(test_generator, steps=len(test_generator))
    print("Test Loss:", results[0])
    print("Test Accuracy:", results[1])
    
    # Save results and evaluation metrics
    save_results(model, history, results, output_dir, class_names)
    
    # Visualize sample preprocessed images
    visualize_preprocessing_samples(preprocessor, test_df, output_dir)
    
    return model, history


def find_images_dir(data_dir):
    """Find the directory containing HAM10000 images"""
    # Check common directory names
    possible_dirs = [
        os.path.join(data_dir, 'images'),
        os.path.join(data_dir, 'HAM10000_images'),
        os.path.join(data_dir, 'HAM10000_images_part1'),
        os.path.join(data_dir, 'ISIC2018_Task3_Training_Input')
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path) and len(os.listdir(dir_path)) > 0:
            print(f"Found images in {dir_path}")
            return dir_path
    
    # If no specific directory found, return the data_dir itself
    print(f"No specific image directory found, using {data_dir}")
    return data_dir


def build_model(num_classes, img_size=(224, 224), model_type='resnet50', learning_rate=0.0001):
    """
    Build a classification model using transfer learning.
    
    Args:
        num_classes: Number of classes
        img_size: Input image size
        model_type: Model architecture ('resnet50', 'efficientnet', etc.)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    # Define input shape
    input_shape = (*img_size, 3)
    
    # Select base model
    if model_type == 'resnet50':
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
    elif model_type == 'efficientnet':
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Freeze early layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Create model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_callbacks(output_dir):
    """Create training callbacks"""
    os.makedirs(output_dir, exist_ok=True)
    
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs')
        )
    ]
    
    return callbacks


def save_results(model, history, test_results, output_dir, class_names):
    """Save model results and visualizations"""
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model.save(os.path.join(output_dir, 'final_model.h5'))
    
    # Save history plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_results[0]}\n")
        f.write(f"Test Accuracy: {test_results[1]}\n")
    
    # Save class names
    np.save(os.path.join(output_dir, 'class_names.npy'), class_names)


def visualize_preprocessing_samples(preprocessor, test_df, output_dir, num_samples=5):
    """Visualize preprocessing steps on sample images"""
    # Create directory
    preprocessed_dir = os.path.join(output_dir, 'preprocessed_samples')
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Sample images
    sample_indices = np.random.choice(len(test_df), min(num_samples, len(test_df)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Load image
        img_path = test_df.iloc[idx]['path']
        try:
            import cv2
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0  # Normalize
            
            # Visualize preprocessing
            preprocessor.visualize_preprocessing(
                img, 
                save_path=os.path.join(preprocessed_dir, f'sample_{i+1}_preprocessing.png')
            )
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")


# Main execution
if __name__ == "__main__":
    # Example preprocessing options
    preprocessing_options = {
        'denoising': True,
        'enhance_contrast': True,
        'segmentation': True,
        'background_removal': True,
        'mixup': True,
        'cutmix': True,
        'elastic': True,
        'augmentation_strength': 'moderate'
    }
    
    # Example training options
    train_options = {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.0001,
        'model_type': 'resnet50',
        'validation_split': 0.2,
        'test_split': 0.1
    }
    
    # Train model with advanced preprocessing
    data_dir = 'data/HAM10000'
    output_dir = 'models/advanced_preprocessing'
    
    # To train the model (uncomment to run):
    # model, history = train_with_advanced_preprocessing(
    #     data_dir, 
    #     output_dir=output_dir,
    #     preprocessing_options=preprocessing_options,
    #     train_options=train_options
    # )
    
    print("To train the model, uncomment the train_with_advanced_preprocessing function call.")

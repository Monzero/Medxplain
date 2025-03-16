import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

class UNetSegmentationTrainer:
    """
    Class for training a U-Net segmentation model for skin lesions.
    Includes utilities for preparing data, training, evaluating, and using the model.
    """
    
    def __init__(self, img_size=(224, 224)):
        """
        Initialize the trainer.
        
        Args:
            img_size: Target image size (height, width)
        """
        self.img_size = img_size
        self.model = None
    
    def build_unet_model(self, input_size=(224, 224, 3)):
        """
        Build a U-Net segmentation model.
        
        Args:
            input_size: Input image shape (height, width, channels)
            
        Returns:
            U-Net model
        """
        inputs = Input(input_size)
        
        # Encoder (downsampling path)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        
        # Bridge
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        
        # Decoder (upsampling path)
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        
        # Output layer
        outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])])
        
        self.model = model
        return model
    
    def prepare_dataset(self, image_dir, mask_dir, split_ratio=0.2):
        """
        Prepare dataset from directories containing images and masks.
        
        Args:
            image_dir: Directory containing original images
            mask_dir: Directory containing segmentation masks
            split_ratio: Validation split ratio
            
        Returns:
            Lists of training and validation data
        """
        # Get all filenames
        image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # Ensure matching files
        assert len(image_filenames) == len(mask_filenames), "Number of images and masks don't match"
        
        print(f"Found {len(image_filenames)} image-mask pairs")
        
        # Load and preprocess images and masks
        images = []
        masks = []
        
        for img_file, mask_file in tqdm(zip(image_filenames, mask_filenames), total=len(image_filenames)):
            # Load image
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            img = img / 255.0  # Normalize
            
            # Load mask
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]))
            mask = (mask > 127).astype(np.float32)  # Binarize
            mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
            
            images.append(img)
            masks.append(mask)
        
        # Convert to numpy arrays
        images = np.array(images)
        masks = np.array(masks)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, masks, test_size=split_ratio, random_state=42
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        
        return X_train, X_val, y_train, y_val
    
    def prepare_from_isic(self, isic_csv, images_dir):
        """
        Prepare dataset from ISIC challenge format.
        
        Args:
            isic_csv: Path to ISIC metadata CSV file
            images_dir: Directory containing images
            
        Returns:
            Lists of training and validation data
        """
        # Load metadata
        df = pd.read_csv(isic_csv)
        
        # Find mask files
        images = []
        masks = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_id = row['image_id']
            
            # Load image
            img_path = os.path.join(images_dir, f"{image_id}.jpg")
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found, skipping")
                continue
                
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            img = img / 255.0  # Normalize
            
            # Load or generate mask (depending on dataset structure)
            # For ISIC 2018 challenge format:
            mask_path = os.path.join(images_dir, f"{image_id}_segmentation.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                # Alternative approach if segmentation is in a different directory
                mask_path = os.path.join(images_dir, "../masks", f"{image_id}_mask.png")
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask {mask_path} not found, skipping")
                    continue
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]))
            mask = (mask > 127).astype(np.float32)  # Binarize
            mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
            
            images.append(img)
            masks.append(mask)
        
        # Convert to numpy arrays
        images = np.array(images)
        masks = np.array(masks)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, masks, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        
        return X_train, X_val, y_train, y_val
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=16):
        """
        Create data generators with augmentation.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            batch_size: Batch size for training
            
        Returns:
            Training and validation generators
        """
        # Data augmentation for training
        data_gen_args = dict(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Create image and mask generators
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)  # Same transformations for masks
        
        # Create generators with the same random seed for synchronized transformations
        seed = 42
        image_datagen.fit(X_train, seed=seed)
        mask_datagen.fit(y_train, seed=seed)
        
        image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
        mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)
        
        # Combine generators
        train_generator = zip(image_generator, mask_generator)
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        val_image_generator = val_datagen.flow(X_val, batch_size=batch_size, seed=seed)
        val_mask_generator = val_datagen.flow(y_val, batch_size=batch_size, seed=seed)
        
        val_generator = zip(val_image_generator, val_mask_generator)
        
        return train_generator, val_generator
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=16, epochs=50, save_dir='models'):
        """
        Train the U-Net model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            batch_size: Batch size for training
            epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
            
        Returns:
            Training history
        """
        # Create model if not already created
        if self.model is None:
            self.build_unet_model(input_size=(*self.img_size, 3))
        
        # Create data generators
        train_generator, val_generator = self.create_data_generators(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Create callbacks
        os.makedirs(save_dir, exist_ok=True)
        model_checkpoint = ModelCheckpoint(
            os.path.join(save_dir, 'unet_segmentation.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train the model
        steps_per_epoch = len(X_train) // batch_size
        validation_steps = len(X_val) // batch_size
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[model_checkpoint, early_stopping, reduce_lr]
        )
        
        # Save the final model
        self.model.save(os.path.join(save_dir, 'unet_segmentation_final.h5'))
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """
        Plot training history.
        
        Args:
            history: Training history object
        """
        # Plot loss
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
        plt.savefig('unet_training_history.png')
        plt.close()
        
        # Plot IoU if available
        if 'io_u' in history.history:
            plt.figure(figsize=(8, 5))
            plt.plot(history.history['io_u'], label='Training IoU')
            plt.plot(history.history['val_io_u'], label='Validation IoU')
            plt.title('Intersection over Union (IoU)')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.legend()
            plt.savefig('unet_iou_history.png')
            plt.close()
    
    def load_model(self, model_path):
        """
        Load a pretrained model.
        
        Args:
            model_path: Path to the model file
        """
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return self.model
    
    def predict_mask(self, image):
        """
        Predict segmentation mask for an image.
        
        Args:
            image: Input image (can be path or numpy array)
            
        Returns:
            Predicted mask
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained. Call load_model() or train() first.")
        
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
        
        # Resize and normalize
        img_resized = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img_norm = img_resized / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_norm, axis=0)
        
        # Predict mask
        mask_pred = self.model.predict(img_batch)[0]
        
        # Binarize mask
        mask_binary = (mask_pred > 0.5).astype(np.uint8)
        
        # Resize mask to original size if needed
        if img.shape[:2] != self.img_size:
            mask_resized = cv2.resize(mask_binary, (img.shape[1], img.shape[0]))
        else:
            mask_resized = mask_binary
        
        return mask_resized
    
    def apply_mask(self, image, mask):
        """
        Apply mask to image.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Masked image
        """
        # Ensure mask is binary and has same shape as image
        if len(mask.shape) == 3 and mask.shape[2] == 1:
            mask_binary = mask[:, :, 0]
        else:
            mask_binary = mask
        
        # Apply mask to each channel
        masked_image = image.copy()
        for c in range(image.shape[2]):
            masked_image[:, :, c] = image[:, :, c] * mask_binary
            
        return masked_image
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test images
            y_test: Test masks
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained. Call load_model() or train() first.")
        
        # Evaluate model
        results = self.model.evaluate(X_test, y_test)
        print("Test Loss:", results[0])
        print("Test Accuracy:", results[1])
        if len(results) > 2:
            print("Test IoU:", results[2])
        
        # Visualize some predictions
        self.visualize_predictions(X_test, y_test, num_samples=5)
        
        return results
    
    def visualize_predictions(self, images, true_masks, num_samples=5, save_dir=None):
        """
        Visualize model predictions.
        
        Args:
            images: Test images
            true_masks: True masks
            num_samples: Number of samples to visualize
            save_dir: Directory to save visualizations (if None, display instead)
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained. Call load_model() or train() first.")
        
        # Select random samples
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
        
        for i, idx in enumerate(indices):
            image = images[idx]
            true_mask = true_masks[idx]
            
            # Predict mask
            pred_mask = self.model.predict(np.expand_dims(image, axis=0))[0]
            pred_mask_binary = (pred_mask > 0.5).astype(np.float32)
            
            # Apply masks
            true_masked = self.apply_mask(image, true_mask)
            pred_masked = self.apply_mask(image, pred_mask_binary)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(true_mask[:, :, 0], cmap='gray')
            axes[0, 1].set_title('True Mask')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(pred_mask_binary[:, :, 0], cmap='gray')
            axes[1, 0].set_title('Predicted Mask')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(pred_masked)
            axes[1, 1].set_title('Segmented Image')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'))
                plt.close()
            else:
                plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize trainer
    unet_trainer = UNetSegmentationTrainer(img_size=(256, 256))
    
    # Demo with synthetic data
    print("Creating synthetic data for demo...")
    X_train = np.random.random((100, 256, 256, 3))
    y_train = np.random.randint(0, 2, (100, 256, 256, 1)).astype(np.float32)
    X_val = np.random.random((20, 256, 256, 3))
    y_val = np.random.randint(0, 2, (20, 256, 256, 1)).astype(np.float32)
    
    # Build model
    model = unet_trainer.build_unet_model()
    print(model.summary())
    
    # Train for a few epochs (just for demonstration)
    print("Training for a few epochs (demo)...")
    unet_trainer.train(X_train, y_train, X_val, y_val, epochs=2, batch_size=8)
    
    print("Demo complete! To use with real data, provide actual image and mask directories.")

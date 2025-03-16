import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
import cv2
import pywt
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import median
from scipy.ndimage import gaussian_filter
import random


class SkinLesionPreprocessor:
    """
    Advanced preprocessing module for skin lesion images, incorporating:
    1. Advanced augmentation (traditional, Mixup, CutMix)
    2. Noise reduction (wavelet denoising, CLAHE)
    3. Segmentation (U-Net, background removal)
    """
    
    def __init__(self, img_size=(224, 224), segmentation_model_path=None):
        """
        Initialize the preprocessor.
        
        Args:
            img_size: Target image size (height, width)
            segmentation_model_path: Path to pre-trained U-Net segmentation model (if available)
        """
        self.img_size = img_size
        self.segmentation_model = None
        
        # Load segmentation model if path provided
        if segmentation_model_path and os.path.exists(segmentation_model_path):
            try:
                self.segmentation_model = load_model(segmentation_model_path)
                print(f"Loaded segmentation model from: {segmentation_model_path}")
            except Exception as e:
                print(f"Error loading segmentation model: {e}")
    
    # ==================== TRADITIONAL AUGMENTATION ====================
    
    def create_augmentation_generator(self, strong=False):
        """
        Create an ImageDataGenerator with appropriate augmentations.
        
        Args:
            strong: If True, apply more aggressive augmentation
            
        Returns:
            An ImageDataGenerator object
        """
        if strong:
            # Stronger augmentation for training
            return ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='reflect'
            )
        else:
            # Milder augmentation
            return ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.9, 1.1],
                fill_mode='reflect'
            )

    def apply_elastic_transform(self, image, alpha=35, sigma=5, random_state=None):
        """
        Apply elastic transformation to an image.
        
        Args:
            image: Input image (height, width, channels)
            alpha: Scaling factor for deformation
            sigma: Smoothing factor
            random_state: Random state for reproducibility
            
        Returns:
            Transformed image
        """
        if random_state is None:
            random_state = np.random.RandomState(None)
            
        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        
        distorted_image = np.zeros_like(image)
        for c in range(image.shape[2]):
            distorted_image[:,:,c] = map_coordinates(image[:,:,c], indices, order=1, mode='reflect').reshape(shape[:2])
            
        return distorted_image
    
    # ==================== MIXUP & CUTMIX AUGMENTATION ====================
    
    def mixup(self, image1, image2, label1, label2, alpha=0.2):
        """
        Apply Mixup augmentation to a pair of images.
        
        Args:
            image1, image2: Input images
            label1, label2: One-hot encoded labels
            alpha: Mixup strength parameter
            
        Returns:
            Mixed image and mixed label
        """
        # Generate mixup coefficient
        lambda_value = np.random.beta(alpha, alpha)
        
        # Apply mixup
        mixed_image = lambda_value * image1 + (1 - lambda_value) * image2
        mixed_label = lambda_value * label1 + (1 - lambda_value) * label2
        
        return mixed_image, mixed_label
    
    def cutmix(self, image1, image2, label1, label2, alpha=0.2):
        """
        Apply CutMix augmentation to a pair of images.
        
        Args:
            image1, image2: Input images
            label1, label2: One-hot encoded labels
            alpha: CutMix strength parameter
            
        Returns:
            Mixed image and mixed label
        """
        # Get image dimensions
        h, w, _ = image1.shape
        
        # Sample random box coordinates
        lambda_value = np.random.beta(alpha, alpha)
        
        # Calculate box size
        cut_ratio = np.sqrt(1.0 - lambda_value)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Calculate box center
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Calculate box boundaries
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Create mask
        mask = np.ones((h, w, 1))
        mask[y1:y2, x1:x2, :] = 0
        
        # Apply cutmix
        mixed_image = image1 * mask + image2 * (1 - mask)
        
        # Adjust lambda based on actual area
        lambda_value = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
        mixed_label = lambda_value * label1 + (1 - lambda_value) * label2
        
        return mixed_image, mixed_label
    
    def apply_mixup_batch(self, images, labels, alpha=0.2, prob=0.5):
        """
        Apply Mixup augmentation to a batch of images.
        
        Args:
            images: Batch of images [batch_size, height, width, channels]
            labels: One-hot encoded labels [batch_size, num_classes]
            alpha: Mixup strength parameter
            prob: Probability of applying mixup to each pair
            
        Returns:
            Augmented batch of images and labels
        """
        batch_size = images.shape[0]
        indices = np.random.permutation(batch_size)
        
        # Create output arrays
        mixed_images = np.copy(images)
        mixed_labels = np.copy(labels)
        
        # Apply mixup selectively
        for i in range(batch_size):
            if np.random.random() < prob:
                mixed_images[i], mixed_labels[i] = self.mixup(
                    images[i], images[indices[i]], 
                    labels[i], labels[indices[i]], 
                    alpha
                )
                
        return mixed_images, mixed_labels
    
    def apply_cutmix_batch(self, images, labels, alpha=0.2, prob=0.5):
        """
        Apply CutMix augmentation to a batch of images.
        
        Args:
            images: Batch of images [batch_size, height, width, channels]
            labels: One-hot encoded labels [batch_size, num_classes]
            alpha: CutMix strength parameter
            prob: Probability of applying cutmix to each pair
            
        Returns:
            Augmented batch of images and labels
        """
        batch_size = images.shape[0]
        indices = np.random.permutation(batch_size)
        
        # Create output arrays
        mixed_images = np.copy(images)
        mixed_labels = np.copy(labels)
        
        # Apply cutmix selectively
        for i in range(batch_size):
            if np.random.random() < prob:
                mixed_images[i], mixed_labels[i] = self.cutmix(
                    images[i], images[indices[i]], 
                    labels[i], labels[indices[i]], 
                    alpha
                )
                
        return mixed_images, mixed_labels
    
    # ==================== NOISE REDUCTION & DENOISING ====================
    
    def apply_wavelet_denoising(self, image, wavelet='db1', level=2, threshold=0.1):
        """
        Apply wavelet transform-based denoising to remove artifacts.
        
        Args:
            image: Input image
            wavelet: Wavelet type (e.g., 'db1', 'haar', 'sym2')
            level: Decomposition level
            threshold: Threshold for coefficient suppression
            
        Returns:
            Denoised image
        """
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Process each channel separately
        denoised = np.zeros_like(img_float)
        
        for c in range(img_float.shape[-1]):
            # Wavelet decomposition
            coeffs = pywt.wavedec2(img_float[:,:,c], wavelet, level=level)
            
            # Apply thresholding to detail coefficients
            modified_coeffs = list(coeffs)
            for i in range(1, len(modified_coeffs)):
                modified_coeffs[i] = tuple(np.sign(d) * np.maximum(np.abs(d) - threshold, 0) for d in modified_coeffs[i])
            
            # Reconstruct image
            denoised[:,:,c] = pywt.waverec2(modified_coeffs, wavelet)
        
        # Normalize to 0-1 range
        denoised = np.clip(denoised, 0, 1)
        
        return denoised
    
    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image: Input image (0-1 range)
            clip_limit: Clipping limit for contrast enhancement
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to 0-255 range
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply CLAHE to each channel
        enhanced = np.zeros_like(img_uint8)
        
        if img_uint8.shape[-1] == 1:  # Grayscale
            enhanced = clahe.apply(img_uint8)
        else:  # Color image
            # Convert to LAB color space
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge((l_enhanced, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to 0-1 range
        return enhanced.astype(np.float32) / 255.0
    
    def apply_median_filter(self, image, size=3):
        """
        Apply median filtering to reduce noise.
        
        Args:
            image: Input image
            size: Size of filter kernel
            
        Returns:
            Filtered image
        """
        # Process each channel
        filtered = np.zeros_like(image)
        for c in range(image.shape[-1]):
            filtered[:,:,c] = median(image[:,:,c], disk(size))
            
        return filtered
    
    # ==================== SEGMENTATION TECHNIQUES ====================
    
    def build_unet_model(self, input_size=(224, 224, 3)):
        """
        Build a U-Net segmentation model.
        
        Args:
            input_size: Input image shape (height, width, channels)
            
        Returns:
            U-Net model
        """
        inputs = Input(input_size)
        
        # Encoder (downsampling)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Bridge
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        
        # Decoder (upsampling)
        up5 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv4))
        merge5 = concatenate([conv3, up5], axis=3)
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
        
        up6 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
        merge6 = concatenate([conv2, up6], axis=3)
        conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
        
        up7 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv1, up7], axis=3)
        conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
        
        # Output
        outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
        
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def train_segmentation_model(self, images, masks, validation_split=0.2, epochs=20, batch_size=16):
        """
        Train the U-Net segmentation model.
        
        Args:
            images: Training images [num_samples, height, width, channels]
            masks: Binary masks [num_samples, height, width, 1]
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Trained model
        """
        # Create U-Net model
        model = self.build_unet_model(input_size=images[0].shape)
        
        # Create data generator with basic augmentation
        data_gen_args = dict(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Create image generators
        image_datagen = ImageDataGenerator(**data_gen_args, validation_split=validation_split)
        mask_datagen = ImageDataGenerator(**data_gen_args, validation_split=validation_split)
        
        # Fit generators on data
        seed = 42
        image_datagen.fit(images, seed=seed)
        mask_datagen.fit(masks, seed=seed)
        
        # Create generators for training and validation
        train_image_generator = image_datagen.flow(
            images, batch_size=batch_size, subset='training', seed=seed
        )
        train_mask_generator = mask_datagen.flow(
            masks, batch_size=batch_size, subset='training', seed=seed
        )
        val_image_generator = image_datagen.flow(
            images, batch_size=batch_size, subset='validation', seed=seed
        )
        val_mask_generator = mask_datagen.flow(
            masks, batch_size=batch_size, subset='validation', seed=seed
        )
        
        # Combine generators
        train_generator = zip(train_image_generator, train_mask_generator)
        val_generator = zip(val_image_generator, val_mask_generator)
        
        # Calculate steps per epoch
        train_steps = len(train_image_generator)
        val_steps = len(val_image_generator)
        
        # Train model
        history = model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=val_steps
        )
        
        # Save model
        model.save('skin_lesion_segmentation_model.h5')
        self.segmentation_model = model
        
        return model, history
    
    def segment_lesion(self, image):
        """
        Segment skin lesion using the U-Net model.
        
        Args:
            image: Input image
            
        Returns:
            Binary mask and segmented image
        """
        if self.segmentation_model is None:
            raise ValueError("Segmentation model not available. Please train or load a model first.")
        
        # Ensure image is in correct format
        if len(image.shape) == 3:
            img = np.expand_dims(image, axis=0)  # Add batch dimension
        else:
            img = image
            
        # Predict mask
        mask = self.segmentation_model.predict(img)
        
        # Post-process mask (threshold, remove small objects)
        mask_binary = (mask[0,:,:,0] > 0.5).astype(np.float32)
        
        # Apply mask to image
        segmented = image.copy()
        for c in range(image.shape[-1]):
            segmented[:,:,c] = image[:,:,c] * mask_binary
            
        return mask_binary, segmented
    
    def remove_background(self, image, threshold=0.1, apply_otsu=True):
        """
        Simple background removal using thresholding.
        
        Args:
            image: Input image (normalized 0-1)
            threshold: Threshold value (if not using Otsu)
            apply_otsu: Whether to apply Otsu's method for thresholding
            
        Returns:
            Image with background removed
        """
        # Convert to grayscale if color
        if image.shape[-1] > 1:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image[:,:,0] * 255).astype(np.uint8)
            
        # Apply Otsu's thresholding or fixed threshold
        if apply_otsu:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(gray, threshold * 255, 255, cv2.THRESH_BINARY)
            
        # Process mask to reduce noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Fill holes
        mask_floodfill = mask.copy()
        h, w = mask.shape
        mask_temp = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(mask_floodfill, mask_temp, (0,0), 255)
        mask_filled = cv2.bitwise_not(mask_floodfill)
        mask_final = mask | mask_filled
        
        # Apply mask to image
        mask_final = mask_final.astype(np.float32) / 255.0
        result = image.copy()
        for c in range(image.shape[-1]):
            result[:,:,c] = image[:,:,c] * mask_final
            
        return result
    
    # ==================== COMBINED PREPROCESSING PIPELINE ====================
    
    def preprocess_pipeline(self, image, denoising=True, enhance_contrast=True, segment=False):
        """
        Full preprocessing pipeline combining multiple techniques.
        
        Args:
            image: Input image (normalized 0-1)
            denoising: Apply wavelet denoising
            enhance_contrast: Apply CLAHE
            segment: Apply U-Net segmentation if model available
            
        Returns:
            Preprocessed image
        """
        # Resize to target size
        if image.shape[:2] != self.img_size:
            image = cv2.resize(image, self.img_size[::-1])
            
        # Ensure image is in 0-1 range
        if image.max() > 1.0:
            image = image / 255.0
            
        # Apply denoising if requested
        if denoising:
            image = self.apply_wavelet_denoising(image)
            
        # Apply contrast enhancement if requested
        if enhance_contrast:
            image = self.apply_clahe(image)
            
        # Apply segmentation if requested and model is available
        if segment and self.segmentation_model is not None:
            _, image = self.segment_lesion(image)
            
        return image
    
    def batch_preprocess(self, images, **kwargs):
        """
        Apply preprocessing pipeline to a batch of images.
        
        Args:
            images: Batch of images [batch_size, height, width, channels]
            **kwargs: Additional arguments for preprocess_pipeline
            
        Returns:
            Preprocessed batch of images
        """
        return np.array([self.preprocess_pipeline(img, **kwargs) for img in images])
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def visualize_preprocessing(self, image, save_path=None):
        """
        Visualize different preprocessing steps on a single image.
        
        Args:
            image: Input image
            save_path: Path to save visualization (if None, display instead)
        """
        # Ensure image is in 0-1 range
        if image.max() > 1.0:
            image = image / 255.0
            
        # Apply different preprocessing steps
        denoised = self.apply_wavelet_denoising(image)
        enhanced = self.apply_clahe(image)
        denoised_enhanced = self.apply_clahe(denoised)
        
        # Apply segmentation if model available
        if self.segmentation_model is not None:
            mask, segmented = self.segment_lesion(image)
            bg_removed = self.remove_background(image)
            
            # Create figure with 6 subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original')
            
            axes[0, 1].imshow(denoised)
            axes[0, 1].set_title('Wavelet Denoised')
            
            axes[0, 2].imshow(enhanced)
            axes[0, 2].set_title('CLAHE Enhanced')
            
            axes[1, 0].imshow(denoised_enhanced)
            axes[1, 0].set_title('Denoised + Enhanced')
            
            axes[1, 1].imshow(mask, cmap='gray')
            axes[1, 1].set_title('Segmentation Mask')
            
            axes[1, 2].imshow(segmented)
            axes[1, 2].set_title('Segmented Image')
        else:
            # Create figure with 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original')
            
            axes[0, 1].imshow(denoised)
            axes[0, 1].set_title('Wavelet Denoised')
            
            axes[1, 0].imshow(enhanced)
            axes[1, 0].set_title('CLAHE Enhanced')
            
            axes[1, 1].imshow(denoised_enhanced)
            axes[1, 1].set_title('Denoised + Enhanced')
        
        # Remove axis ticks
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


# Sample usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = SkinLesionPreprocessor(img_size=(224, 224))
    
    # Load a sample image
    sample_img = cv2.imread('sample_lesion.jpg')
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB) / 255.0
    
    # Apply preprocessing pipeline
    processed_img = preprocessor.preprocess_pipeline(
        sample_img, 
        denoising=True, 
        enhance_contrast=True, 
        segment=False
    )
    
    # Visualize results
    preprocessor.visualize_preprocessing(sample_img, save_path='preprocessing_visualization.png')
    
    print("Preprocessing complete!")

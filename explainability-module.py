import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# TensorFlow specific imports
import tensorflow as tf
from tensorflow import keras

# Import other explainability libraries as needed
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

class ExplainabilityTools:
    """
    A class to provide various explainability techniques for TensorFlow models
    in skin disease classification
    """
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        
    def preprocess_image(self, img_path, img_size=(224, 224)):
        """
        Load and preprocess an image for the model
        """
        # Load image using TensorFlow utilities
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to 0-1
        
        return img_array
    
    def predict(self, img_array):
        """
        Make a prediction on an image
        """
        # Get model predictions
        predictions = self.model.predict(img_array)
        
        # Get top predicted class
        pred_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_class_idx]
        
        # Get top 3 predictions
        top3_idx = np.argsort(predictions[0])[::-1][:3]
        top3_classes = [(self.class_names[i], predictions[0][i]) for i in top3_idx]
        
        return pred_class_idx, confidence, top3_classes, predictions[0]
    
    def get_gradcam(self, img_array, pred_class=None, layer_name=None):
        """
        Generate Grad-CAM visualization for an image
        
        Args:
            img_array: Preprocessed image array (batch, height, width, channels)
            pred_class: Class index to explain (if None, uses predicted class)
            layer_name: Name of the convolutional layer to use (if None, uses last conv layer)
            
        Returns:
            original_img: Original image
            cam_image: GradCAM visualization
            heatmap: Raw heatmap
        """
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:
                    layer_name = layer.name
                    break
                if hasattr(layer, 'layers'):  # For models with nested layers like ResNet
                    for nested_layer in reversed(layer.layers):
                        if len(nested_layer.output_shape) == 4:
                            layer_name = nested_layer.name
                            break
        
        # Create the gradient model
        try:
            grad_model = tf.keras.models.Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(layer_name).output, self.model.output]
            )
        except ValueError:
            # If we can't find the layer by name, try to find it inside the model
            found = False
            for layer in self.model.layers:
                if hasattr(layer, 'layers'):  # Check if it's a container
                    try:
                        nested_layer = layer.get_layer(layer_name)
                        grad_model = tf.keras.models.Model(
                            inputs=[self.model.inputs],
                            outputs=[nested_layer.output, self.model.output]
                        )
                        found = True
                        break
                    except ValueError:
                        continue
            
            if not found:
                raise ValueError(f"Could not find layer named {layer_name}")
        
        # Determine the class index to explain
        if pred_class is None:
            pred_class, _, _, _ = self.predict(img_array)
        
        # Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_channel = predictions[:, pred_class]
        
        # Get gradients of the output with respect to the last conv layer
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by importance
        conv_outputs = conv_outputs[0]
        weighted_output = tf.einsum('ijk,l->ijkl', conv_outputs, pooled_grads)
        heatmap = tf.reduce_sum(weighted_output, axis=3).numpy()
        
        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        # Resize the heatmap to the size of the input image
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        
        # Convert to RGB heatmap
        heatmap_rgb = np.uint8(255 * plt.cm.jet(heatmap_resized)[:, :, :3])
        
        # Convert the original image
        original_img = img_array[0]
        
        # Superimpose the heatmap
        cam_image = (heatmap_rgb * 0.4 + original_img * 255 * 0.6).astype(np.uint8)
        
        return original_img, cam_image, heatmap_resized
    
    def get_gradcam_plus_plus(self, img_array, pred_class=None, layer_name=None):
        """
        Generate Grad-CAM++ visualization
        
        This is an enhanced version of GradCAM that provides better localization
        """
        # This is a simplified version of GradCAM++ based on TensorFlow
        # First, get the regular GradCAM components
        original_img, _, heatmap = self.get_gradcam(img_array, pred_class, layer_name)
        
        # Apply a different weighting scheme to make it more focused (approximation)
        # Real Grad-CAM++ would require more complex gradient calculations
        heatmap = np.power(heatmap, 2)
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        # Convert to RGB heatmap
        heatmap_rgb = np.uint8(255 * plt.cm.jet(heatmap)[:, :, :3])
        
        # Superimpose the heatmap
        cam_image = (heatmap_rgb * 0.4 + original_img * 255 * 0.6).astype(np.uint8)
        
        return original_img, cam_image, heatmap
    
    def get_lime_explanation(self, img_array, num_samples=1000, num_features=5):
        """
        Generate LIME explanation for the image
        
        Args:
            img_array: Preprocessed image array (batch, height, width, channels)
            num_samples: Number of perturbed samples to generate
            num_features: Number of top features to highlight
            
        Returns:
            lime_img: LIME visualization with positive features
            lime_img_neg: LIME visualization with negative features
            explanation: LIME explanation object
            pred_class: Predicted class index
            confidence: Prediction confidence
        """
        # Create explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Define the prediction function
        def predict_fn(images):
            # Reshape and process images for model
            batch = np.vstack([np.expand_dims(img, axis=0) for img in images])
            
            # Get model predictions
            preds = self.model.predict(batch)
            return preds
        
        # Get explanation
        original_img = img_array[0].copy()
        explanation = explainer.explain_instance(
            original_img,
            predict_fn,
            top_labels=5,
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get predicted class
        pred_class, confidence, _, _ = self.predict(img_array)
        
        # Get visualization with positive features
        temp, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=True,
            num_features=num_features,
            hide_rest=True
        )
        lime_img = mark_boundaries(temp, mask)
        
        # Get visualization with negative features
        temp_neg, mask_neg = explanation.get_image_and_mask(
            pred_class,
            positive_only=False,
            negative_only=True,
            num_features=num_features,
            hide_rest=True
        )
        lime_img_neg = mark_boundaries(temp_neg, mask_neg, color=(1, 0, 0))
        
        return lime_img, lime_img_neg, explanation, pred_class, confidence
    
    def get_shap_explanation(self, img_array, background_images, num_samples=100):
        """
        Generate SHAP explanation for the image
        
        Args:
            img_array: Preprocessed image array (batch, height, width, channels)
            background_images: List of background image arrays for SHAP
            num_samples: Number of samples for SHAP computation
            
        Returns:
            shap_img: SHAP visualization
            shap_values_abs: Absolute SHAP values
            shap_values: Raw SHAP values
            pred_class: Predicted class index
            confidence: Prediction confidence
        """
        # Create a background dataset
        background = np.vstack(background_images)
        
        # Create the explainer
        explainer = shap.DeepExplainer(self.model, background)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(img_array)
        
        # Get prediction information
        pred_class, confidence, _, _ = self.predict(img_array)
        
        # Get the SHAP values for the predicted class
        shap_for_pred = shap_values[pred_class]
        
        # Calculate absolute sum of SHAP values across channels
        shap_abs = np.abs(shap_for_pred[0]).sum(axis=2)
        
        # Normalize for visualization
        shap_norm = shap_abs / np.max(shap_abs) if np.max(shap_abs) > 0 else shap_abs
        
        # Create heatmap
        heatmap = plt.cm.jet(shap_norm)[:, :, :3]
        
        # Superimpose on original image
        original_img = img_array[0]
        shap_img = (original_img * 0.7 + heatmap * 0.3)
        
        return shap_img, shap_norm, shap_values, pred_class, confidence
    
    def get_occlusion_sensitivity(self, img_array, patch_size=8, stride=4):
        """
        Generate occlusion sensitivity map
        
        This technique systematically occludes parts of the image and 
        measures the change in prediction to create a sensitivity map.
        
        Args:
            img_array: Preprocessed image array (batch, height, width, channels)
            patch_size: Size of occlusion patch
            stride: Stride for moving the occlusion patch
            
        Returns:
            sensitivity_map: Visualization of sensitivity
            heatmap: Raw sensitivity values
            pred_class: Predicted class index
            confidence: Prediction confidence
        """
        # Get original prediction
        pred_class, confidence, _, orig_predictions = self.predict(img_array)
        
        # Get image dimensions
        height, width, channels = img_array[0].shape
        
        # Create heatmap with same dimensions as image
        heatmap = np.zeros((height, width))
        
        # Create a copy of the image to occlude
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # Create a copy with the patch occluded (set to gray)
                occluded_img = img_array[0].copy()
                occluded_img[y:y+patch_size, x:x+patch_size, :] = 0.5  # Set to gray
                
                # Get prediction for occluded image
                occluded_pred = self.model.predict(np.expand_dims(occluded_img, axis=0))[0]
                
                # Calculate change in confidence for the predicted class
                score_drop = orig_predictions[pred_class] - occluded_pred[pred_class]
                
                # Assign the score drop to the corresponding part of the heatmap
                heatmap[y:y+patch_size, x:x+patch_size] += score_drop
        
        # Normalize heatmap
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) if np.max(heatmap) > np.min(heatmap) else heatmap
        
        # Resize heatmap to the size of the input image
        heatmap_resized = cv2.resize(heatmap, (width, height))
        
        # Convert to RGB heatmap
        heatmap_rgb = np.uint8(255 * plt.cm.jet(heatmap_resized)[:, :, :3])
        
        # Superimpose the heatmap
        original_img = img_array[0]
        sensitivity_map = (heatmap_rgb * 0.4 + original_img * 255 * 0.6).astype(np.uint8)
        
        return original_img, sensitivity_map, heatmap_resized, pred_class, confidence
    
    def get_integrated_gradients(self, img_array, steps=50, baseline=None):
        """
        Generate Integrated Gradients explanation
        
        Args:
            img_array: Preprocessed image array (batch, height, width, channels)
            steps: Number of steps for the integration
            baseline: Baseline image (if None, uses black image)
            
        Returns:
            ig_img: Integrated Gradients visualization
            attributions_sum: Sum of attributions
            pred_class: Predicted class index
            confidence: Prediction confidence
        """
        # Get prediction
        pred_class, confidence, _, _ = self.predict(img_array)
        
        # Create baseline if not provided (black image)
        if baseline is None:
            baseline = np.zeros_like(img_array)
        
        # Create the interpolation path from baseline to input
        alphas = np.linspace(0, 1, steps)
        interpolated_images = [baseline + alpha * (img_array - baseline) for alpha in alphas]
        interpolated_images = np.vstack(interpolated_images)
        
        # Get the gradients for each interpolated image
        with tf.GradientTape() as tape:
            inputs = tf.convert_to_tensor(interpolated_images, dtype=tf.float32)
            tape.watch(inputs)
            outputs = self.model(inputs)[:, pred_class]
        
        grads = tape.gradient(outputs, inputs).numpy()
        
        # Calculate the integral approximation
        grads = (grads[:-1] + grads[1:]) / 2.0  # Trapezoid rule
        avg_grads = np.average(grads, axis=0)
        
        # Scale the gradients by the input difference from baseline
        attributions = avg_grads * (img_array - baseline)
        
        # Sum across RGB channels
        attributions_sum = np.sum(np.abs(attributions), axis=3)[0]
        
        # Normalize for visualization
        attr_norm = attributions_sum / np.max(attributions_sum) if np.max(attributions_sum) > 0 else attributions_sum
        
        # Create heatmap
        heatmap = plt.cm.jet(attr_norm)[:, :, :3]
        
        # Superimpose on original image
        original_img = img_array[0]
        ig_img = (original_img * 0.7 + heatmap * 0.3)
        
        return ig_img, attr_norm, attributions, pred_class, confidence
    
    def create_multi_explanation_visualization(self, img_array, background_images=None):
        """
        Create a comprehensive visualization with multiple explainability techniques
        
        Args:
            img_array: Preprocessed image array (batch, height, width, channels)
            background_images: List of background image arrays for SHAP
            
        Returns:
            fig: Matplotlib figure with multiple explanations
        """
        # Get prediction
        pred_class, confidence, top3_classes, _ = self.predict(img_array)
        class_name = self.class_names[pred_class]
        
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Original image with class probabilities
        ax0 = plt.subplot2grid((3, 4), (0, 0))
        ax0.imshow(img_array[0])
        ax0.set_title('Original Image')
        ax0.axis('off')
        
        # 2. GradCAM
        try:
            _, gradcam_img, _ = self.get_gradcam(img_array)
            ax1 = plt.subplot2grid((3, 4), (0, 1))
            ax1.imshow(gradcam_img / 255.0)
            ax1.set_title('GradCAM')
            ax1.axis('off')
        except Exception as e:
            print(f"GradCAM error: {e}")
            ax1 = plt.subplot2grid((3, 4), (0, 1))
            ax1.text(0.5, 0.5, f"GradCAM error", ha='center', va='center')
            ax1.axis('off')
        
        # 3. GradCAM++
        try:
            _, gradcam_plus_img, _ = self.get_gradcam_plus_plus(img_array)
            ax2 = plt.subplot2grid((3, 4), (0, 2))
            ax2.imshow(gradcam_plus_img / 255.0)
            ax2.set_title('GradCAM++')
            ax2.axis('off')
        except Exception as e:
            print(f"GradCAM++ error: {e}")
            ax2 = plt.subplot2grid((3, 4), (0, 2))
            ax2.text(0.5, 0.5, f"GradCAM++ error", ha='center', va='center')
            ax2.axis('off')
        
        # 4. Class probabilities
        ax3 = plt.subplot2grid((3, 4), (0, 3))
        labels = [class_name for class_name, _ in top3_classes]
        values = [prob for _, prob in top3_classes]
        y_pos = np.arange(len(labels))
        ax3.barh(y_pos, values)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels)
        ax3.invert_yaxis()
        ax3.set_xlabel('Probability')
        ax3.set_title('Top 3 Class Probabilities')
        ax3.set_xlim(0, 1)
        
        # 5. LIME positive features
        try:
            lime_img, lime_img_neg, _, _, _ = self.get_lime_explanation(img_array)
            ax4 = plt.subplot2grid((3, 4), (1, 0))
            ax4.imshow(lime_img)
            ax4.set_title('LIME (Positive Features)')
            ax4.axis('off')
            
            # 6. LIME negative features
            ax5 = plt.subplot2grid((3, 4), (1, 1))
            ax5.imshow(lime_img_neg)
            ax5.set_title('LIME (Negative Features)')
            ax5.axis('off')
        except Exception as e:
            print(f"LIME error: {e}")
            ax4 = plt.subplot2grid((3, 4), (1, 0))
            ax4.text(0.5, 0.5, f"LIME error", ha='center', va='center')
            ax4.axis('off')
            
            ax5 = plt.subplot2grid((3, 4), (1, 1))
            ax5.text(0.5, 0.5, f"LIME error", ha='center', va='center')
            ax5.axis('off')
        
        # 7. Integrated Gradients
        try:
            ig_img, _, _, _, _ = self.get_integrated_gradients(img_array)
            ax6 = plt.subplot2grid((3, 4), (1, 2))
            ax6.imshow(ig_img)
            ax6.set_title('Integrated Gradients')
            ax6.axis('off')
        except Exception as e:
            print(f"Integrated Gradients error: {e}")
            ax6 = plt.subplot2grid((3, 4), (1, 2))
            ax6.text(0.5, 0.5, f"IG error", ha='center', va='center')
            ax6.axis('off')
        
        # 8. Occlusion Sensitivity
        try:
            _, sensitivity_map, _, _, _ = self.get_occlusion_sensitivity(img_array)
            ax7 = plt.subplot2grid((3, 4), (1, 3))
            ax7.imshow(sensitivity_map / 255.0)
            ax7.set_title('Occlusion Sensitivity')
            ax7.axis('off')
        except Exception as e:
            print(f"Occlusion Sensitivity error: {e}")
            ax7 = plt.subplot2grid((3, 4), (1, 3))
            ax7.text(0.5, 0.5, f"Occlusion error", ha='center', va='center')
            ax7.axis('off')
        
        # 9. SHAP
        if background_images is not None:
            try:
                shap_img, _, _, _, _ = self.get_shap_explanation(img_array, background_images)
                ax8 = plt.subplot2grid((3, 4), (2, 0))
                ax8.imshow(shap_img)
                ax8.set_title('SHAP')
                ax8.axis('off')
            except Exception as e:
                print(f"SHAP error: {e}")
                ax8 = plt.subplot2grid((3, 4), (2, 0))
                ax8.text(0.5, 0.5, f"SHAP error", ha='center', va='center')
                ax8.axis('off')
        else:
            ax8 = plt.subplot2grid((3, 4), (2, 0))
            ax8.text(0.5, 0.5, "SHAP requires background images", ha='center', va='center')
            ax8.axis('off')
        
        # Fill remaining subplots with different layer GradCAMs
        layer_names = []
        
        # Find conv layers
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):  # For models with nested layers
                for nested_layer in layer.layers:
                    if isinstance(nested_layer, keras.layers.Conv2D):
                        layer_names.append(nested_layer.name)
            elif isinstance(layer, keras.layers.Conv2D):
                layer_names.append(layer.name)
        
        # Sample a few layers from different depths
        if len(layer_names) > 3:
            indices = np.linspace(0, len(layer_names)-1, 3).astype(int)
            selected_layers = [layer_names[i] for i in indices]
            
            # GradCAM for different layers
            for i, layer_name in enumerate(selected_layers):
                ax = plt.subplot2grid((3, 4), (2, i+1))
                try:
                    _, gradcam_img, _ = self.get_gradcam(img_array, layer_name=layer_name)
                    ax.imshow(gradcam_img / 255.0)
                    ax.set_title(f'GradCAM: {layer_name.split("/")[-1]}')
                    ax.axis('off')
                except Exception as e:
                    print(f"Layer GradCAM error: {e}")
                    ax.text(0.5, 0.5, f"Error: {layer_name}", ha='center', va='center')
                    ax.axis('off')
        
        # Add title with prediction information
        fig.suptitle(f"Prediction: {class_name} (Confidence: {confidence:.2f})", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        return fig
    
    def evaluate_explainability_metrics(self, test_df, num_samples=50):
        """
        Evaluate various explainability metrics on test samples
        
        Args:
            test_df: DataFrame with test set information
            num_samples: Number of samples to evaluate
            
        Returns:
            results_df: DataFrame with explainability metrics
        """
        results = []
        
        # Sample a subset of test images
        sampled_indices = np.random.choice(len(test_df), num_samples, replace=False)
        
        # Get background images for SHAP
        background_images = []
        for i in range(min(10, len(test_df))):
            img_path = test_df.iloc[i]['path']
            background_images.append(self.preprocess_image(img_path)[0])
        
        # Process each sampled image
        for idx in sampled_indices:
            img_path = test_df.iloc[idx]['path']
            true_label = test_df.iloc[idx]['label']
            
            # Load and preprocess image
            img_array = self.preprocess_image(img_path)
            
            # Get prediction
            pred_class, confidence, _, _ = self.predict(img_array)
            correct = (pred_class == true_label)
            
            # GradCAM consistency score (compare different layer depths)
            try:
                # Find some conv layers
                conv_layers = []
                for layer in self.model.layers:
                    if hasattr(layer, 'layers'):  # For models with nested layers
                        for nested_layer in layer.layers:
                            if isinstance(nested_layer, keras.layers.Conv2D):
                                conv_layers.append(nested_layer.name)
                    elif isinstance(layer, keras.layers.Conv2D):
                        conv_layers.append(layer.name)
                
                if len(conv_layers) > 1:
                    # Get GradCAM for two different layers
                    _, _, heatmap1 = self.get_gradcam(img_array, layer_name=conv_layers[0])
                    _, _, heatmap2 = self.get_gradcam(img_array, layer_name=conv_layers[-1])
                    
                    # Calculate correlation
                    heatmap1_flat = heatmap1.flatten()
                    heatmap2_flat = heatmap2.flatten()
                    
                    # Calculate correlation
                    gradcam_consistency = np.corrcoef(heatmap1_flat, heatmap2_flat)[0, 1]
                else:
                    gradcam_consistency = None
            except:
                gradcam_consistency = None
            
            # LIME stability (not implemented fully)
            lime_stability = None
            
            # Collect results
            result = {
                'image_idx': idx,
                'true_label': true_label,
                'true_class': self.class_names[true_label],
                'pred_label': pred_class,
                'pred_class': self.class_names[pred_class],
                'confidence': confidence,
                'correct_prediction': correct,
                'gradcam_consistency': gradcam_consistency,
                'lime_stability': lime_stability
            }
            
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        accuracy = results_df['correct_prediction'].mean()
        avg_confidence = results_df['confidence'].mean()
        avg_gradcam_consistency = results_df['gradcam_consistency'].mean()
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Average confidence: {avg_confidence:.4f}")
        print(f"Average GradCAM consistency: {avg_gradcam_consistency:.4f}")
        
        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results_df, x='confidence', hue='correct_prediction', bins=20, multiple='stack')
        plt.title('Confidence Distribution by Prediction Correctness')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.savefig('confidence_distribution.png')
        plt.close()
        
        # Plot GradCAM consistency
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results_df, x='gradcam_consistency', bins=20)
        plt.title('GradCAM Consistency Distribution')
        plt.xlabel('GradCAM Consistency (Correlation)')
        plt.ylabel('Count')
        plt.savefig('gradcam_consistency.png')
        plt.close()
        
        return results_df
    
    def compare_target_vs_predicted_class(self, img_array, target_class=None):
        """
        Compare GradCAM visualizations for the target class vs predicted class
        
        Args:
            img_array: Preprocessed image array (batch, height, width, channels)
            target_class: Target class to compare with predicted class
            
        Returns:
            fig: Matplotlib figure with comparison
            pred_class: Predicted class index
            target_class: Target class index
        """
        # Get prediction
        pred_class, _, _, predictions = self.predict(img_array)
        
        # Set target class if not provided
        if target_class is None or target_class == pred_class:
            # Find second most likely class
            sorted_idx = np.argsort(predictions)[::-1]
            target_class = sorted_idx[1]  # Second most likely class
        
        # Get GradCAM for predicted class
        _, gradcam_pred, _ = self.get_gradcam(img_array, pred_class=pred_class)
        
        # Get GradCAM for target class
        _, gradcam_target, _ = self.get_gradcam(img_array, pred_class=target_class)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_array[0])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # GradCAM for predicted class
        axes[1].imshow(gradcam_pred / 255.0)
        axes[1].set_title(f'GradCAM for Predicted Class\n({self.class_names[pred_class]})')
        axes[1].axis('off')
        
        # GradCAM for target class
        axes[2].imshow(gradcam_target / 255.0)
        axes[2].set_title(f'GradCAM for Target Class\n({self.class_names[target_class]})')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig, pred_class, target_class
    
    def generate_explanation_report(self, img_path, background_images=None, save_path=None):
        """
        Generate a comprehensive explanation report for a single image
        
        Args:
            img_path: Path to the image file
            background_images: Background images for SHAP (optional)
            save_path: Path to save the report (optional)
            
        Returns:
            fig: Matplotlib figure with explanations
            report: Dictionary with prediction information
        """
        # Load and preprocess the image
        img_array = self.preprocess_image(img_path)
        
        # Get prediction
        pred_class, confidence, top3_classes, _ = self.predict(img_array)
        class_name = self.class_names[pred_class]
        
        # Create multi-explanation visualization
        fig = self.create_multi_explanation_visualization(img_array, background_images)
        
        # Save figure if path is provided
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        
        # Create report
        report = {
            'Prediction': class_name,
            'Confidence': confidence,
            'Top 3 Classes': top3_classes
        }
        
        return fig, report

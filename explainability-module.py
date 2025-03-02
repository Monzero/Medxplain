import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# For explainability
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
from captum.attr import IntegratedGradients, Occlusion, DeepLift, NoiseTunnel, LayerAttribution
from captum.attr import visualization as viz

class ExplainabilityTools:
    """
    A class to provide various explainability techniques for deep learning models
    in skin disease classification
    """
    def __init__(self, model, class_names, device):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_image_for_display(self, img_tensor):
        """
        Preprocess image tensor for display
        """
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        return img_np
        
    def get_prediction(self, img_tensor):
        """
        Get model prediction for an image
        """
        with torch.no_grad():
            output = self.model(img_tensor.unsqueeze(0).to(self.device))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            pred_class = predicted.item()
            conf_value = confidence.item()
            
        return pred_class, conf_value, probabilities[0].cpu().numpy()
        
    def get_gradcam(self, img_tensor, target_class=None, method='gradcam'):
        """
        Generate CAM visualization for the given image using different methods
        
        Args:
            img_tensor: Input image tensor
            target_class: Target class for CAM (None for predicted class)
            method: One of 'gradcam', 'gradcam++', 'scorecam', 'xgradcam', 'ablationcam', 'eigencam'
        """
        # Prepare the input
        input_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Select the CAM method
        cam_method = {
            'gradcam': GradCAM,
            'gradcam++': GradCAMPlusPlus,
            'scorecam': ScoreCAM,
            'xgradcam': XGradCAM,
            'ablationcam': AblationCAM,
            'eigencam': EigenCAM
        }
        
        # Get target layer (varies by model architecture)
        if hasattr(self.model, 'get_gradcam_layer'):
            target_layer = [self.model.get_gradcam_layer()]
        else:
            # For ResNet architecture
            try:
                target_layer = [self.model.model.layer4[-1]]
            except AttributeError:
                target_layer = [self.model.layer4[-1]]
        
        # Create the CAM object
        cam = cam_method.get(method.lower(), GradCAM)(model=self.model, target_layers=target_layer)
        
        # Define target
        if target_class is None:
            pred_class, _, _ = self.get_prediction(img_tensor)
            target_class = pred_class
            
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Convert tensor to numpy for visualization
        rgb_img = self.preprocess_image_for_display(img_tensor)
        
        # Overlay CAM on original image
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        return cam_image, grayscale_cam, target_class
    
    def get_lime_explanation(self, img_tensor, num_samples=1000, num_features=5):
        """
        Generate LIME explanation for the given image
        """
        # Create the LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Convert the PyTorch tensor to a numpy array for LIME
        img_np = self.preprocess_image_for_display(img_tensor)
        
        # Define the prediction function for LIME
        def batch_predict(images):
            batch = torch.stack(tuple(torch.from_numpy(i.transpose(2, 0, 1)).float() for i in images))
            batch = batch.to(self.device)
            self.model.eval()
            with torch.no_grad():
                output = self.model(batch)
            return output.detach().cpu().numpy()
        
        # Generate explanation
        explanation = explainer.explain_instance(
            img_np, 
            batch_predict, 
            top_labels=5, 
            hide_color=0, 
            num_samples=num_samples
        )
        
        # Get the top predicted class
        pred_class, conf_value, _ = self.get_prediction(img_tensor)
        
        # Get the explanation for the predicted class
        temp, mask = explanation.get_image_and_mask(
            pred_class, 
            positive_only=True, 
            num_features=num_features, 
            hide_rest=True
        )
        
        # Create the visualization
        lime_img = mark_boundaries(temp, mask)
        
        # Also get the negative importance
        temp_neg, mask_neg = explanation.get_image_and_mask(
            pred_class, 
            positive_only=False, 
            negative_only=True,
            num_features=num_features, 
            hide_rest=True
        )
        
        lime_img_neg = mark_boundaries(temp_neg, mask_neg, color=(1, 0, 0))  # Red for negative
        
        return lime_img, lime_img_neg, explanation, pred_class, conf_value
    
    def get_shap_explanation(self, img_tensor, background_imgs, n_samples=100):
        """
        Generate SHAP explanation for the given image
        
        Args:
            img_tensor: Input image tensor
            background_imgs: List of background image tensors
            n_samples: Number of samples for SHAP
        """
        # Convert background images to a tensor
        background = torch.stack(background_imgs).to(self.device)
        
        # Define a function to get model outputs
        def model_output(images):
            self.model.eval()
            with torch.no_grad():
                return self.model(images)
        
        # Create the explainer
        explainer = shap.DeepExplainer(model_output, background)
        
        # Get SHAP values
        input_tensor = img_tensor.unsqueeze(0).to(self.device)
        shap_values = explainer.shap_values(input_tensor)
        
        # Get the prediction
        pred_class, conf_value, _ = self.get_prediction(img_tensor)
        
        # Create visualization
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        
        # Normalize image for display
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Create SHAP visualization
        shap_img = np.zeros_like(img_np)
        abs_shap_values = np.abs(shap_values[pred_class][0]).transpose(1, 2, 0)
        
        # Sum across channels for importance
        abs_shap_sum = abs_shap_values.sum(axis=2)
        
        # Normalize for visualization
        abs_shap_norm = (abs_shap_sum - abs_shap_sum.min()) / (abs_shap_sum.max() - abs_shap_sum.min() + 1e-10)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * abs_shap_norm), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Blend original image with heatmap
        shap_img = 0.7 * img_np + 0.3 * heatmap
        
        return shap_img, abs_shap_norm, shap_values, pred_class, conf_value
    
    def get_integrated_gradients(self, img_tensor, n_steps=50):
        """
        Generate Integrated Gradients explanation
        """
        # Prepare input
        input_tensor = img_tensor.unsqueeze(0).to(self.device).requires_grad_()
        
        # Get the prediction
        pred_class, conf_value, _ = self.get_prediction(img_tensor)
        
        # Create the IG attributor
        ig = IntegratedGradients(self.model)
        
        # Compute attributions
        attributions = ig.attribute(input_tensor, target=pred_class, n_steps=n_steps)
        
        # Convert to numpy for visualization
        img_np = self.preprocess_image_for_display(img_tensor)
        attr_np = attributions[0].cpu().detach().numpy().transpose(1, 2, 0)
        
        # Compute magnitude of attributions
        attr_magnitude = np.abs(attr_np).sum(axis=2)
        
        # Normalize for visualization
        attr_norm = (attr_magnitude - attr_magnitude.min()) / (attr_magnitude.max() - attr_magnitude.min() + 1e-10)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attr_norm), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Blend original image with heatmap
        ig_img = 0.7 * img_np + 0.3 * heatmap
        
        return ig_img, attr_norm, attributions, pred_class, conf_value
    
    def get_occlusion_map(self, img_tensor, window=(8, 8, 3), stride=(4, 4, 3)):
        """
        Generate occlusion sensitivity map
        """
        # Prepare input
        input_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Get the prediction
        pred_class, conf_value, _ = self.get_prediction(img_tensor)
        
        # Create the occlusion attributor
        occlusion = Occlusion(self.model)
        
        # Compute attributions
        attributions = occlusion.attribute(
            input_tensor,
            target=pred_class,
            sliding_window_shapes=window,
            strides=stride,
            baselines=0.0,
            show_progress=True
        )
        
        # Convert to numpy for visualization
        img_np = self.preprocess_image_for_display(img_tensor)
        attr_np = attributions[0].cpu().detach().numpy().transpose(1, 2, 0)
        
        # Compute magnitude of attributions
        attr_magnitude = np.abs(attr_np).sum(axis=2)
        
        # Normalize for visualization
        attr_norm = (attr_magnitude - attr_magnitude.min()) / (attr_magnitude.max() - attr_magnitude.min() + 1e-10)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attr_norm), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Blend original image with heatmap
        occlusion_img = 0.7 * img_np + 0.3 * heatmap
        
        return occlusion_img, attr_norm, attributions, pred_class, conf_value
    
    def create_multi_explanation_visualization(self, img_tensor, background_imgs=None):
        """
        Create a visualization that combines multiple explainability techniques
        """
        # Get the original image
        img_np = self.preprocess_image_for_display(img_tensor)
        
        # Get the prediction
        pred_class, conf_value, probabilities = self.get_prediction(img_tensor)
        class_name = self.class_names[pred_class]
        
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Original image
        ax0 = plt.subplot2grid((3, 4), (0, 0))
        ax0.imshow(img_np)
        ax0.set_title('Original Image')
        ax0.axis('off')
        
        # 2. GradCAM
        gradcam_img, _, _ = self.get_gradcam(img_tensor, method='gradcam')
        ax1 = plt.subplot2grid((3, 4), (0, 1))
        ax1.imshow(gradcam_img)
        ax1.set_title('GradCAM')
        ax1.axis('off')
        
        # 3. GradCAM++
        gradcam_plus_img, _, _ = self.get_gradcam(img_tensor, method='gradcam++')
        ax2 = plt.subplot2grid((3, 4), (0, 2))
        ax2.imshow(gradcam_plus_img)
        ax2.set_title('GradCAM++')
        ax2.axis('off')
        
        # 4. Class probabilities
        ax3 = plt.subplot2grid((3, 4), (0, 3))
        y_pos = np.arange(len(self.class_names))
        ax3.barh(y_pos, probabilities)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(self.class_names)
        ax3.invert_yaxis()
        ax3.set_xlabel('Probability')
        ax3.set_title('Class Probabilities')
        
        # 5. LIME positive
        if True:  # Try-except for LIME
            try:
                lime_img, lime_img_neg, _, _, _ = self.get_lime_explanation(img_tensor)
                ax4 = plt.subplot2grid((3, 4), (1, 0))
                ax4.imshow(lime_img)
                ax4.set_title('LIME (Positive)')
                ax4.axis('off')
                
                # 6. LIME negative
                ax5 = plt.subplot2grid((3, 4), (1, 1))
                ax5.imshow(lime_img_neg)
                ax5.set_title('LIME (Negative)')
                ax5.axis('off')
            except Exception as e:
                print(f"LIME error: {e}")
                lime_error_text = f"LIME error: {str(e)[:50]}..."
                ax4 = plt.subplot2grid((3, 4), (1, 0))
                ax4.text(0.5, 0.5, lime_error_text, ha='center', va='center')
                ax4.axis('off')
                ax4.set_title('LIME (Error)')
                
                ax5 = plt.subplot2grid((3, 4), (1, 1))
                ax5.axis('off')
        
        # 7. Integrated Gradients
        try:
            ig_img, _, _, _, _ = self.get_integrated_gradients(img_tensor)
            ax6 = plt.subplot2grid((3, 4), (1, 2))
            ax6.imshow(ig_img)
            ax6.set_title('Integrated Gradients')
            ax6.axis('off')
        except Exception as e:
            print(f"IG error: {e}")
            ax6 = plt.subplot2grid((3, 4), (1, 2))
            ax6.text(0.5, 0.5, f"IG error: {str(e)[:50]}...", ha='center', va='center')
            ax6.axis('off')
            ax6.set_title('IG (Error)')
        
        # 8. Occlusion Sensitivity
        try:
            occ_img, _, _, _, _ = self.get_occlusion_map(img_tensor)
            ax7 = plt.subplot2grid((3, 4), (1, 3))
            ax7.imshow(occ_img)
            ax7.set_title('Occlusion Sensitivity')
            ax7.axis('off')
        except Exception as e:
            print(f"Occlusion error: {e}")
            ax7 = plt.subplot2grid((3, 4), (1, 3))
            ax7.text(0.5, 0.5, f"Occlusion error: {str(e)[:50]}...", ha='center', va='center')
            ax7.axis('off')
            ax7.set_title('Occlusion (Error)')
        
        # 9. SHAP (if background images are provided)
        if background_imgs is not None:
            try:
                shap_img, _, _, _, _ = self.get_shap_explanation(img_tensor, background_imgs)
                ax8 = plt.subplot2grid((3, 4), (2, 0))
                ax8.imshow(shap_img)
                ax8.set_title('SHAP')
                ax8.axis('off')
            except Exception as e:
                print(f"SHAP error: {e}")
                ax8 = plt.subplot2grid((3, 4), (2, 0))
                ax8.text(0.5, 0.5, f"SHAP error: {str(e)[:50]}...", ha='center', va='center')
                ax8.axis('off')
                ax8.set_title('SHAP (Error)')
        else:
            ax8 = plt.subplot2grid((3, 4), (2, 0))
            ax8.text(0.5, 0.5, "SHAP requires background images", ha='center', va='center')
            ax8.axis('off')
            ax8.set_title('SHAP (Not Available)')
        
        # 10. ScoreCAM
        try:
            scorecam_img, _, _ = self.get_gradcam(img_tensor, method='scorecam')
            ax9 = plt.subplot2grid((3, 4), (2, 1))
            ax9.imshow(scorecam_img)
            ax9.set_title('ScoreCAM')
            ax9.axis('off')
        except Exception as e:
            print(f"ScoreCAM error: {e}")
            ax9 = plt.subplot2grid((3, 4), (2, 1))
            ax9.text(0.5, 0.5, f"ScoreCAM error: {str(e)[:50]}...", ha='center', va='center')
            ax9.axis('off')
            ax9.set_title('ScoreCAM (Error)')
        
        # 11. XGradCAM
        try:
            xgradcam_img, _, _ = self.get_gradcam(img_tensor, method='xgradcam')
            ax10 = plt.subplot2grid((3, 4), (2, 2))
            ax10.imshow(xgradcam_img)
            ax10.set_title('XGradCAM')
            ax10.axis('off')
        except Exception as e:
            print(f"XGradCAM error: {e}")
            ax10 = plt.subplot2grid((3, 4), (2, 2))
            ax10.text(0.5, 0.5, f"XGradCAM error: {str(e)[:50]}...", ha='center', va='center')
            ax10.axis('off')
            ax10.set_title('XGradCAM (Error)')
        
        # 12. EigenCAM
        try:
            eigencam_img, _, _ = self.get_gradcam(img_tensor, method='eigencam')
            ax11 = plt.subplot2grid((3, 4), (2, 3))
            ax11.imshow(eigencam_img)
            ax11.set_title('EigenCAM')
            ax11.axis('off')
        except Exception as e:
            print(f"EigenCAM error: {e}")
            ax11 = plt.subplot2grid((3, 4), (2, 3))
            ax11.text(0.5, 0.5, f"EigenCAM error: {str(e)[:50]}...", ha='center', va='center')
            ax11.axis('off')
            ax11.set_title('EigenCAM (Error)')
        
        # Add title with prediction information
        fig.suptitle(f"Prediction: {class_name} (Confidence: {conf_value:.2f})", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        return fig
    
    def evaluate_explainability_metrics(self, test_loader, num_samples=50):
        """
        Evaluate various explainability metrics on test samples
        """
        results = []
        
        # Sample a subset of test images
        sampled_images = []
        sampled_labels = []
        count = 0
        
        for images, labels in test_loader:
            for i in range(len(images)):
                if count < num_samples:
                    sampled_images.append(images[i])
                    sampled_labels.append(labels[i].item())
                    count += 1
                else:
                    break
            if count >= num_samples:
                break
        
        # Get background images for SHAP
        background_imgs = sampled_images[:10]  # Use first 10 as background
        
        # Process each sampled image
        for idx, (img_tensor, true_label) in enumerate(zip(sampled_images, sampled_labels)):
            print(f"Processing image {idx+1}/{num_samples}...")
            
            # Get prediction
            pred_class, conf_value, _ = self.get_prediction(img_tensor)
            correct = (pred_class == true_label)
            
            # GradCAM consistency score (compare GradCAM and GradCAM++)
            try:
                _, gradcam_map, _ = self.get_gradcam(img_tensor, method='gradcam')
                _, gradcam_plus_map, _ = self.get_gradcam(img_tensor, method='gradcam++')
                
                # Flatten the maps
                gradcam_flat = gradcam_map.flatten()
                gradcam_plus_flat = gradcam_plus_map.flatten()
                
                # Calculate correlation
                gradcam_consistency = np.corrcoef(gradcam_flat, gradcam_plus_flat)[0, 1]
            except:
                gradcam_consistency = None
            
            # LIME stability (not fully implemented)
            lime_stability = None  # Would require multiple runs
            
            # Collect results
            result = {
                'image_idx': idx,
                'true_label': true_label,
                'true_class': self.class_names[true_label],
                'pred_label': pred_class,
                'pred_class': self.class_names[pred_class],
                'confidence': conf_value,
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
    
    def compare_target_vs_predicted_class(self, img_tensor, target_class=None):
        """
        Compare GradCAM visualizations for the target class vs predicted class
        """
        # Get prediction
        pred_class, conf_value, _ = self.get_prediction(img_tensor)
        
        # Set target class if not provided
        if target_class is None or target_class == pred_class:
            # Find second most likely class
            _, probabilities = self.model(img_tensor.unsqueeze(0).to(self.device)).max(1)
            probabilities = torch.nn.functional.softmax(probabilities, dim=1)[0].cpu().numpy()
            sorted_indices = np.argsort(probabilities)[::-1]
            target_class = sorted_indices[1]  # Second most likely class
        
        # Get GradCAM for predicted class
        gradcam_pred, _, _ = self.get_gradcam(img_tensor, target_class=pred_class)
        
        # Get GradCAM for target class
        gradcam_target, _, _ = self.get_gradcam(img_tensor, target_class=target_class)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img_np = self.preprocess_image_for_display(img_tensor)
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # GradCAM for predicted class
        axes[1].imshow(gradcam_pred)
        axes[1].set_title(f'GradCAM for Predicted Class\n({self.class_names[pred_class]})')
        axes[1].axis('off')
        
        # GradCAM for target class
        axes[2].imshow(gradcam_target)
        axes[2].set_title(f'GradCAM for Target Class\n({self.class_names[target_class]})')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig, pred_class, target_class
    
    def generate_explanation_report(self, img_tensor, background_imgs=None, save_path=None):
        """
        Generate a comprehensive explanation report for a single image
        """
        # Get prediction
        pred_class, conf_value, probabilities = self.get_prediction(img_tensor)
        class_name = self.class_names[pred_class]
        
        # Create multi-explanation visualization
        fig = self.create_multi_explanation_visualization(img_tensor, background_imgs)
        
        # Save figure if path is provided
        if save_path:
            fig.savefig(save_path)
        
        # Create text report
        report = {
            'Prediction': class_name,
            'Confidence': conf_value,
            'Top 3 Classes': [
                (self.class_names[i], probabilities[i]) 
                for i in np.argsort(probabilities)[::-1][:3]
            ]
        }
        
        return fig, report

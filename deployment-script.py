import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Import the explainability module
from explainability_module_tf import ExplainabilityTools

# Load model and class names
def load_model():
    # Load the model
    model = tf.keras.models.load_model('saved_model')
    
    # Load class names
    class_names = np.load('class_names.npy', allow_pickle=True)
    
    return model, class_names

# Process image
def preprocess_image(image_pil):
    # Resize image
    image_pil = image_pil.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(image_pil) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Get prediction from model
def get_prediction(model, img_array, class_names):
    # Get model predictions
    predictions = model.predict(img_array)
    
    # Get top predicted class
    pred_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[pred_class_idx]
    confidence = predictions[0][pred_class_idx]
    
    # Get top 3 predictions
    top3_idx = np.argsort(predictions[0])[::-1][:3]
    top3_classes = [(class_names[i], predictions[0][i]) for i in top3_idx]
    
    return predicted_class, confidence, top3_classes, pred_class_idx

# Function to convert matplotlib figure to HTML
def fig_to_html(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_str}" />'

# Gradio interface function with explainability
def classify_and_explain(input_image, explanation_method):
    # Load model and class names
    model, class_names = load_model()
    
    # Create explainability tools
    explainer = ExplainabilityTools(model, class_names)
    
    # Process image
    img_array = preprocess_image(input_image)
    
    # Get prediction
    predicted_class, confidence, top3_classes, pred_class_idx = get_prediction(model, img_array, class_names)
    
    # Generate explanation based on selected method
    if explanation_method == "GradCAM":
        _, cam_img, _ = explainer.get_gradcam(img_array)
        explanation_img = Image.fromarray(cam_img)
        
    elif explanation_method == "GradCAM++":
        _, cam_img, _ = explainer.get_gradcam_plus_plus(img_array)
        explanation_img = Image.fromarray(cam_img)
        
    elif explanation_method == "LIME":
        lime_img, _, _, _, _ = explainer.get_lime_explanation(img_array)
        # Convert from float [0,1] to uint8 [0,255]
        lime_img = (lime_img * 255).astype(np.uint8)
        explanation_img = Image.fromarray(lime_img)
        
    elif explanation_method == "Occlusion Sensitivity":
        _, sensitivity_map, _, _, _ = explainer.get_occlusion_sensitivity(img_array)
        explanation_img = Image.fromarray(sensitivity_map)
        
    elif explanation_method == "Multi-Explanation":
        # Create background images for SHAP (just duplicate the input image for simplicity)
        bg_images = [img_array[0]] * 10
        
        # Generate comprehensive explanation
        fig = explainer.create_multi_explanation_visualization(img_array, bg_images)
        explanation_html = fig_to_html(fig)
        
        # Format results
        result_html = f"""
        <div style='text-align: center;'>
            <h2>Prediction: {predicted_class} ({confidence:.2%})</h2>
            <h3>Top 3 Predictions:</h3>
            <ul style='list-style-type: none;'>
                <li>{top3_classes[0][0]}: {top3_classes[0][1]:.2%}</li>
                <li>{top3_classes[1][0]}: {top3_classes[1][1]:.2%}</li>
                <li>{top3_classes[2][0]}: {top3_classes[2][1]:.2%}</li>
            </ul>
        </div>
        {explanation_html}
        """
        
        return result_html
    
    else:
        # Default to GradCAM if method not recognized
        _, cam_img, _ = explainer.get_gradcam(img_array)
        explanation_img = Image.fromarray(cam_img)
    
    # Format results for image-based explanations
    results = (
        f"Prediction: {predicted_class} ({confidence:.2%})\n\n"
        f"Top 3 Predictions:\n"
        f"1. {top3_classes[0][0]}: {top3_classes[0][1]:.2%}\n"
        f"2. {top3_classes[1][0]}: {top3_classes[1][1]:.2%}\n"
        f"3. {top3_classes[2][0]}: {top3_classes[2][1]:.2%}"
    )
    
    return results, explanation_img

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Skin Disease Classifier with Explainability") as app:
        gr.Markdown("# Skin Disease Classifier with Explainability")
        gr.Markdown("Upload an image of a skin lesion to get a diagnosis and explanation.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                explanation_method = gr.Radio(
                    ["GradCAM", "GradCAM++", "LIME", "Occlusion Sensitivity", "Multi-Explanation"],
                    label="Explanation Method",
                    value="GradCAM"
                )
                submit_btn = gr.Button("Classify and Explain")
            
            with gr.Column():
                with gr.Tab("Standard Explanation"):
                    prediction_text = gr.Textbox(label="Prediction")
                    explanation_image = gr.Image(label="Explanation Visualization")
                
                with gr.Tab("Comprehensive Report"):
                    html_output = gr.HTML()
        
        # Handle the different outputs based on the selected method
        submit_btn.click(
            fn=lambda img, method: classify_and_explain(img, method) if method != "Multi-Explanation" 
                                  else (None, None, classify_and_explain(img, method)),
            inputs=[input_image, explanation_method],
            outputs=[prediction_text, explanation_image, html_output]
        )
        
        gr.Markdown("""
        ## Explanation Methods
        - **GradCAM**: Highlights regions that influenced the prediction
        - **GradCAM++**: An improved version of GradCAM with better localization
        - **LIME**: Explains predictions by perturbing the input
        - **Occlusion Sensitivity**: Systematically occludes parts of the image to find important regions
        - **Multi-Explanation**: Provides a comprehensive view with multiple explanation techniques
        
        ## About the Model
        This model was trained on the HAM10000 dataset to classify 7 different types of skin lesions:
        1. Actinic keratoses and intraepithelial carcinoma (akiec)
        2. Basal cell carcinoma (bcc)
        3. Benign keratosis-like lesions (bkl)
        4. Dermatofibroma (df)
        5. Melanoma (mel)
        6. Melanocytic nevi (nv)
        7. Vascular lesions (vasc)
        """)
    
    return app

# Main function
def main():
    # Create and launch the interface
    app = create_interface()
    app.launch(share=True)

if __name__ == "__main__":
    main()

import os
import numpy as np
import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Import the explainability module
from explainability_module import ExplainabilityTools

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model and class names
def load_model():
    # Load class names
    class_names = np.load('class_names.npy', allow_pickle=True)
    
    # Load model
    model = torch.jit.load('model_scripted.pt')
    model.eval()
    model.to(device)
    
    return model, class_names

# Process image and make prediction
def process_image(image_pil):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform image
    img_tensor = transform(image_pil)
    
    return img_tensor

# Get model prediction
def get_prediction(model, img_tensor, class_names):
    # Add batch dimension and move to device
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        
    # Get class name and probability
    predicted_class = class_names[predicted_class_idx]
    probability = probabilities[predicted_class_idx].item()
    
    # Get top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    top3_classes = [(class_names[idx.item()], prob.item()) for idx, prob in zip(top3_idx, top3_prob)]
    
    return predicted_class, probability, top3_classes, predicted_class_idx

# Function to convert matplotlib figure to HTML
def fig_to_html(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_str}" />'

# Gradio inference function with explainability
def classify_and_explain(input_image, explanation_method):
    # Load model and class names
    model, class_names = load_model()
    
    # Process image
    img_tensor = process_image(input_image)
    
    # Get prediction
    predicted_class, probability, top3_classes, class_idx = get_prediction(model, img_tensor, class_names)
    
    # Initialize explainability tools
    explainer = ExplainabilityTools(model, class_names, device)
    
    # Generate explanation based on selected method
    if explanation_method == "GradCAM":
        cam_img, _, _ = explainer.get_gradcam(img_tensor, method='gradcam')
        explanation_img = cam_img
        
    elif explanation_method == "GradCAM++":
        cam_img, _, _ = explainer.get_gradcam(img_tensor, method='gradcam++')
        explanation_img = cam_img
        
    elif explanation_method == "LIME":
        lime_img, _, _, _, _ = explainer.get_lime_explanation(img_tensor)
        explanation_img = lime_img
        
    elif explanation_method == "Integrated Gradients":
        ig_img, _, _, _, _ = explainer.get_integrated_gradients(img_tensor)
        explanation_img = ig_img
        
    elif explanation_method == "Multi-Explanation":
        # Get sample background images
        bg_tensor = img_tensor.unsqueeze(0).repeat(10, 1, 1, 1)
        bg_list = [bg_tensor[i] for i in range(10)]
        
        # Generate comprehensive explanation
        fig, _ = explainer.generate_explanation_report(img_tensor, bg_list)
        explanation_html = fig_to_html(fig)
        plt.close(fig)
        
        # Format results
        result_html = f"""
        <div style='text-align: center;'>
            <h2>Prediction: {predicted_class} ({probability:.2%})</h2>
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
        cam_img, _, _ = explainer.get_gradcam(img_tensor, method='gradcam')
        explanation_img = cam_img
    
    # Format results for image-based explanations
    results = (
        f"Prediction: {predicted_class} ({probability:.2%})\n\n"
        f"Top 3 Predictions:\n"
        f"1. {top3_classes[0][0]}: {top3_classes[0][1]:.2%}\n"
        f"2. {top3_classes[1][0]}: {top3_classes[1][1]:.2%}\n"
        f"3. {top3_classes[2][0]}: {top3_classes[2][1]:.2%}"
    )
    
    # Convert numpy array to PIL Image
    explanation_pil = Image.fromarray((explanation_img * 255).astype(np.uint8))
    
    return results, explanation_pil

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Skin Disease Classifier with Explainability") as app:
        gr.Markdown("# Skin Disease Classifier with Explainability")
        gr.Markdown("Upload an image of a skin lesion to get a diagnosis and explanation.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                explanation_method = gr.Radio(
                    ["GradCAM", "GradCAM++", "LIME", "Integrated Gradients", "Multi-Explanation"],
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
        - **Integrated Gradients**: Attributes importance to input features
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

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from flask import Flask, request, render_template, jsonify
import io
import base64
import traceback
import cv2
import time

#pip install google-cloud-aiplatform
from google.cloud import aiplatform
from google.oauth2 import service_account

# GCP Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "822141318653")
ENDPOINT_ID = os.environ.get("GCP_ENDPOINT_ID", "9172360194884108288")
LOCATION = "us-central1"
# Set environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/monilshah/Documents/GitHub/Medxplain/GCP/Keys/deductive-case-453813-c1-61bf5477abfa.json"

# Create credentials object
credentials = service_account.Credentials.from_service_account_file(
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
)

app = Flask(__name__)

# Constants
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable
model = None

# Class mapping
classes = {
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    1: ('bcc', 'Basal cell carcinoma'),
    2: ('bkl', 'Benign keratosis-like lesions'),
    3: ('df', 'Dermatofibroma'),
    4: ('nv', 'Melanocytic nevi'),
    5: ('vasc', 'Pyogenic granulomas and hemorrhage'),
    6: ('mel', 'Melanoma')
}

# At the top of your file, have both cloud endpoint and a minimal local model
model = None  # For explainability features only

def load_model_for_explainability():
    """Load model locally JUST for explainability purposes"""
    global model
    if model is None:
        try:
            model_path = 'best_model_64_VIT.keras'
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
                print("Local model loaded for explainability")
            else:
                print("Local model not found, explainability features will be limited")
        except Exception as e:
            print(f"Error loading local model: {e}")

def get_simplified_visualization(img_array, pred_class):
    """Generate a simplified visualization when model layers aren't available"""
    try:
        # Get the original image
        img = (img_array[0] * 255).astype(np.uint8)
        
        # Create a simple heatmap based on image features
        # This is not Grad-CAM but a simple visualization
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply some basic image processing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Create a heatmap
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        
        # Overlay on original image
        superimposed_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        
        # Convert to base64 for sending to frontend
        is_success, buffer = cv2.imencode(".jpg", superimposed_img)
        if is_success:
            io_buf = io.BytesIO(buffer)
            return base64.b64encode(io_buf.getvalue()).decode('utf-8')
        
        return None
    except Exception as e:
        print(f"Error generating simplified visualization: {e}")
        traceback.print_exc()
        return None
            
def preprocess_image(image, target_size=(64, 64)):
    """Preprocess the image for model prediction"""
    # Resize the image
    image = image.resize(target_size)
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_gradcam(img_array, pred_class):
    """Generate a simplified Grad-CAM visualization"""
    global model
    
    try:
        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            # If no Conv2D layer found, try to find another meaningful layer
            for layer in reversed(model.layers):
                if hasattr(layer, 'activation') and layer.activation is not None:
                    last_conv_layer = layer
                    break
        
        if last_conv_layer is None:
            print("Could not find suitable layer for Grad-CAM")
            return None
            
        print(f"Using layer {last_conv_layer.name} for Grad-CAM")
        
        # Create a model that outputs both the predictions and the activations
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Compute the gradient of the top predicted class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_channel = predictions[:, pred_class]
        
        # Gradient of the output neuron with respect to the conv layer output
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by importance
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs), 
            axis=-1
        )
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize to match the original image
        heatmap = cv2.resize(heatmap, (64, 64))
        
        # Convert to RGB heatmap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        img = (img_array[0] * 255).astype(np.uint8)
        superimposed_img = heatmap * 0.4 + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # Convert to base64 for sending to frontend
        is_success, buffer = cv2.imencode(".jpg", superimposed_img)
        if is_success:
            io_buf = io.BytesIO(buffer)
            return base64.b64encode(io_buf.getvalue()).decode('utf-8')
        
        return None
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        traceback.print_exc()
        return None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    # Note: We're not loading a local model anymore
    
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file part in the request'
        })
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'error': 'No file selected'
        })
    
    try:
        # Read and preprocess the image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Log image details
        print("Image format:", image.format)
        print("Image size:", image.size)
        print("Image mode:", image.mode)
        
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Save the uploaded image
        img_path = os.path.join(UPLOAD_FOLDER, 'temp.jpg')
        image.save(img_path)
        
        # Preprocess image
        img_array = preprocess_image(image)
        
        # Get prediction from GCP endpoint
        prediction_result = get_prediction(img_array)
        
        if prediction_result is None:
            return jsonify({'error': 'Error getting prediction from GCP endpoint'})
        
        # Extract class probabilities and predicted class
        probs = prediction_result['class_probabilities']
        pred_class = prediction_result['predicted_class']
        confidence = float(probs[pred_class] * 100)
        
        print("Predicted class:", pred_class)
        print("Class mapping:", classes[pred_class])
        
        # Generate visualization - use simplified version for cloud model
        if model is None:  # If local model isn't available
            gradcam_base64 = get_simplified_visualization(img_array, pred_class)
        else:  # If local model is available
            gradcam_base64 = get_gradcam(img_array, pred_class)
        
        # Get top 3 predictions for the chart
        top_k = 3
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_classes = [classes[idx][0] for idx in top_indices]
        top_full_names = [classes[idx][1] for idx in top_indices]
        top_probs = [float(probs[idx] * 100) for idx in top_indices]
        
        # Before returning the results JSON
        print("Top classes:", top_classes)
        print("Top full names:", top_full_names)
        print("Top probs:", top_probs)
        
        # Return results
        return jsonify({
            'success': True,
            'class_code': classes[pred_class][0],
            'class_name': classes[pred_class][1],
            'confidence': confidence,
            'gradcam': gradcam_base64,
            'top_classes': top_classes,
            'top_full_names': top_full_names,
            'top_probs': top_probs
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Error processing image: {str(e)}'
        })

def initialize_model_client():
    """Initialize the Vertex AI client"""
    aiplatform.init(project="822141318653")
    
    # Store endpoint as global variable
    global endpoint
    endpoint = aiplatform.Endpoint("9172360194884108288")
    print(f"Successfully connected to endpoint: {"9172360194884108288"}")

# Call this function during app startup
initialize_model_client()
    
def log_prediction_performance(start_time, response):
    """Log prediction performance metrics"""
    end_time = time.time()
    duration = end_time - start_time
    print(f"Prediction completed in {duration:.2f}s - Status: {response['success']}")

def format_instance_for_prediction(img_array):
    """Format image array for saved_model format in GCP"""
    # For a 64x64x3 input image, keeping the original 3D structure
    # Format 0: [[[0.5] * 3] * 64] * 64 - a 64x64x3 structure
    
    # Ensure the image has the right dimensions
    if img_array.shape[1:] != (64, 64, 3):
        # Resize if needed
        from skimage.transform import resize
        img = resize(img_array[0], (64, 64, 3), anti_aliasing=True)
    else:
        img = img_array[0]  # Just use the first image in the batch
    
    # Convert to nested list format (maintaining 3D structure)
    instance = img.tolist()
    
    # Return as a list of instances (batch of 1)
    return [instance]

def get_prediction(img_array):
    """Get prediction from GCP endpoint using saved_model format"""
    try:
        # Format data correctly for saved_model
        instances = format_instance_for_prediction(img_array)
        
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # Get the endpoint
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)
        
        # Make prediction with the structured format
        response = endpoint.predict(instances)
        
        # Process the response
        predictions = response.predictions[0]
        
        # Convert to expected format
        result = {
            'class_probabilities': predictions,
            'predicted_class': int(np.argmax(predictions))
        }
        
        return result
        
    except Exception as e:
        print(f"Error getting prediction from GCP endpoint: {e}")
        traceback.print_exc()
        return None

def test_simple_request():
    """Test a simple request to the endpoint"""
    from google.cloud import aiplatform
    
    try:
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # Get the endpoint
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)
        
        # Create a very basic test instance
        # This needs to match your model's expected input format
        test_instance = {"image": [[0.5] * 3] * 64} # Simple 1x64x3 array filled with 0.5
        
        # Print request being sent
        print(f"Sending test request to endpoint: {test_instance}")
        
        # Make prediction
        response = endpoint.predict([test_instance])
        
        print("Response received!")
        print(f"Prediction shape: {len(response.predictions)}")
        
        return True
    except Exception as e:
        print(f"Endpoint test failed: {e}")
        traceback.print_exc()
        return False

# Call the test function
#test_simple_request()


if __name__ == '__main__':
    #load_model()  # Load model at startup
    load_model_for_explainability()
    app.run(debug=True, host='0.0.0.0', port=3000, use_reloader=False)


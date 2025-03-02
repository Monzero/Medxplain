def simple_gradcam(model, img_array, target_class=None):
    """
    A reliable GradCAM implementation that avoids complex tracking issues
    
    Args:
        model: The trained TensorFlow model
        img_array: Input image array (1, height, width, channels)
        target_class: Target class index (if None, uses predicted class)
        
    Returns:
        original_img: Original image
        heatmap_img: Visualization with heatmap overlay
        heatmap: Raw heatmap for further processing
    """
    import tensorflow as tf
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    
    # Find a suitable convolutional layer
    conv_layer = None
    for layer in reversed(model.layers):
        # Check for Conv2D layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer = layer
            print(f"Using layer: {layer.name}")
            break
    
    # If no direct conv layer, look in nested layers
    if conv_layer is None:
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                for nested_layer in reversed(layer.layers):
                    if isinstance(nested_layer, tf.keras.layers.Conv2D):
                        conv_layer = nested_layer
                        print(f"Using nested layer: {nested_layer.name}")
                        break
                if conv_layer is not None:
                    break
    
    # If still no conv layer found, use a layer with 'conv' in the name
    if conv_layer is None:
        for layer in model.layers:
            if 'conv' in layer.name.lower():
                conv_layer = layer
                print(f"Falling back to layer: {layer.name}")
                break
    
    # If we still can't find a suitable layer, use a saliency map approach instead
    if conv_layer is None:
        print("No convolutional layer found. Using saliency map instead.")
        return saliency_map(model, img_array, target_class)
    
    # Create a model that outputs both the conv layer and the final predictions
    try:
        # Try creating an intermediate model
        feature_model = tf.keras.Model(inputs=model.inputs, outputs=conv_layer.output)
        
        # Get the feature maps and predictions
        feature_maps = feature_model.predict(img_array)
        predictions = model.predict(img_array)
        
        # Determine the target class
        if target_class is None:
            target_class = np.argmax(predictions[0])
        
        # Get input gradient directly
        input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = model(input_tensor)
            target_output = predictions[:, target_class]
        
        # Get gradients of input
        grads = tape.gradient(target_output, input_tensor)
        
        # Get gradients at the feature map level (approximate)
        # We're pooling gradients up to the feature map level
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        
        # Weight the feature maps with the gradient values
        weighted_features = feature_maps[0] * tf.reshape(pooled_grads[0], (1, 1, -1))
        
        # Average the weighted feature maps
        cam = np.mean(weighted_features, axis=-1)
        
    except Exception as e:
        print(f"Error in GradCAM: {e}")
        print("Falling back to direct feature visualization")
        return direct_feature_vis(model, img_array, conv_layer.name)
    
    # Ensure proper normalization
    cam = np.maximum(cam, 0)  # ReLU
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7)  # Normalize
    
    # Resize CAM to input image size
    cam_resized = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
    
    # Convert original image to uint8
    original_img = np.uint8(img_array[0] * 255)
    
    # Create heatmap
    heatmap_colored = np.uint8(255 * plt.cm.jet(cam_resized)[:, :, :3])
    
    # Superimpose heatmap on original image
    superimposed = heatmap_colored * 0.4 + original_img * 0.6
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return original_img, superimposed, cam_resized

def direct_feature_vis(model, img_array, layer_name):
    """
    Simple feature map visualization without gradients
    """
    import tensorflow as tf
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    
    # Try to get the layer
    try:
        # Create intermediate model to extract features
        feature_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
        feature_maps = feature_model.predict(img_array)
        
        # Average across all feature maps
        cam = np.mean(feature_maps[0], axis=-1)
        
        # Normalize
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7)
        
        # Resize
        cam_resized = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
        
    except Exception as e:
        print(f"Error in direct visualization: {e}")
        return saliency_map(model, img_array)
        
    # Original image
    original_img = np.uint8(img_array[0] * 255)
    
    # Create heatmap
    heatmap_colored = np.uint8(255 * plt.cm.jet(cam_resized)[:, :, :3])
    
    # Superimpose
    superimposed = heatmap_colored * 0.4 + original_img * 0.6
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return original_img, superimposed, cam_resized

def saliency_map(model, img_array, target_class=None):
    """
    Fallback to a simple saliency map approach
    """
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Get model prediction
    predictions = model.predict(img_array)
    if target_class is None:
        target_class = np.argmax(predictions[0])
    
    # Convert to tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # Get gradients
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_output = predictions[:, target_class]
    
    # Calculate gradients
    grads = tape.gradient(target_output, img_tensor)
    
    # Take absolute value and max across color channels
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]
    
    # Normalize
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-7)
    
    # Original image
    original_img = np.uint8(img_array[0] * 255)
    
    # Create heatmap
    heatmap_colored = np.uint8(255 * plt.cm.jet(saliency)[:, :, :3])
    
    # Superimpose
    superimposed = heatmap_colored * 0.4 + original_img * 0.6
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return original_img, superimposed, saliency

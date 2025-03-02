def get_gradcam(model, img_array, target_class=None):
    """
    A more robust implementation of GradCAM for TensorFlow models
    
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
    from PIL import Image
    
    # Convert image to tensor if it's a numpy array
    if isinstance(img_array, np.ndarray):
        img_tensor = tf.convert_to_tensor(img_array)
    else:
        img_tensor = img_array
    
    # Find a suitable convolutional layer
    conv_layer = None
    for layer in reversed(model.layers):
        # Check for direct Conv2D layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer = layer
            print(f"Using layer: {layer.name}")
            break
            
    # If no conv layer found in main model, try to find in nested models
    if conv_layer is None:
        for layer in model.layers:
            if hasattr(layer, 'layers'):  # Check if it's a container
                for nested_layer in reversed(layer.layers):
                    if isinstance(nested_layer, tf.keras.layers.Conv2D):
                        conv_layer = nested_layer
                        print(f"Using nested layer: {nested_layer.name} from {layer.name}")
                        break
                if conv_layer is not None:
                    break
    
    # If still no layer found, try to find a layer with 'conv' in its name
    if conv_layer is None:
        for layer in model.layers:
            if 'conv' in layer.name.lower():
                conv_layer = layer
                print(f"Falling back to layer with 'conv' in name: {layer.name}")
                break
                
    if conv_layer is None:
        raise ValueError("Could not find any convolutional layer in the model")
        
    # Create a simplified model that goes from input to the conv layer
    conv_output = conv_layer.output
    
    # Get predictions
    with tf.GradientTape() as tape:
        # Get conv layer output
        conv_output_value = None
        inputs = img_tensor
        
        # Track the conv layer output
        for layer in model.layers:
            if layer == conv_layer:
                conv_output_value = layer(inputs)
            else:
                try:
                    inputs = layer(inputs)
                except:
                    # Skip layers that don't connect properly
                    pass
        
        # Compute the actual prediction 
        preds = model(img_tensor, training=False)
        
        # Get the predicted class if target_class is None
        if target_class is None:
            target_class = tf.argmax(preds[0])
        
        # Get the predicted class values
        target_class_channel = preds[:, target_class]
    
    # Gradient of the predicted class with respect to the output feature map
    # This is the core of GradCAM
    grads = tape.gradient(target_class_channel, conv_output_value)
    
    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply pooled gradients with the output of the last conv layer
    conv_output_value = conv_output_value.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    # For numerical stability
    pooled_grads = pooled_grads / (np.std(pooled_grads) + 1e-8)
    
    # Weight the channels by importance (element-wise multiplication)
    for i in range(pooled_grads.shape[-1]):
        conv_output_value[:, :, i] *= pooled_grads[i]
    
    # Average over all channels
    heatmap = np.mean(conv_output_value, axis=-1)
    
    # ReLU the heatmap (only positive influence)
    heatmap = np.maximum(heatmap, 0) 
    
    # Normalize for visualization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Resize the heatmap to original image size
    img = img_array[0]
    height, width = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # Apply colormap
    heatmap_colored = np.uint8(255 * plt.cm.jet(heatmap_resized))[:, :, :3]
    
    # Convert original image to uint8
    original_img = np.uint8(img * 255)
    
    # Superimpose the heatmap
    superimposed = heatmap_colored * 0.4 + original_img * 0.6
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return original_img, superimposed, heatmap_resized

# Alternative implementation using eager execution
def get_gradcam_eager(model, img_array, target_class=None, last_conv_layer_name=None):
    """
    Alternative GradCAM implementation using eager execution
    
    Args:
        model: The trained TensorFlow model
        img_array: Input image array (1, height, width, channels)
        target_class: Target class index (if None, uses predicted class)
        last_conv_layer_name: Name of the conv layer to use (if None, will try to find one)
        
    Returns:
        original_img: Original image
        heatmap_img: Visualization with heatmap overlay
        heatmap: Raw heatmap for further processing
    """
    import tensorflow as tf
    import numpy as np
    import cv2
    
    # Find the last conv layer if not specified
    if last_conv_layer_name is None:
        # Try to find convolutional layers
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                print(f"Using layer: {last_conv_layer_name}")
                break
                
        # If not found, look in submodels
        if last_conv_layer_name is None:
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    for sublayer in reversed(layer.layers):
                        if isinstance(sublayer, tf.keras.layers.Conv2D):
                            last_conv_layer_name = sublayer.name
                            print(f"Using nested layer: {last_conv_layer_name}")
                            break
                    if last_conv_layer_name is not None:
                        break
                        
    # Run prediction to get class if not provided
    if target_class is None:
        preds = model.predict(img_array)
        target_class = np.argmax(preds[0])
        
    # Get the output of the last conv layer
    last_conv_layer = None
    
    # Try to find the layer in the main model
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # Try to find in nested models
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                try:
                    last_conv_layer = layer.get_layer(last_conv_layer_name)
                    break
                except ValueError:
                    continue
    
    if last_conv_layer is None:
        raise ValueError(f"Could not find layer: {last_conv_layer_name}")
        
    # Create a model that returns the last conv layer output
    last_conv_layer_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )
    
    # Define a function to compute the gradient of the target class with respect to the conv output
    @tf.function
    def compute_gradients(img_input, target_class_idx):
        with tf.GradientTape() as tape:
            # Get conv layer output
            conv_output = last_conv_layer_model(img_input)
            tape.watch(conv_output)
            
            # Apply remaining layers to get prediction
            # Find next layer index after conv layer
            found_layer = False
            output = conv_output
            
            # Manually apply remaining layers
            for layer in model.layers:
                if found_layer:
                    try:
                        output = layer(output)
                    except:
                        pass
                elif layer.name == last_conv_layer.name or (hasattr(layer, 'layers') and any(l.name == last_conv_layer.name for l in layer.layers if hasattr(l, 'name'))):
                    found_layer = True
            
            # Just use model.predict if manual application fails
            if not found_layer:
                output = model(img_input)
                
            # Get the target class score
            target_output = output[:, target_class_idx]
            
        # Compute gradients
        grads = tape.gradient(target_output, conv_output)
        return conv_output, grads
    
    # Get the last conv layer output and gradients
    try:
        conv_output, grads = compute_gradients(img_array, target_class)
    except:
        # Fallback - full eager execution without graph
        print("Falling back to non-graph execution")
        with tf.GradientTape() as tape:
            # Cast input to tensor
            inputs = tf.cast(img_array, tf.float32)
            tape.watch(inputs)
            
            # Get model prediction
            preds = model(inputs)
            target_output = preds[:, target_class]
            
        # Get gradients of input
        input_grads = tape.gradient(target_output, inputs)
        
        # Calculate crude saliency map
        saliency = tf.reduce_sum(tf.abs(input_grads), axis=-1)
        saliency = saliency.numpy()[0]
        
        # Resize and normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Original image
        original_img = np.uint8(img_array[0] * 255)
        
        # Create heatmap
        heatmap_colored = np.uint8(255 * plt.cm.jet(saliency))[:, :, :3]
        
        # Superimpose
        superimposed = heatmap_colored * 0.4 + original_img * 0.6
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        return original_img, superimposed, saliency
    
    # Calculate weighted feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_output = conv_output.numpy()[0]
    
    # Weight feature maps
    for i in range(pooled_grads.shape[0]):
        conv_output[:, :, i] *= pooled_grads[i]
        
    # Average over channels
    heatmap = np.mean(conv_output, axis=-1)
    
    # ReLU
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Resize
    img = img_array[0]
    height, width = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # Create colored heatmap
    heatmap_colored = np.uint8(255 * plt.cm.jet(heatmap_resized))[:, :, :3]
    
    # Original image
    original_img = np.uint8(img * 255)
    
    # Superimpose
    superimposed = heatmap_colored * 0.4 + original_img * 0.6
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return original_img, superimposed, heatmap_resized

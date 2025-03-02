def predict_all_samples(model, test_df, batch_size=32, img_size=(224, 224)):
    """
    A more reliable method to predict all samples without using a generator.
    This loads and processes images one by one, ensuring predictions for all samples.
    
    Args:
        model: Trained TensorFlow model
        test_df: DataFrame containing test data with 'path' column
        batch_size: Batch size for prediction
        img_size: Image size expected by the model
        
    Returns:
        predictions: Array of model predictions for all test samples
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    
    # Initialize array to store predictions
    num_samples = len(test_df)
    num_classes = model.output_shape[1]
    all_predictions = np.zeros((num_samples, num_classes))
    
    # Process images in batches
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_paths = test_df['path'].iloc[i:end_idx].values
        
        # Load and preprocess the batch
        batch_images = []
        for path in batch_paths:
            img = load_img(path, target_size=img_size)
            img_array = img_to_array(img)
            img_array = img_array / 255.0  # Normalize
            batch_images.append(img_array)
        
        # Convert to array and predict
        batch_array = np.array(batch_images)
        batch_predictions = model.predict(batch_array, verbose=0)
        
        # Store predictions
        all_predictions[i:end_idx] = batch_predictions
        
        # Print progress
        if (i // batch_size) % 10 == 0:
            print(f"Processed {end_idx}/{num_samples} images...")
    
    print(f"Prediction complete. Final shape: {all_predictions.shape}")
    return all_predictions

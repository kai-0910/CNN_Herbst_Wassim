import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess_image(image, label, target_size=(96, 96)):
    """
    Standardizes image size, normalizes pixel values, and one-hot encodes labels.
    """
    # Standardize image size: Ensure all images are explicitly resized
    image = tf.image.resize(image, target_size)
    
    # Normalize values: Cast to float32 and scale to [0, 1] range
    image = tf.cast(image, tf.float32) / 255.0
    
    # Transform labels to categorical: STL-10 has 10 classes
    label = tf.one_hot(label, depth=10)
    
    return image, label

def load_stl10_data(val_split_ratio=0.2, batch_size=32):
    """
    Loads STL-10, creates a validation split, applies preprocessing, 
    and returns batched tf.data.Dataset objects.
    """
    print("Downloading and loading STL-10 dataset...")
    # Load the dataset from tensorflow_datasets
    dataset, info = tfds.load('stl10', with_info=True, as_supervised=True)
    
    train_full = dataset['train']
    test_dataset = dataset['test']
    
    # The training set has 5,000 images. Let's calculate the split.
    num_train_full = info.splits['train'].num_examples
    num_val = int(num_train_full * val_split_ratio)
    
    # Create a validation dataset from the training dataset
    # Shuffle the training data before splitting to ensure random distribution
    train_full = train_full.shuffle(buffer_size=num_train_full, seed=42)
    
    val_dataset = train_full.take(num_val)
    train_dataset = train_full.skip(num_val)
    
    print(f"Dataset split: {num_train_full - num_val} training, {num_val} validation, {info.splits['test'].num_examples} testing.")

    # Apply the preprocessing function to all datasets
    # AUTOTUNE dynamically adjusts the number of parallel calls based on available CPU
    train_dataset = train_dataset.map(lambda x, y: preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y: preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(lambda x, y: preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the datasets and prefetch them to optimize GPU memory loading
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print("Data preprocessing complete!")
    return train_dataset, val_dataset, test_dataset
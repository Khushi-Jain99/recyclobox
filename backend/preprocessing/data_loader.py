import os
import tensorflow as tf
from utils.config_helper import load_config

def get_data_augmentation_pipeline():
    """
    Defines a sequence of Keras layers for image data augmentation.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    return data_augmentation

def load_datasets(config_path="backend/config/config.yaml"):
    """
    Loads train, validation, and test datasets from the processed directory.
    """
    config = load_config(config_path)
    
    processed_dir = config['dataset']['processed_dir']
    img_height = config['model']['input_shape'][0]
    img_width = config['model']['input_shape'][1]
    batch_size = config['model']['batch_size']
    seed = config['dataset']['random_seed']
    
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=config['dataset']['classes'],
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=seed
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=config['dataset']['classes'],
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=False,
        seed=seed
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=config['dataset']['classes'],
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=False,
        seed=seed
    )
    
    class_names = train_ds.class_names
    
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()
    test_ds = test_ds.cache()
    
    data_augmentation = get_data_augmentation_pipeline()
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names

import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=7):
    """
    Builds a custom lightweight Convolutional Neural Network (CNN).
    """
    model = models.Sequential([
        # Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Dense Head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_mobilenet_transfer(input_shape=(224, 224, 3), num_classes=7, trainable_base_layers=0):
    """
    Builds a Transfer Learning model using pre-trained MobileNetV2.
    
    Args:
        input_shape (tuple): Dimension of input image.
        num_classes (int): Number of target segregation categories.
        trainable_base_layers (int): Number of layers in MobileNetV2 base to unfreeze.
                                     0 means fully frozen (feature extraction mode).
    """
    # Load MobileNetV2 base weights pre-trained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base layers (Feature Extraction)
    base_model.trainable = False
    
    # Unfreeze specific top layers of MobileNetV2 if fine-tuning is enabled
    if trainable_base_layers > 0:
        base_model.trainable = True
        # Freeze all layers except the last n layers
        for layer in base_model.layers[:-trainable_base_layers]:
            layer.trainable = False
            
    # Construct model pipeline
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    # Test models compilation and print summaries
    cnn = build_custom_cnn()
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("=== Custom CNN Summary ===")
    cnn.summary()
    
    mobilenet = build_mobilenet_transfer()
    mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n=== MobileNetV2 Transfer Learning Summary ===")
    mobilenet.summary()

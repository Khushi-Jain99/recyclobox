import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_datasets
from model import build_mobilenet_transfer

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def plot_history(history, fine_tune_history=None, save_path="models/training_curves.png"):
    """
    Plots training and validation accuracy/loss over epochs, showing fine-tuning transition if available.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    if fine_tune_history:
        acc += fine_tune_history.history['accuracy']
        val_acc += fine_tune_history.history['val_accuracy']
        loss += fine_tune_history.history['loss']
        val_loss += fine_tune_history.history['val_loss']
        
    initial_epochs = len(history.history['accuracy'])
    
    plt.figure(figsize=(12, 6))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    if fine_tune_history:
        plt.plot([initial_epochs-1, initial_epochs-1], 
                 plt.ylim(), label='Start Fine Tuning', linestyle='--', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    if fine_tune_history:
        plt.plot([initial_epochs-1, initial_epochs-1], 
                 plt.ylim(), label='Start Fine Tuning', linestyle='--', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves plot to {save_path}")
    plt.close()

def main():
    # Detect GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found GPU: {gpus[0].name}. Using GPU for acceleration.")
        # Enable dynamic memory allocation to prevent TensorFlow from seizing all VRAM
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. Running on CPU (this might be slow).")
        
    config = load_config()
    
    input_shape = tuple(config['model']['input_shape'])
    num_classes = len(config['dataset']['classes'])
    epochs = config['model']['epochs']
    lr = config['model']['learning_rate']
    weights_path = config['model']['weights_path']
    checkpoint_dir = config['model']['checkpoint_dir']
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    print("\n--- Phase 1: Loading Datasets ---")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    
    print("\n--- Phase 2: Building and Compiling Feature Extractor ---")
    model = build_mobilenet_transfer(input_shape=input_shape, num_classes=num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model initialized. Trainable variables: {len(model.trainable_variables)}")
    
    # Configure Callbacks
    checkpoint_path = os.path.join(checkpoint_dir, "cp_warmup.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=config['training']['reduce_lr_patience'],
        min_lr=1e-6,
        verbose=1
    )
    
    print("\n--- Phase 3: Warming Up Model Head (Feature Extraction) ---")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[cp_callback, early_stop, reduce_lr]
    )
    
    # Check if we should fine-tune
    fine_tune_epochs = config['model']['fine_tune_epochs']
    fine_tune_history = None
    
    if fine_tune_epochs > 0:
        print("\n--- Phase 4: Fine-Tuning Top MobileNetV2 Layers ---")
        # Load the warm-up best weights
        model.load_weights(checkpoint_path)
        
        # Build transfer model with trainable top layers
        # Unfreeze last 20 layers of the base model
        model = build_mobilenet_transfer(input_shape=input_shape, num_classes=num_classes, trainable_base_layers=20)
        model.load_weights(checkpoint_path)
        
        # Recompile with low learning rate
        fine_tune_lr = config['model']['fine_tune_learning_rate']
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model re-compiled for fine-tuning. Trainable variables: {len(model.trainable_variables)}")
        
        checkpoint_path_ft = os.path.join(checkpoint_dir, "cp_finetune.ckpt")
        cp_callback_ft = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path_ft,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
        
        # Run fine tuning training
        fine_tune_history = model.fit(
            train_ds,
            epochs=fine_tune_epochs,
            validation_data=val_ds,
            callbacks=[cp_callback_ft, early_stop, reduce_lr]
        )
        
        # Load best weights from fine tuning
        model.load_weights(checkpoint_path_ft)
        
    # Save the final model in standard Keras format
    model.save(weights_path)
    print(f"Saved final trained model to {weights_path}")
    
    # Save plots
    plot_history(history, fine_tune_history)
    
    print("\n--- Phase 5: Evaluating Model on Test Dataset ---")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Get true and predicted labels for confusion matrix / report
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()

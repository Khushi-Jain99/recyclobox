import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory of training folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.data_loader import load_datasets
from models.model import build_mobilenet_transfer

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def plot_history(history, fine_tune_history=None, save_path="saved_models/training_curves.png"):
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
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found GPU: {gpus[0].name}. Using GPU for acceleration.")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. Running on CPU.")
        
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
    
    fine_tune_epochs = config['model']['fine_tune_epochs']
    fine_tune_history = None
    
    if fine_tune_epochs > 0:
        print("\n--- Phase 4: Fine-Tuning Top MobileNetV2 Layers ---")
        model.load_weights(checkpoint_path)
        
        model = build_mobilenet_transfer(input_shape=input_shape, num_classes=num_classes, trainable_base_layers=20)
        model.load_weights(checkpoint_path)
        
        fine_tune_lr = config['model']['fine_tune_learning_rate']
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        checkpoint_path_ft = os.path.join(checkpoint_dir, "cp_finetune.ckpt")
        cp_callback_ft = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path_ft,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
        
        fine_tune_history = model.fit(
            train_ds,
            epochs=fine_tune_epochs,
            validation_data=val_ds,
            callbacks=[cp_callback_ft, early_stop, reduce_lr]
        )
        
        model.load_weights(checkpoint_path_ft)
        
    model.save(weights_path)
    print(f"Saved final trained model to {weights_path}")
    
    plot_history(history, fine_tune_history)
    
    print("\n--- Phase 5: Evaluating Model on Test Dataset ---")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
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

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, f1_score, classification_report

def main():
    # 1. Use raw strings (r"...") for Windows paths to avoid escape character errors
    # 2. Point to the actual .keras model file, not the Jupyter Notebook file
    model_path = r"C:\Users\user\OneDrive\Desktop\MLOps\models\best_paddy_model.keras" 
    
    # 3. Define the directory where your labeled validation images are stored
    labeled_dir = r"C:\Users\user\OneDrive\Desktop\MLOps\data\Labeled_data"
    
    # 4. Define where you want the metrics.json file to be saved
    output_dir = r"C:\Users\user\OneDrive\Desktop\MLOps\output"
    
    os.makedirs(output_dir, exist_ok=True)

    img_size = (224, 224)
    batch_size = 32

    # Load validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        labeled_dir, validation_split=0.2, subset="validation",
        seed=42, image_size=img_size, batch_size=batch_size
    )
    
    class_names = val_ds.class_names
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Load model and predict
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    y_pred_probs = model.predict(val_ds, verbose=1).flatten()
    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0).flatten()

    # Calculate best decision threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])

    # Export metrics
    metrics = {
        "best_threshold": best_threshold,
        "f1_score": float(f1_scores[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx])
    }

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(classification_report(y_true, (y_pred_probs > best_threshold).astype(int), target_names=class_names))

if __name__ == "__main__":
    main()
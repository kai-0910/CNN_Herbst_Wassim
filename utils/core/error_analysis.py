from utils.plotting.helpers import plot_confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix

def analyze_model_errors(model, dataset, class_names, num_mismatches=20):
    """
    Evaluates a model on a dataset, prints the worst class mismatches,
    ranks the classes by accuracy, and plots a confusion matrix.
    """
    print("Extracting perfectly aligned predictions and true labels...")
    
    y_pred_probs = []
    y_true = []

    # Loop through the dataset EXACTLY once to guarantee alignment
    for images, labels in dataset:
        batch_preds = model.predict(images, verbose=0)
        y_pred_probs.extend(batch_preds)
        y_true.extend(labels.numpy())

    # Convert to clean numpy arrays
    y_pred_probs = np.array(y_pred_probs)
    y_true = np.array(y_true)

    print("\n" + "="*40)
    print(" AUTOMATED ERROR ANALYSIS")
    print("="*40)

    # Convert probabilities/one-hot to simple class indices
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_classes = np.argmax(y_true, axis=1)
    else:
        y_true_classes = y_true 
        
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Calculate raw confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # 1. Find and sort the worst mismatches
    mismatches = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0: 
                mismatches.append((class_names[i], class_names[j], cm[i, j]))

    mismatches.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {num_mismatches} Worst Mismatches (True Class -> Predicted As):")
    for true_name, pred_name, count in mismatches[:num_mismatches]:
        print(f" - {true_name.capitalize():<10} mistaken for {pred_name.capitalize():<10} : {count} times")

    # 2. Calculate and rank class accuracies
    class_accuracies = []
    for i in range(len(class_names)):
        total_samples = np.sum(cm[i, :]) 
        if total_samples > 0:
            correct_predictions = cm[i, i]   
            accuracy = (correct_predictions / total_samples) * 100
            class_accuracies.append((class_names[i], accuracy))

    class_accuracies.sort(key=lambda x: x[1], reverse=True)

    print("\nClass Accuracy Ranking (Best to Worst):")
    for rank, (class_name, accuracy) in enumerate(class_accuracies, 1):
        print(f" {rank:>2}. {class_name.capitalize():<10} : {accuracy:.1f}%")

    # 3. Plot the visual matrix 
    # (Assumes plot_confusion_matrix is available in the same file/environment)
    plot_confusion_matrix(y_true, y_pred_probs, class_names)
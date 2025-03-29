import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_metrics(metrics_dict, model_name, save_path):
    """
    Plot accuracy and loss graphs for a model
    """
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(metrics_dict['train_acc'], label='Training Accuracy')
    plt.plot(metrics_dict['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()


    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(metrics_dict['train_loss'], label='Training Loss')
    plt.plot(metrics_dict['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


    # Save plot
    plt.tight_layout()
    save_file = os.path.join(save_path, f'{model_name.lower().replace(" ", "_")}_metrics.png')
    plt.savefig(save_file)
    plt.close()
    print(f"Saved metrics plot for {model_name} at: {save_file}")

def generate_sample_metrics():
    """
    Generate sample training metrics for demonstration
    """
    epochs = 50
    x = np.linspace(0, epochs, epochs)
    
    # Detection Model (Mask R-CNN) metrics
    detection_metrics = {
        'train_acc': 0.65 + 0.24 * (1 - np.exp(-x/20)) + np.random.normal(0, 0.02, epochs),
        'val_acc': 0.60 + 0.29 * (1 - np.exp(-x/20)) + np.random.normal(0, 0.02, epochs),
        'train_loss': 0.8 * np.exp(-x/15) + 0.2 + np.random.normal(0, 0.02, epochs),
        'val_loss': 0.9 * np.exp(-x/15) + 0.2 + np.random.normal(0, 0.02, epochs)
    }

    # Classification Model (ResNet-50) metrics
    classification_metrics = {
        'train_acc': 0.70 + 0.22 * (1 - np.exp(-x/15)) + np.random.normal(0, 0.02, epochs),
        'val_acc': 0.65 + 0.27 * (1 - np.exp(-x/15)) + np.random.normal(0, 0.02, epochs),
        'train_loss': 0.7 * np.exp(-x/10) + 0.15 + np.random.normal(0, 0.02, epochs),
        'val_loss': 0.8 * np.exp(-x/10) + 0.15 + np.random.normal(0, 0.02, epochs)
    }

    # Recommendation Model (BERT) metrics
    recommendation_metrics = {
        'train_acc': 0.75 + 0.15 * (1 - np.exp(-x/10)) + np.random.normal(0, 0.02, epochs),
        'val_acc': 0.72 + 0.18 * (1 - np.exp(-x/10)) + np.random.normal(0, 0.02, epochs),
        'train_loss': 0.6 * np.exp(-x/8) + 0.1 + np.random.normal(0, 0.02, epochs),
        'val_loss': 0.7 * np.exp(-x/8) + 0.1 + np.random.normal(0, 0.02, epochs)
    }

    return detection_metrics, classification_metrics, recommendation_metrics

def main():
    # Set save path
    save_path = r"C:\Users\prasa\OneDrive\Desktop\New folder"
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate metrics
    detection_metrics, classification_metrics, recommendation_metrics = generate_sample_metrics()
    
    # Plot and save metrics for each model
    plot_metrics(detection_metrics, "Mask R-CNN Detection", save_path)
    plot_metrics(classification_metrics, "ResNet-50 Classification", save_path)
    plot_metrics(recommendation_metrics, "BERT Recommendation", save_path)
    
    # Generate combined performance summary
    plt.figure(figsize=(12, 6))
    models = ['Detection', 'Classification', 'Recommendation']
    final_accuracies = [
        detection_metrics['val_acc'][-1],
        classification_metrics['val_acc'][-1],
        recommendation_metrics['val_acc'][-1]
    ]
    
    plt.bar(models, final_accuracies, color=['blue', 'green', 'red'])
    plt.title('Final Model Accuracies')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(final_accuracies):
        plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'combined_performance.png'))
    plt.close()
    
    # Save metrics summary
    summary_file = os.path.join(save_path, 'metrics_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Model Performance Summary\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for model, metrics in zip(
            ['Detection (Mask R-CNN)', 'Classification (ResNet-50)', 'Recommendation (BERT)'],
            [detection_metrics, classification_metrics, recommendation_metrics]
        ):
            f.write(f"{model}:\n")
            f.write(f"Final Training Accuracy: {metrics['train_acc'][-1]:.2%}\n")
            f.write(f"Final Validation Accuracy: {metrics['val_acc'][-1]:.2%}\n")
            f.write(f"Final Training Loss: {metrics['train_loss'][-1]:.4f}\n")
            f.write(f"Final Validation Loss: {metrics['val_loss'][-1]:.4f}\n\n")

if __name__ == "__main__":
    main()
    print("All metrics plots and summary have been generated successfully!")
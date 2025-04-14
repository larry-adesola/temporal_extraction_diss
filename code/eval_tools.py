from sklearn.metrics import classification_report
import torch
import numpy as np
import matplotlib.pyplot as plt

def calc_f1(predicted_labels, all_labels):
    confusion = np.zeros((4, 4))
    for i in range(len(predicted_labels)):
        confusion[all_labels[i]][predicted_labels[i]] += 1

    acc = 1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion)
    true_positive = 0
    for i in range(4-1):
        true_positive += confusion[i][i]
    prec = true_positive/(np.sum(confusion)-np.sum(confusion,axis=0)[-1])
    rec = true_positive/(np.sum(confusion)-np.sum(confusion[-1][:]))
    f1 = 2*prec*rec / (rec+prec)

    return acc, prec, rec, f1, confusion

def evaluate_model(model, dataloader, device, pos_enabled):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
          if pos_enabled:
            input_ids, attention_mask, event_ix, labels, pos_id = (item.to(device) for item in batch)
            logits = model(input_ids, attention_mask, event_ix, pos_ids=pos_id)
          else:
            input_ids, attention_mask, event_ix, labels = (item.to(device) for item in batch)
            logits = model(input_ids, attention_mask, event_ix)

          predictions = torch.argmax(logits, dim=1)

          all_labels.extend(labels.cpu().numpy())
          all_predictions.extend(predictions.cpu().numpy())

    print(classification_report(all_labels, all_predictions, target_names=["BEFORE", "AFTER", "EQUAL", "VAGUE"]))

    acc, prec, rec, f1, confusion = calc_f1(all_predictions, all_labels)
    
    print(f"Acc={acc}, Precision={prec}, Recall={rec}, F1={f1}")
    print(f"Confusion={confusion}")

def plot_training_curves(train_losses, train_accuracies):
    """Simple function to plot training loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'r-o', label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.ylim([0, 1])
    plt.legend()

    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/diss/graph.png')
    #plt.show()
    
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import preprocessor




def evaluate_model(model, device):
    model.eval()

    labelArr = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in preprocessor.dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            prediction = torch.argmax(outputs, dim=1)
            labelArr.extend(labels.cpu().numpy())
            predictions.extend(prediction.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(labelArr, predictions)
    print(f'Accuracy: {accuracy:.4f}')
    # Print classification report
    print(classification_report(labelArr, predictions, target_names=preprocessor.image_datasets['train'].classes))
import torch
import sklearn.metrics as mtrcs

import learner # this causes learner script to run

def evaluator(): # defining this as function so it doesn't necessarily run if file imported elsewhere
    for i in range (0, learner.num_epochs):
        loadedTestSet = learner.preprocessor.dataloaders['test']
        thisEpoch = torch.load('emotion_detector_epoch_' + str(i + 1) + '.pth')
        learner.model.load_state_dict(thisEpoch)
        learner.model.eval()

        gcpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        learner.model.to(gcpu)
        labelArr, predictions = [], []

        with torch.no_grad(): 
            for inputs, labels in loadedTestSet:
                inputs, labels = inputs.to(gcpu), labels.to(gcpu)
                outputs = learner.model(inputs)
                _, prediction = torch.max(outputs, 1)
                labelArr.extend(labels.cpu().numpy())
                predictions.extend(prediction.cpu().numpy())

        accuracyScore = mtrcs.accuracy_score(labels, predictions)
        #precisionScore = mtrcs.precision_score(labels, predictions, average = 'weighted') # precision, as opposed to accuracy??

        print("Model" + i + "accuracy:", accuracyScore)

evaluator() # calling here for now
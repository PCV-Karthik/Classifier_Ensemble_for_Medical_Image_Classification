import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def performance(test_generator, model, fusion_function=None):
    # Predict classes for test set
    y_true = test_generator.classes
    y_pred = []
    if fusion_function is None:
        y_pred_probs = model.predict(test_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        for i in range(len(test_generator)):
            pred = fusion_function(test_generator[i][0], model)
            y_pred.append(pred)
            # ans = "True Value: " + str(np.argmax(test_generator[i][1])) + " Alexnet Value: " + str( np.argmax(
            # alexnetPred)) + " Googlenet Value: " + str(np.argmax(googlenetPred)), " Fusion Value: " + str( pred) +
            # " Predicted Model Name: " + str(name) final.append(ans)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy * 100)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Visualize confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='magma', xticklabels=test_generator.class_indices,
                yticklabels=test_generator.class_indices)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

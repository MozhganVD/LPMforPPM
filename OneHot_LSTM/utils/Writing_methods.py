import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, \
    precision_score, confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score


def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)

    predictions = [int(np.round(a)) for a in y_pred]
    predictions = np.array(predictions)

    y_test_integer = [int(np.round(a)) for a in y_test]
    y_test_integer = np.array(y_test_integer)

    CF_matrix = confusion_matrix(y_test_integer, predictions)
    report = classification_report(y_test_integer, predictions, digits=5)
    # Print the precision and recall, among other metrics
    Accuracy = accuracy_score(y_test_integer, predictions)
    F1_Score = f1_score(y_test_integer, predictions, average='weighted')
    Precision = precision_score(y_test_integer, predictions, average='weighted')
    Recall = recall_score(y_test_integer, predictions, average='weighted')
    cohen_kappa = cohen_kappa_score(y_test_integer, predictions)
    yhat_probs = y_pred[:, 0]
    auc = roc_auc_score(y_test, yhat_probs)

    return CF_matrix, report, Accuracy, F1_Score, Precision, Recall, cohen_kappa, auc


def evaluate_each_prefix(model, test_X, test_y, prefixes, checkpoint_dir, filename):
    prefix, accuracies, fscores, precisions, recalls, cohen_kappas, n_cases, auc = [], [], [], [], [], [], [], []
    for p in np.unique(prefixes):
        indices = [i for i, x in enumerate(prefixes) if x <= p]
        if len(indices) < 2:
            continue
        _, _, Accuracy, F1_Score, Precision, Recall, Cohen_kappa, AUC = evaluate(model, test_X[np.array(indices)],
                                                                                 test_y[np.array(indices)])

        accuracies.append(Accuracy)
        fscores.append(F1_Score)
        precisions.append(Precision)
        recalls.append(Recall)
        cohen_kappas.append(Cohen_kappa)
        n_cases.append(len(indices))
        auc.append(AUC)
        prefix.append(p)

    results_df = pd.DataFrame({"prefix": prefix, "n_cases": n_cases, "accuracy": accuracies,
                               "fscore": fscores, "precision": precisions, "recall": recalls,
                               "cohen_kappa": cohen_kappas, "AUC": auc})

    results_df.to_csv(checkpoint_dir + filename + ".csv", index=False)


def write_results_to_text(CF_matrix, report, Accuracy, F1_Score, Precision, Recall, Running_Time, Cohen_kappa, AUC,
                          results_file):
    with open(results_file, 'w') as f:
        f.write("Running time is %.5f \n" % Running_Time)
        f.write(str(CF_matrix))
        f.write("\n")
        f.write(str(report))
        f.write("\n")
        f.write("Accuracy = %.5f \n" % Accuracy)
        f.write("F1_Score = %.5f \n" % F1_Score)
        f.write("Precision = %.5f \n" % Precision)
        f.write("Recall = %.5f \n" % Recall)
        f.write("Cohen Kappa = %.5f \n" % Cohen_kappa)
        f.write("AUC = %.5f \n" % AUC)
        f.close()

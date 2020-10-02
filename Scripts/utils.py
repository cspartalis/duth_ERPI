import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score,\
                            recall_score,f1_score

def std_metrics(golden_truth, predictions):
    '''Standard Metrics Functions
    Input:
        The golden truth and the predictions
    Output:
        Prints the standard metrics (classification report, accuracy, precision, recall, f1)
    '''
    print(classification_report(golden_truth, predictions))

    precision  = precision_score(golden_truth, predictions, average='binary')
    recall     = recall_score(golden_truth, predictions, average='binary')
    f1_measure = f1_score(golden_truth, predictions, average='binary')

    print('Precision: %.3f' % precision)
    print('Recall: %.3f' % recall)
    print('F1: %.3f' % f1_measure)

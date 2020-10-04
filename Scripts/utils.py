import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score,\
                            recall_score,f1_score
import seaborn as sns
import matplotlib.pyplot as plt

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

def datetime_plot(title, x, y, palette):
    '''Barplot used in EDA of datetime
    Input:
        title   : title of the plot
        x       : x-axis of the plot
        y       : y-axis of the plot
        palette : palette of the plot
    Output:
        Barplot
    '''
    # Set the colors and the style of the figure
    sns.set_style("whitegrid")

    # Set the width and height of the figure
    plt.figure(figsize=(10,6))

    # Add title
    plt.title(title)

    # Bar chart showing average arrival delay for Spirit Airlines flights by month
    sns.barplot(x=x, y=y, palette=palette)

    # Add label for vertical axis
    plt.ylabel('Number of posts')

    plt.show()
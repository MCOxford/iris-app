#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt


def visualise_classifier(X, y, classifier, feature1_name=None, feature2_name=None, class_labels=None,
                         mesh_step_size=0.01):
    """
    Adapted from "Artificial Intelligence with Python" by Prateek Joshi (Packt Publishing)
    """
    fig, ax = plt.subplots()
    min_x1, max_x1 = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    min_x2, max_x2 = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    x, y1 = np.meshgrid(np.arange(min_x1, max_x1, mesh_step_size), np.arange(min_x2, max_x2, mesh_step_size))
    output = classifier.predict(np.c_[x.ravel(), y1.ravel()])
    output = output.reshape(x.shape)
    ax.pcolormesh(x, y1, output, cmap=plt.cm.gray, shading='auto')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=45, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    if feature1_name is None:
        ax.set_xlabel('x1')
    else:
        ax.set_xlabel(feature1_name)
    if feature2_name is None:
        ax.set_ylabel('x2')
    else:
        ax.set_ylabel(feature2_name)
    ax.set_xlim(min_x1, max_x1)
    ax.set_ylim(min_x2, max_x2)
    handles, labels = scatter.legend_elements()
    if class_labels is None:
        ax.legend(handles, classifier.labels, title='Classes')
    else:
        ax.legend(handles, class_labels, title='Classes')
    plt.show()

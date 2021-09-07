#!/usr/bin/env python3

"""
Code giving an overview of the basic ML classifiers (supervised or otherwise):
    - Logistic regression
    - K-means clustering
    - Naive Bayes
    
We also include functions to visualise our models and print out reports on accuracy,
recall, etc.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt

from ml_model import MLModel
from utilities import visualise_classifier

DATASET_PATH = "C:/Users/User/Documents/python/IrisApplication/iris.csv"


class IrisGUI(tk.Frame):
    """
    GUI class that uses the extracts and experiments with the iris dataset
    """
    X = None
    y = None
    X_feature1 = None
    X_feature2 = None
    ml_model = MLModel()
    is_fitted = False
    my_font = ("Arial", 10, "bold")
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    def __init__(self, master=None, file_path=DATASET_PATH):
        super().__init__(master)
        self.feature_label = tk.Label(self.master, bg='white', width=100,
                                      text='select TWO data features (The species column is automatically chosen as '
                                           'the y-label)',
                                      font=self.my_font)
        self.quit = tk.Button(self.master, text="QUIT", font=self.my_font, fg="red",
                              command=self.master.destroy)
        self.master = master
        self.master.title('Iris-ML')
        self.master.geometry('720x720')
        self.file_path = file_path
        self.l = tk.Label(self.master, bg='white', width=100, height=3, text='WELCOME!', font=self.my_font)
        self.pack()
        self.l.pack()
        self.feature_checkbuttons()
        self.data_button = tk.Button(self.master, text="Get Data", font=self.my_font,
                                     command=self.get_data)
        self.data_button.pack(side='top', pady=10)
        self.ml_model_selection()

    def feature_checkbuttons(self):
        self.feature_label.pack(side='top', pady=10)
        self.sepal_length = tk.IntVar()
        self.sepal_width = tk.IntVar()
        self.petal_length = tk.IntVar()
        self.petal_width = tk.IntVar()
        self.cb_sl = tk.Checkbutton(self.master, text=' Sepal Length (cm)', font=self.my_font,
                                    variable=self.sepal_length, onvalue=1, offvalue=0)
        self.cb_sl.pack(side='top', anchor=tk.W)
        self.cb_sw = tk.Checkbutton(self.master, text=' Sepal Width (cm)', font=self.my_font, variable=self.sepal_width,
                                    onvalue=2, offvalue=0)
        self.cb_sw.pack(side='top', anchor=tk.W)
        self.cb_pl = tk.Checkbutton(self.master, text=' Petal Length (cm)', font=self.my_font,
                                    variable=self.petal_length, onvalue=3, offvalue=0)
        self.cb_pl.pack(side='top', anchor=tk.W)
        self.cb_pw = tk.Checkbutton(self.master, text=' Petal Width (cm)', font=self.my_font, variable=self.petal_width,
                                    onvalue=4, offvalue=0)
        self.cb_pw.pack(side='top', anchor=tk.W)

    def getdata_button(self):
        self.data_button = tk.Button(self.master, text="Get Data",
                                     command=self.get_data)
        self.data_button.pack(side='top', pady=10)

    def ml_model_selection(self):
        self.ml_label = tk.Label(self.master, bg='white', width=100, text='Choose an ML model to fit',
                                 font=self.my_font)
        self.ml_label.pack(side='top', pady=10)
        self.ml_id = tk.IntVar()
        self.R_lr = tk.Radiobutton(self.master, text="Logistic Regression", font=self.my_font, variable=self.ml_id,
                                   value=1)
        self.R_lr.pack(side='top', anchor=tk.W)
        self.R_svm = tk.Radiobutton(self.master, text="SVM (deg. 2)", font=self.my_font, variable=self.ml_id, value=2)
        self.R_svm.pack(side='top', anchor=tk.W)
        self.R_nb = tk.Radiobutton(self.master, text="Naive Bayes", font=self.my_font, variable=self.ml_id, value=3)
        self.R_nb.pack(side='top', anchor=tk.W)
        self.ml_button = tk.Button(self.master, text='Fit Model', font=self.my_font, command=self.fit_model)
        self.ml_button.pack(side='top')

    def model_options(self):
        self.vis_button = tk.Button(self.master, text='Visualise model', font=self.my_font,
                                    command=self.plot_classifier)
        self.cov_button = tk.Button(self.master, text='Confusion\nmatrix', font=self.my_font,
                                    command=self.plot_confusion_matrix)
        self.sce_button = tk.Button(self.master, text='accuracy\nscore', font=self.my_font,
                                    command=self.print_accuracy_score)
        self.rep_button = tk.Button(self.master, text='Show report', font=self.my_font, command=self.print_report)
        self.pre_button = tk.Button(self.master, text='Predict Label', font=self.my_font, command=self.predict_label)
        options = [self.vis_button, self.cov_button, self.sce_button, self.rep_button, self.pre_button]
        for opt in options:
            opt.pack(side='top')

    def quit_button(self):
        self.quit.pack(side="top", anchor=tk.S, pady=10)

    def get_data(self):
        try:
            dataframe = pd.read_csv(self.file_path, encoding='utf-8')
            le = LabelEncoder()
            dataframe['species'] = le.fit_transform(np.array(dataframe['species']))
            features = [i - 1 for i in [self.sepal_length.get(), self.sepal_width.get(), self.petal_length.get(),
                                        self.petal_width.get()] if i > 0]
            if len(features) != 2:
                self.l.config(text='Please pick exactly TWO features', font=self.my_font, fg='red')
                return
            self.X = np.array(dataframe.iloc[0:, features])
            self.y = np.array(dataframe.iloc[0:, 4])
            self.X_feature1, self.X_feature2 = dataframe.columns[features[0]], dataframe.columns[features[1]]
            self.l.config(text='Constructed dataset', fg='green')
        except FileNotFoundError:
            self.l.config(text='File not found', fg='red')
        except Exception as e:
            print(e)
            self.l.config(text='Something went wrong', fg='red')

    def fit_model(self):
        if self.X is None or self.y is None:
            self.l.config(text='Dataset not constructed yet', fg='red')
            return
        ind = self.ml_id.get()
        if ind == 0:
            self.l.config(text='No model selected yet', fg='red')
            return
        self.ml_model.split_data(self.X, self.y)
        self.ml_model.fit_classifier(ind)
        if ind == 1:
            self.l.config(text='Fitted logistic regression model w/ features [{f1},{f2}]'.format(f1=self.X_feature1,
                                                                                                 f2=self.X_feature2),
                          fg='green')
        if ind == 2:
            self.l.config(text='Fitted SVM', fg='green')
        if ind == 3:
            self.l.config(text='Fitted Naive Bayes model', fg='green')
        if not self.is_fitted:
            self.model_options()
            self.is_fitted = True

    def plot_classifier(self):
        kwargs = {
            'feature1_name': self.X_feature1,
            'feature2_name': self.X_feature2,
            'class_labels': self.species,
        }
        visualise_classifier(self.X, self.y, self.ml_model.clf, **kwargs)

    def plot_confusion_matrix(self):
        fig, ax = plt.subplots()
        fig.suptitle('Confusion Matrix')
        mat = self.ml_model.give_confusion_matrix()
        ax.imshow(mat, interpolation='nearest', cmap=plt.cm.gray)
        fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.gray), ax=ax)
        ax.set_xticks(self.ml_model.labels)
        ax.set_xticklabels(self.species, rotation=45)
        ax.set_xlabel('Predicted Labels')
        ax.set_yticks(self.ml_model.labels)
        ax.set_yticklabels(self.species, rotation=45)
        ax.set_ylabel('True Labels')
        plt.show()

    def print_accuracy_score(self):
        score = self.ml_model.give_score()
        txt = 'Accuracy score:\t' + str(score)
        self.l.config(text=txt, fg='black')

    def print_report(self):
        report = self.ml_model.give_report(names=self.species)
        report_window = tk.Toplevel(self.master)
        report_window.title('Report')
        report_label = tk.Label(report_window, bg='white', font=self.my_font, text=report)
        report_label.pack()

    def predict_label(self):
        def return_label():
            try:
                x1 = float(entry1.get())
                x2 = float(entry2.get())
            except:
                info_label.configure(text='Make sure entries are real values', fg='red')
                return
            if x1 < 0 or x2 < 0:
                info_label.configure(text='Make sure entries are positive', fg='red')
                return
            arr = np.array([x1, x2]).reshape(1, -1)
            ind = self.ml_model.clf.predict(arr)[0]
            info_label.configure(text='Predicted species: {sp}'.format(sp=self.species[ind]), fg='green')

        predict_window = tk.Toplevel(self.master)
        predict_window.title('Predict Species')

        label = tk.Label(predict_window, bg='white', width=100,
                         text='choose (non-negative) real values for {x1} and {x2}.'.format(
                             x1=self.X_feature1.replace('_', ' '), x2=self.X_feature2.replace('_', ' ')),
                         font=self.my_font)
        label.grid(column=0, columnspan=2, row=0, pady=5)

        info_label = tk.Label(predict_window, bg='white', width=100,
                              text='', font=self.my_font)
        info_label.grid(column=0, columnspan=2, row=1, pady=2)

        label_feature1 = tk.Label(predict_window,
                                  text=self.X_feature1.replace('_', ' ') + ': ',
                                  font=self.my_font)
        label_feature1.grid(column=0, row=2)
        entry1 = tk.Entry(predict_window, width=20)
        entry1.insert(0, '0.0')
        entry1.grid(column=1, row=2)

        label_feature2 = tk.Label(predict_window,
                                  text=self.X_feature2.replace('_', ' ') + ': ',
                                  font=self.my_font)
        label_feature2.grid(column=0, row=3)
        entry2 = tk.Entry(predict_window, width=20)
        entry2.insert(0, '0.0')
        entry2.grid(column=1, row=3)

        button = tk.Button(predict_window, text="Predict", font=self.my_font, command=return_label)
        button.grid(column=0, columnspan=2, row=4)


if __name__ == '__main__':
    root = tk.Tk()
    app = IrisGUI(master=root)
    app.mainloop()

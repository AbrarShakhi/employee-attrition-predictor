try:
    import pandas as pd
    import numpy as np
    
    import matplotlib.pyplot as plt
    import plotly.express as px
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.figure_factory as ff

    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn import metrics
    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestRegressor
    
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    
    from os import path
    from sys import exit
    import math
except:
    print("Some libraries are not installed. Please install the required libraries.")
    exit(1)


class DatasetLoader:
    def __init__(self):
        self.kaggle_url = "thedevastator/employee-attrition-and-factors"
        
        self.file_name = "HR_Analytics.csv.csv"
        self.dir_name = "dataset"
        
        self.file_path = path.join("..", self.dir_name, self.file_name)
        self.github_url = f'https://raw.githubusercontent.com/AbrarShakhi/employee-attrition-predictor/main/{self.dir_name}/{self.file_name}'
    
    def load_from_kaggle(self):
        try:
            return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, self.kaggle_url, self.file_name)
        except:
            return None

    def load_from_github(self):
        try:
            return pd.read_csv(self.github_url)
        except:
            return None

    def load_from_local(self):
        try:
            return pd.read_csv(self.file_path)
        except:
            return None

class Main:
    def __init__(self):
        self.dataset_loader = DatasetLoader()
        self.df = None
        self.rseed = 42

    def load_data(self):
        self.df = self.dataset_loader.load_from_kaggle()
        if self.df is None:
            self.df = self.dataset_loader.load_from_github()
        if self.df is None:
            self.df = self.dataset_loader.load_from_local()
        if self.df is None:
            print("Failed to load the dataset from all sources.")
            exit(1)

    def display_info(self):
        print(self.df.head())
        print(self.df.info())
        print(self.df.describe())

    def delete_unnecessary_columns(self):
        self.df = self.df.dropna()
        self.target_column = "Attrition"

        columns=[col for col in self.df.columns if len(self.df[col].unique()) == 1]
        self.df = self.df.drop(columns=columns, axis=1)

        self.df = self.df.drop(columns=["EmployeeNumber"])

    def encode_categorical_columns(self):
        self.df[self.target_column] = self.df[self.target_column].map({'Yes': 1, 'No': 0})

        for col in self.df.columns:
            self.df[col] = self.df[col].astype("category").cat.codes

    def split_data(self):
        X = self.df.drop('Attrition', axis=1)  # Features
        y = self.df['Attrition']               # Lebels
        X = pd.get_dummies(X, drop_first=True)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.rseed, stratify=y
        )

    def standardize_data(self):
        scaler = preprocessing.StandardScaler()
        # scaler = preprocessing.MinMaxScaler()

        X_train = scaler.fit_transform(self.X_train)
        X_test = scaler.transform(self.X_test)


    def train_predict(self, model):
        # train the model
        model.fit(self.X_train, self.y_train)

        # predict y from X test
        y_pred = model.predict(self.X_test)

        return y_pred
    
    def show_result(self, kernel, y_pred):
        print(f"Accuracy Score {kernel}:", metrics.accuracy_score(self.y_test, y_pred))

        print(f"\nConfusion Matrix {kernel}:")
        print(metrics.confusion_matrix(self.y_test, y_pred))

        print(f"\nClassification Report {kernel}:")
        print(metrics.classification_report(self.y_test, y_pred))



if __name__ == "__main__":
    main = Main()
    main.load_data()
    main.display_info()
    main.delete_unnecessary_columns()
    main.encode_categorical_columns()
    main.split_data()
    main.standardize_data()

    # RVF
    model = svm.SVC(kernel='rbf', C=17.0, gamma='scale', random_state=main.rseed)
    y_pred = main.train_predict(model)
    main.show_result("RADIAL BASIS FUNCTION", y_pred)

    # LINEAR L1
    model = svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, max_iter=1000000000, random_state=main.rseed)
    y_pred = main.train_predict(model)
    main.show_result("LINEAR L1", y_pred)
    
    # LINEAR L2
    model = svm.LinearSVC(penalty='l2', loss='squared_hinge', max_iter=1000000000, random_state=main.rseed)
    y_pred = main.train_predict(model)
    main.show_result("LINEAR L2", y_pred)


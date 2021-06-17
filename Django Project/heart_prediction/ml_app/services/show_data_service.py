

from ml_app.submodels.model_configuration import ModelConfiguration
from ml_app.submodels.patient_model import Patient
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2


class ShowData:
    def __init__(self):
        self.categorical_val=['sex','cp','fbs','restecg','exang','slope','ca','thal']

    def load_data(self,model_id):
        name = 'ml_app'
        model=ModelConfiguration.objects.get(id=model_id)
        df=pd.read_csv(model.source_file)
        sns.set_style("whitegrid")
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(10, 8))

        # Scatter with postivie examples
        plt.scatter(df.age[df.target == 1],
                    df.thalach[df.target == 1])

        # Scatter with negative examples
        plt.scatter(df.age[df.target == 0],
                    df.thalach[df.target == 0])

        # Add some helpful info
        plt.title("Heart Disease in function of Age and Max Heart Rate")
        plt.xlabel("Age")
        plt.ylabel("Max Heart Rate")
        plt.legend(["Disease", "No Disease"])
        plt.savefig('media/images/scatter1.png',quality=10)

        img = cv2.imread('media/images/scatter1.png')

        corr_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(15, 15))
        ax = sns.heatmap(corr_matrix,
                         annot=True,
                         linewidths=0.5,
                         fmt=".2f",
                         cmap="YlGnBu")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.figure.savefig("media/images/heatmap1.png",quality=10)

        img1 = cv2.imread('media/images/heatmap1.png')
        plt.clf()
        plt.style.use("ggplot")

        plt1=df.drop('target', axis=1).corrwith(df.target).plot(kind='bar', grid=True, figsize=(12, 8),
                                                           title="Correlation with target")
        plt1.figure.savefig('media/images/correlation.png', quality=10)

        img2 = cv2.imread('media/images/correlation.png')
        return [img,img1,img2]
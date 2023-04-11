import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from helpers import Helpers as hp
import os
from os.path import join
from sklearn.metrics import confusion_matrix


def doPCA(df: pd.DataFrame, df_info: pd.DataFrame):
    
    components_PCA = 3
    x = np.array(df)
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=components_PCA)
    principalComponents = pca.fit_transform(x)
    dfPCA = pd.DataFrame(data=principalComponents, columns=[
                         f'PC{i+1}' for i in range(components_PCA)])

    dfPCA['label'] = df_info['name']

    loadings = pca.components_
    num_pc = pca.n_features_

    # fig = px.scatter_3d(dfPCA, x='PC1', y='PC2', z='PC3',
    #                     color='label', hover_data=df_info.to_dict('series'))
    # hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'PCA')
    # # fig.show()

    return dfPCA


def doLDA(df: pd.DataFrame, df_info: pd.DataFrame):
    components_LDA = 3
    x = np.array(df)
    y = df_info['name']
    
    print(x.shape, y.shape)
    lda = LinearDiscriminantAnalysis(n_components=components_LDA)
    linearComponents = lda.fit_transform(x, y)

    dfLDA = pd.DataFrame(data=linearComponents, columns=[
                         f'LD{i+1}' for i in range(components_LDA)])

    dfLDA['label'] = y

    fig = px.scatter_3d(dfLDA, x='LD1', y='LD2', z='LD3',
                        color='label', hover_data=df_info.to_dict('series'))
    # fig.show()
    hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'LDA')
    return dfLDA, lda


def loo_cv(df: pd.DataFrame, df_info: pd.DataFrame) -> float:
    result = []
    lda = LinearDiscriminantAnalysis(n_components=3)
    for i in range(len(df.index)):

        df_test = df.iloc[i]
        label_test = df_info.iloc[i]

        df_train = np.array(df.drop(df.index[i]))
        label_train = df_info.drop(df.index[i])['name'].to_list()

        lda.fit_transform(df_train, label_train)

        prediction = lda.predict([df_test])[0]
        result.append([prediction, label_test['name']])

    df = pd.DataFrame(result, columns=['pred', 'sample'])
    df['true_pred'] = df['pred'] == df['sample']
    accuracy = (df.true_pred.value_counts()[True]/len(df))*100
    print(f'accuracy of loo_cv is {accuracy}%')
    cm = confusion_matrix(df['sample'], df['pred'])
    print(cm)



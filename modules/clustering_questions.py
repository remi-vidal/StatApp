import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from IPython.display import display, HTML
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

def clustering(df, n_clusters):

    model = AgglomerativeClustering(n_clusters=4)
    model = model.fit(df)

    df['clust']=model.labels_
    display(df)

    #Répartition des clusters
    print('\033[1m' + "\nRépartition des clusters : \n" + '\033[0m')
    print(df['clust'].value_counts())


def dendogramme(df):

    plt.figure(figsize=(10, 7))
    plt.title("Customer Dendograms")
    shc.dendrogram(shc.linkage(df, method='ward'))
    plt.savefig('../data/dendogram.pdf')


def create_db_appliques():
    #Imports
    df_text_versions = pd.read_csv("../raw_data/deep_course_cf_text_versions.csv", low_memory = False)
    df_themes = pd.read_csv("../raw_data/deep_course_cf_themes.csv", low_memory = False)
    df_types = pd.read_csv("../raw_data/deep_course_cf_types.csv", low_memory = False)
    df_text_versions_node = pd.read_csv("../raw_data/deep_course_cf_text_version_nodes.csv", low_memory = False)

    df_themes.columns = df_themes.columns.str.replace('id','theme_id')
    df_types.columns = df_types.columns.str.replace('id','type_id')


    df_text = pd.merge(df_text_versions, df_themes, on='theme_id')
    df_text = pd.merge(df_text, df_types, on="type_id")

    df_appliques = pd.merge(df_text_versions_node, df_text, how='left', left_on = ['txt_version_surrogate_uuid'], right_on = ['surrogate_uuid'])
    return df_appliques

def create_fusion(df, n_clusters):

    # Dataframe où chaque ligne correspond à un client, chaque colonne à un texte.
    # Les cellules valent 1 si le texte est appliqué au client, 0 sinon. 

    df_app = pd.read_csv("../data/dummies_text.csv", index_col = "txt_node_id")

    # On ne garde que les clients qui ont un questionnaire suffisamment rempli
    df_app = df_app[df_app.index.isin(df.index)]

    model = AgglomerativeClustering(n_clusters)
    model = model.fit(df)
    pd.options.mode.chained_assignment = None 
    df_app['clust'] = model.labels_ 

    df_app['txt_node_id'] = df_app.index
    df_app = df_app.reset_index(drop=True)

    fus = pd.read_csv("../data/fusion.csv") 

    # On rajoute le cluster
    fus=pd.merge(fus,df_app[['txt_node_id','clust']],how='left', on='txt_node_id')

    return fus



def analyse_cluster(fus, n_clusters):

    for i in range (n_clusters):

        table = (fus.loc[fus['clust']==i].groupby('txt_node_id')['domain'].value_counts()/fus.loc[fus['clust']==i].groupby('txt_node_id')['domain'].count())*100

        print('\033[1m' + "\n Cluster {}\n".format(i) + '\033[0m')
        print(table.groupby('domain').mean())


def similarity(df, clust_number, answ, thres=0.9):
    perc = df[df["clust"]==clust_number].mean().to_frame()
    perc.columns = ["pourcentage"]
    perc = perc.loc[perc.index.str.contains(answ)]
    
    return perc[perc["pourcentage"]>thres]
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import display, HTML
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

def clustering(df, n_clusters):

    model = AgglomerativeClustering(n_clusters) #metrique ward par
    model = model.fit(df)

    df['clust']=model.labels_
    display(df)

    #Répartition des clusters
    print('\033[1m' + "\nRépartition des clusters : \n" + '\033[0m')
    print(df['clust'].value_counts())



def prediction(df,test_set):
    """
    Affecte chaque élément du test_set au cluster le plus proche à partir des labels du clustering présents dans df   
    """
    y = df['clust']  #labels du clustering
    df_temp = df.drop(columns=["clust"])
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(df_temp, y)

    return neigh.predict(test_set)



def score(fus, test_set, labels_test_set):

    precision_list=[]
    recall_list=[]
    # f_score_list=[]

    # C'est cette base qui fait foi...
    txt_client = pd.read_csv("../raw_data/deep_course_cf_text_version_nodes.csv", low_memory = False)
    
    n = len(test_set)

    for i in range(n):

        #Union des textes du cluster attribué au nouveau client
        union_txt_cluster = fus[fus['clust']==labels_test_set[i]].surrogate_uuid.unique()

        #Textes appliqués au client du test set
        test_set_txt = txt_client[txt_client["txt_node_id"]==test_set.index[i]]["txt_version_surrogate_uuid"]


        intersect = np.intersect1d(union_txt_cluster,test_set_txt)

        precision_list.append(len(intersect)/len(union_txt_cluster))

        recall_list.append(len(intersect)/len(test_set_txt))
        

    scores = pd.DataFrame(
        {"precision" : precision_list,
        "recall" : recall_list},
        index = test_set.index)

    scores["f_score"] = 2*(scores['precision']*scores['recall'])/(scores['precision']+scores['recall'])
    
    return scores   




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

    fus = fus[fus["txt_node_id"].isin(df.index)]

    return fus



def analyse_cluster(fus, n_clusters):

    for i in range (n_clusters):

        table = (fus.loc[fus['clust']==i].groupby('txt_node_id')['domain'].value_counts()/fus.loc[fus['clust']==i].groupby('txt_node_id')['domain'].count())*100

        print('\033[1m' + "\n Cluster {}\n".format(i) + '\033[0m')
        print(table.groupby('domain').mean())



def probas_txt_cluster(fus, clust_number, sort = True):

    # Calcul des pourcentage des textes au sein du cluster
    probas_txt = (fus.loc[fus['clust']==clust_number].groupby('surrogate_uuid')["txt_node_id"].count()/fus.loc[fus['clust']==clust_number]["txt_node_id"].nunique()).to_frame()
    
    # Base des textes. On prend celle de fus, un peu plus réduite (3181 textes sur les 3515 de deep_course_cf_text_versions.csv)
    list_text = pd.DataFrame(index=fus["surrogate_uuid"].unique())

    # On affecte les probas aux textes, en rajoutant 0 si le texte n'a pas été considéré dans le clustering.
    df_merge = pd.merge(list_text, probas_txt, left_index=True, right_index=True, how='outer').fillna(0)

    if sort : 
        #Classement du texte le plus probable au moins probable
        return df_merge.sort_values("txt_node_id", ascending = False)
    
    else :
        return df_merge



def similarity(df, clust_number, answ, thres=0.9):
    perc = df[df["clust"]==clust_number].mean().to_frame()
    perc.columns = ["pourcentage"]
    perc = perc.loc[perc.index.str.contains(answ)]
    
    return perc[perc["pourcentage"]>thres].sort_values("pourcentage", ascending = False)
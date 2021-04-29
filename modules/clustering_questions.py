import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from IPython.display import display, HTML
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt


def clustering(df, n_clusters):
    """Clusterise les clients de la dataframe. Les labels des clusters sont ajoutés à df dans une colonne

    Args:
        df (pandas dataframe): dataframe avec les clients en ligne et les indicatrice des réponses en colonne
        n_clusters (int): nombre de clusters
    """

    model = AgglomerativeClustering(n_clusters) #metrique ward par
    model = model.fit(df)

    df['clust']=model.labels_
    display(df)

    #Répartition des clusters
    print('\033[1m' + "\nRépartition des clusters : \n" + '\033[0m')
    print(df['clust'].value_counts())



def prediction(df,test_set):
    """
    Affecte chaque élément du test_set au cluster le plus proche à partir des labels du clustering présent dans df   
    """

    y = df['clust']  #labels du clustering
    df_temp = df.drop(columns=["clust"])
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(df_temp, y)

    return neigh.predict(test_set)



def score(fus, txt_client, test_set, labels_test_set):
    """Renvoie les différents scores (recall, precision, f_score)
       à partir d'une prédiction
    """
    precision_list=[]
    recall_list=[]
    faux_negatifs_list = []
    nombre_txt_list = []
    nombre_txt_cluster = []

    n = len(test_set)

    for i in range(n):

        #Union des textes du cluster attribué au nouveau client
        union_txt_cluster = fus[fus['clust']==labels_test_set[i]].surrogate_uuid.unique()

        #Textes appliqués au client du test set
        test_set_txt = txt_client[txt_client["txt_node_id"]==test_set.index[i]]["txt_version_surrogate_uuid"]
        # print(len(test_set_txt))

        intersect = np.intersect1d(union_txt_cluster,test_set_txt)
        precision_list.append(len(intersect)/len(union_txt_cluster))
        if len(test_set_txt) != 0:
            recall_list.append(len(intersect)/len(test_set_txt))
        else:
            recall_list.append(1)
        faux_negatifs_list.append(len(test_set_txt)-len(intersect))

        nombre_txt_list.append(len(test_set_txt))
        nombre_txt_cluster.append(len(union_txt_cluster))

    scores = pd.DataFrame(
        {"precision" : precision_list,
        "recall" : recall_list,
        "faux_negatifs" : faux_negatifs_list},
        index = test_set.index)
    

    # scores["f_score"] = 2*(scores['precision']*scores['recall'])/(scores['precision']+scores['recall'])
    scores["Nombre de textes à identifier"] = nombre_txt_list
    scores["Nombre de textes proposés"] = nombre_txt_cluster
    
    return scores   



def cross_validation(df, txt_client, n_clusters, K=10, shuffle = False):
    """Validation croisée pour choisir le nombre de cluster optimal

    Args:
        df: base des réponses : client en abscisses, réponses en ordonnée
        txt_client: sur chaque ligne, couple (texte-client) 
        n_clusters: nombres de clusters à tester
        K : Nombre d'échantillons pour la validation croisée
        shuffle : mélanger la base des réponses avant le découpage ?
    """
    precision_list = [] 
    recall_list = []
    f_score_list = []

    for n in n_clusters :
        
        print('\033[1m', n, " cluster(s)" + '\033[0m')
        kf = KFold(n_splits = K, shuffle=shuffle)

        scores = pd.DataFrame()

        for train_index, test_index in kf.split(df):

            # Création du train test et du test set, découpage de la dataframe
            train_set, test_set = df.iloc[train_index], df.iloc[test_index]

            # Entraînement du modèle
            model = AgglomerativeClustering(n)
            model = model.fit(train_set)

            train_set['clust']=model.labels_
            #Base de lien texte/client + cluster
            fus = create_fusion(train_set, n)

            scores = pd.concat([scores, score(fus, txt_client, test_set, prediction(train_set,test_set))])

        print(scores['faux_negatifs'].sum()," faux négatifs", " parmi ", 
            len(txt_client[txt_client["txt_node_id"].isin(df.index)]["txt_version_surrogate_uuid"]), " textes à identifier\n")

        print(scores.groupby("faux_negatifs")["faux_negatifs"].count().to_string())

        mean_scores = scores.mean()
        
        precision_list.append(mean_scores[0])
        recall_list.append(mean_scores[1])
        f_score_list.append(mean_scores[3])

        print("En moyenne ", mean_scores[2], "faux négatifs par client, soit", mean_scores[2]*len(scores), "au total")
        print("La médiane est de ",scores["faux_negatifs"].median()," faux négatifs\n")

    print('\033[1m' + "Liste précision :" + '\033[0m', precision_list,
          '\033[1m' + "\nListe recall :" + '\033[0m', recall_list,
          '\033[1m' + "\nListe f_scores :" + '\033[0m', f_score_list)
    

    # Graphe des scores
    X = n_clusters
    plt.plot(X,precision_list)
    plt.plot(X,recall_list)
    plt.plot(X,f_score_list)

    plt.legend(['precision', 'recall', 'f_score'])
    plt.xlabel("Nombre de clusters")
    plt.show()




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
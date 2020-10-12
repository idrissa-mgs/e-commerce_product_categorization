import pandas as pd

import ast
import os
import datetime
import re

#import matplotlib.pyplot as plt
import pickle
from cleantext import clean

os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok= True)




## Phase 1 ##
## LOAD DATA, BUILT HIERARCHY AND CLEAN ##



def make_flat_cat(dataframe_path):
    """
    Cette fonction permet de lire le dataframe brute puis de mettre à plat
    la colonne catégories afin de mettre sous forme de colonnes les catégorie 1 2 et 3.
    On a constaté que les catégorie 3 et 4 represente la même catégorie, c'est pourquoi
    nous avons remplacer la 4 par la 3.
    Ici on s'est intéresser qu'à la description On peut cependant utiliser toutes les autres attributs.
    :param dataframe_path:  Le chemin vers le dataframe
    :return: un dataframe avec comme colonnes (description, category_1, cat_2, cat_3)
    """
    df_iter = pd.read_csv(dataframe_path, chunksize = 100000)
    df_list = []
    taille = 0
    for df in df_iter:
        taille += len(df)
        df_list.append(df)
    df = pd.concat(df_list)
    df_copy = df.copy()
    df_copy['Catégories'] = df_copy['Catégories'].apply(lambda x: ast.literal_eval(x))
    for i in range(4):
        df_copy[f'category_{i + 1}'] = df_copy['Catégories'].apply(lambda x: x[f'Catégorie {i + 1}'][:-2])
    # del df_copy['Catégories']
    cols = ['Description', 'category_1', 'category_2', 'category_4']

    df_copy = df_copy[cols]
    df_copy.rename(columns ={'Description': 'description', 'category_4' : 'category_3'}, inplace =True)
    df_copy = df_copy.dropna(['description'])
    df_copy['description'] = df_copy['description'].apply(lambda x: str(x))
    df_copy['description'] = df_copy['description'].apply(clean_data)
    return df_copy


def clean_data(sentence):
    """
    Cette fonction permet de nettoyer les données, mis en minuscules, suppression des caractères spéciaux,
    suppression des caractères numériques isolés, etc
    :param sentence: une phrase, un document
    :return: le document nettoyé
    """
    sentence = sentence.replace("'", 'e ')
    clean_ = clean(sentence, lower=True, no_punct=True, fix_unicode=True, lang='fr', no_digits=False,
                   no_line_breaks=True)
    clean_ = re.sub(r'\s+\d+\s+', ' ', clean_)
    clean_ = re.sub(r'\s+\d+\s+', ' ', clean_)
    clean_ = re.sub(r'\|', ' ', clean_)
    clean_ = re.sub(r's+[\+x]\s+', ' ', clean_)
    clean_ = re.sub(r'^\W', ' ', clean_)
    # clean_=re.sub(r'^[a-zA-Z0-9]',' ',clean_)
    clean_ = clean_.replace('®', ' ')
    clean_ = clean_.replace('●', ' ')
    clean_ = clean_.replace('×', ' ')
    clean_ = clean_.replace('◆', ' ')
    clean_ = clean_.replace('➤', ' ')
    clean_ = clean_.replace('★', ' ')
    clean_ = clean_.replace('💕', ' ')
    clean_ = clean_.replace('🏃', ' ')
    clean_ = clean_.replace('♥', ' ')
    clean_ = clean_.replace('磅', ' ')

    clean_ = clean_.replace('\u200b\u200b12', ' ')
    clean_ = re.sub(r'\s+', ' ', clean_)
    return clean_



## PHASE 2  SUPPRESSION DES CATEGORIES RARES ET CREATIONS DES LABELS

def drop_cats(df,threshold=1, filtre =False):
    """
    Cette fonction permet de supprimer du dataset les catégories dont le nombre d'exemple
    est inférieur à un certain seuil
    :param df: le dataframe  de base nettoyé
    :param threshold:
    :return: le dataframe final prs pour la modélisation avec les bonnes cat
    """
    serie=df['category_3'].value_counts()<=threshold
    cat_to_baned=list(serie[serie.values==True].index)
    df_copy = df[~df['category_3'].isin(cat_to_baned)]
    if filtre :
        ##  On charge ici le dictionnaire qui contient les vrais id et les catégories textuelles associées
        ## On verifie bien que les catégories présentent dans le dataframe sont bien de
        ## Vraies catégories
        with open('data/dico_corr_origin.pkl', 'rb') as f:
            dico_id_text = pickle.load(f)
        df_copy = df_copy[df_copy['category_3'].isin(dico_id_text.keys())]
        ## Le dictionnaire chargé fait ici fait la correspondance entre les vraies ids et
        ## Les ids artificiels
        with open('data/correspondance_ids.pickle','rb') as f:
            correspondace_ids = pickle.load(f)
    df_copy['labels'] = df_copy['category_3'].apply(lambda x : correspondace_ids[x] if x in correspondace_ids.keys() else None)
    df_copy.dropna(subset = ['labels'], inplace = True)

    #df_copy.to_csv('new_dataset.csv', index=False)
    return df_copy


    


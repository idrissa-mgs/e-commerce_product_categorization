from data_preprocessing import *
import fasttext
import csv
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


def fasttext_representation(dataset, load_date= False) :
    """
    Cette fonction permet de faire le passage entre les données textuelles
    et les données numériques, entrées de nos modèles par la representation
    cbow de fasttext

    :param dataset: Le dataframe pret issu des scripts de data_preprocessing
    :return: Les variables indépendantes sous forme matricielle et les labels
    """
    df_description = dataset[['description']]
    df_description.to_csv('all_description.txt')

    if not load_data:
        start_time = datetime.datetime.now()
        fast_model = fasttext.train_unsupervised('all_description.txt', 'cbow', thread=8, epoch=5)
        end_time = datetime.datetime.now()
        fast_model.save_model("model/fasttext_unsup_model.bin")
        # print('DURATION, temps mis pour la representation des mots: ', end_time-start_time)
        # print('Nombre total de mots représentés :',len(fast_model.get_words()))
    else :
        fast_model = fasttext.load_model('model/fastext_unsup_model.bin')
    x_input = np.array([fast_model.get_sentence_vector(x) for x in df_description.description.tolist()])
    y_input = dataset['labels'].values()
    #np.save('data/x_input.npy', x_input)
    #np.save('data/y_input.npy', y_input)
    return x_input, y_input


def training_prep(x_input,y_input, load_data= False):
    """
    Cette fonction permet d'abord de diviser nos données en données de train et
    de test puis de faire du sur-échantillonnage sur le données de train
    :param x_input: matrice des variables indépendantes
    :param y_input: vecteurs des labels
    :return: (x_train, y_train, x_test, y_test)
    """
    random_state = 43
    if load_data:
        x_input  = np.load('data/x_input.npy', allow_pickle=True)
        y_input = np.load('data/y_input.npy', allow_pickle=True)
    x_train, x_test, y_train, y_test = train_test_split(x_input, y_input, test_size=0.3, random_state= random_state, stratify=y_input)
    x_train_res, y_train_res = balance_data(x_train, y_train)

    ### save data
    np.save('data/x_train.npy',x_train_res)
    np.save('data/y_train.npy', y_train_res)
    np.save('data/x_test.npy', x_test)
    np.save('data/y_test.npy', y_test)

    return x_train_res, x_test, y_train_res, y_test



def make_balance(y,level=200):
    """
    fonction essentiel pour parfaire le Sur echantillonnage nombre minimum de la classes fixé à 200
    :param y:
    :param level:
    :return:
    """
    count_dict=Counter(y)
    for key,val in count_dict.items():
        if val <level:
            count_dict[key]=level
    return count_dict


def balance_data(X, y):
    """
    Sur echantillonnage nombre minimum de la classes fixé à 200
    :param X:
    :param y:
    :return:
    """
    RANDOM_STATE = 43
    ros = RandomOverSampler(random_state=RANDOM_STATE, sampling_strategy=make_balance)
    # y=y.astype(int)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res

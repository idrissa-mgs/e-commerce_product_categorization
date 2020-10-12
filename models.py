from word_representation import *
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score,multi_labeled, h_precision_score, h_recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import fasttext
def f_net_model(x_train, y_train, nb_layer, load_data = False):
    """
    Construction du modèle f_net
    :param nb_layer: le nombre de neuronnes dans la couche cachée
    :return: le modèle construit
    """
    RANDOM_STATE = 43
    ## load the hierarchy
    hierarchy = pickle.load(open(os.path.join("data", "hierarchy.pickle"), "rb"))
    ## if we may load_data:
    if load_data :
        x_train = np.load(os.path.join("data", "x_train.npy"), allow_pickle= True)
        y_train = np.load(os.path.join("data", "y_train.npy"), allow_pickle=True)

    base_estimator = MLPClassifier(hidden_layer_sizes=(nb_layer), activation='relu', max_iter=200)
    clf = HierarchicalClassifier(base_estimator=base_estimator,
                                 class_hierarchy=hierarchy,
                                 prediction_depth="mlnp",
                                 algorithm="lcpn",
                                 progress_wrapper=None)
    start = datetime.datetime.now()
    clf = clf.fit(x_train, y_train)
    end = datetime.datetime.now()
    duration = end - start
    print("training duration :", str(duration))
    joblib.dump(clf, os.path.join("model", f"f_net_{str(nb_layer)}.joblib"))
    return clf

def f_rl_model(x_train, y_train, load_data =False):
    """
    Cette fonction crée le modèle de regression logistique comme
    classifieurs locaux par parent noeud
    :param x_train: x_train
    :param y_train: y_train
    :param load_data: si on charge les données ou pas
    :return: le model f_rl
    """
    hierarchy = pickle.load(open(os.path.join("data", "hierarchy.pickle"), "rb"))
    ## if we may load_data:
    if load_data:
        x_train = np.load(os.path.join("data", "x_train.npy"), allow_pickle=True)
        y_train = np.load(os.path.join("data", "y_train.npy"), allow_pickle=True)

    base_estimator = LogisticRegression(random_state = RANDOM_STATE, C = 10, multi_class = "ovr", n_jobs=-1,solver="newton-cg")
    clf = HierarchicalClassifier(base_estimator=base_estimator,
                                 class_hierarchy=hierarchy,
                                 prediction_depth="mlnp",
                                 algorithm="lcpn",
                                 progress_wrapper=None)
    start = datetime.datetime.now()
    clf = clf.fit(x_train, y_train)
    end = datetime.datetime.now()
    duration = end - start
    print("training duration :", str(duration))
    joblib.dump(clf, os.path.join("model", "f_rl.joblib"))
    return clf

def fasttext_classifier_model(dataset):
    """
    Cette fonction construit le modèle de classifier plat fastext
    :param dataset: notre dataset issue des modification sur le df brut
    :return: le classifier fastext
    """

    df_train_fastext = dataset[['category_3', 'description']]
    df_train_fastext.rename(columns={'category_3': 'labels'}, inplace=True)
    # shuffle the dataset
    df_train_fastext = df_train_fastext.sample(frac=1).reset_index(drop=True)
    train_df = df_train_fastext[:3500000]  # firts 300K row for train and the rest for test
    test_df = df_train_fastext[3500000:]
    train_df.to_csv('data/train_fasttext.txt', sep=' ', header=None, index=False)
    test_df.to_csv('data/test_fasttext.txt', sep=' ', header=None, index=False)
    start = datetime.datetime.now()
    #print("start training fasttext classifier  ")
    fasttext_clf = fasttext.train_supervised('data/train_fasttext.txt', lr=1.0, epoch=25, wordNgrams=2)
    #print("end training fasttext classifier  ")
    print("Fastext classifier duration : ",datetime.datetime.now() - start)
    ## PREDICTION
    print("--------- Prediction -----------")
    start = datetime.datetime.now()
    fasttext_clf.test('data/test_fasttext.txt')
    end = datetime.datetime.now()
    print('Test duration : ', end-start)
    fasttext_clf = fasttext.save_model('model/fastext_classifier.bin')

    return fasttext_clf

from models import *



def predict(x_test, clf):
    """
    Cette fonction permet de predire les label pour un x données et le modèle (hierarchique)
    :param x_test:
    :param clf:
    :return:
    """
    #model = joblib.load(model_path)
    y_pred = clf.predict(x_test)
    y_pred = np.array([int(x) for x in y_pred])
    np.save('data/y_pred.npy',y_pred)
    return y_pred

def evaluation(model_path, x_test, y_test):
    """
    Cette fonction permet d'évaluer le modèle en calculant le hprecision, h_recall
    et le hf1 (h_fbeta)
    :param model_path: le chemin du modèle à évaluer
    :param x_test: x_test
    :param y_test: y_true
    :return: les différentes metriques
    """
    clf = joblib.load(model_path)
    y_pred = predict(x_test, clf)
    test_shape = (y_test.shape[0], 1)
    y_test_reshaped = y_test.reshape(test_shape)
    y_pred_reshaped = y_pred.reshape(test_shape)
    with multi_labeled(y_test_reshaped, y_pred_reshaped, clf.graph_) as (y_test_, y_pred_, graph):
        h_fbeta = h_fbeta_score(y_test_, y_pred_, graph)
        h_precision = h_precision_score(y_test_, y_pred_, graph)
        h_recall = h_recall_score(y_test_, y_pred_, graph)
    return h_precision, h_recall, h_fbeta
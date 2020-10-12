from prediction_evaluation import *



if __name__== "__main__":
    ##preparation des données
    df  = make_flat_cat('data/crawlbot_data.csv')
    dataset = drop_cats(df)
    ## for training

    x,y = fasttext_representation(dataset)
    x_train, x_test, y_train, y_test = training_prep(x,y)
    #f_net_100 = f_net_model(x_train, y_train, nb_layer=100, load_data=False)

    ## for  evaluation
    #h_precision, h_recall, h_fbeta = ('model/f_net_100.joblib', x_test, y_test)

    #print('La hf1 correspondant :', h_fbeta)

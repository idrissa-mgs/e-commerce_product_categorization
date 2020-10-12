
# catbot 2.0

Catbot est un projet permettant de catégoriser automiquement des produits à partir de leur fiche produit.
Ceci est la deuxième version.

## Getting Started - How it works

### Introduction, objectif du bot


Les modèle ici présentés se servent uniquement de la description pour catégoriser, les autres attributs (titre, marque, prix) peuvent facilement être inclus.
On utilise ici 3 modèles, le modèle f_net utilisant un réseau de neuronnes comme classifieurs locaux, une regression logistique aussi comme classifieurs locaux,
et un dernier modèle de classification plate. Le détails de ces modèles peut être retrouvé sur le mémoire placé dans dropbox.


Le programme se divise en 4 grandes partie.\
Partie 1: Lecture, nettoyage et normalisation du dataframe principal.\
Partie 2: La représentation des documents sous forme matricielle. On utilise ici la representation de fasttext avec le modèle cbow.\
Partie 3: La mis en place des différentes modèles. Il en a trois deux modèles hiérarchique et un modèle plat (modèle de classification de fasttext)\
Partie 4: C'est la phase de prédiction et d'évaluation du (des) modèle(s). 


### Fonctionnement
Les deux premiers étapes peuvent être lancées en local (sur la vm) afin de cuillir les données d'entrées pour les différents modèles. 
L'entrainement des modèles se fait sur sagemaker en important en s3 les données d'entrées du modèle (x_train, y_train, x_test, y_test, hierarchy, ...).
Une fois l'entrainement fait sur sagemaker, on peut télécharger le modèle puis éffectuer les predictions en cas de besoin. Cependant l'évaluation
des modèles peut également se faire sur sagemaker. Si vous êtes à l'aise avec sagemaker, vous pouvez directement vous servir des scripts que j'ai utilisé à cet
effet sinon vous pouvez trouver de nombreux tutos sur le sujet.

Les modèles étant souvent très lourds (en termes de capacités de stockage), il serait mieux de les charger sur le disque temporaire de la vm à chaque fois qu'on en a besoin.

Dans le dossier il existe plusieurs dictionnaire dans le dossier data:\
-correspondance_ids.pickle qui fait la correspondance  entre les vrais ids et les ids artificiels (entiers) de tous les niveaux.\
-hierarchy.pickle :  qui correspond à la hiérarchie des données.\
-dico_corr_origin.pkl qui est le fichier original faisant la correspondance entre les ids et les catégories textuelles associées.\
-dico_corr.pickle qui est le nouveau dico_corr faisant la correspondance entre les ids artificiels et les catégories textuelles correspondantes.\
Le détail sur la construction de ces derniers peut etre retrouvé dans les notebooks.



L'arborescence générale des fichiers du projet est la suivante :
```bash
catbot_new_version (ou simplement catbot)
├── data
├── modele
├── data_preprocessing.py
├── word_representation.py
├── models.py
├── prediction_evaluation.py
├── main.py
```

## Prerequisites

Une certaine familiariaté avec les modèles de machine learning et une connaissance de base de la classification hiérarchique sont nécessaire pour mener à bien ce projet.
Pour ce qui est des modules à installer on peut bien ce referer aux modules importés (sous python 3.7)

## Authors

* **Nicolas BAOUAYA-MOULOMBA** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)
* **Idrissa MAGASSA** - *Collaborateur* = [Idrissa](https://github.com/idrissa-mgs)

## License

Ce programme est la propriété de Dropix.

## Acknowledgments



* Hat tip to anyone whose code was used
* Inspiration
* etc

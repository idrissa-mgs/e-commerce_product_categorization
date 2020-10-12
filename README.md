
# Automatize E-commerce product categorization

This project aims to automatize product categorization on a given marketplace

## Getting Started - How it works

### Introduction
We use here hierarchical classification and its metrics to classify e-commerce products. We only use here the description associated to the products, but in a future work we may experiment the others attributes such as titles, Brands and/or prices.
Here we experiment 3 models. The first use fasttext document representation models and a simple neural networks as local classifier per parent node. The second the same type of word representation but use Logistic Regression as LCPN. The last model use what's we call in the litterature of hierarchical classification flat model with fasttext supervised classification model.


The project is divived in 4 parts.\
Part 1: Read clean and normalize data.\
Part 2: word representation with fasttext cbow model.\
Part 3: Here we built our models: 2 hierarchical models and one flat (modèle de classification de fasttext)\
Part 4: Prediction and models evaluation . 


### Folder and files

In the working directory we create a folder (data) that contains all files needed particularly many dictionnairies such as:\
-correspondance_ids.pickle this dico makes the link between artificial ids and the corresponding ids on the marketplace.\
-hierarchy.pickle :  represent hierarchy of data.\
-dico_corr_origin.pkl this dico makes the link between ids on the marketplace and the associated textual categories.\
-dico_corr.pickle links artificial ids and the corresponding textual categories.\




The project is structured as :
```bash
project_name
├── data
├── modele
├── data_preprocessing.py
├── word_representation.py
├── models.py
├── prediction_evaluation.py
├── main.py
```

## Prerequisites

Some skills are needed to understand this projects: Notions on Hierarchical classification, Understanding of Machine learning models
The 3.7  version of python was used 

## Acknowledgments



* Hat tip to anyone whose code was used
* Inspiration
* etc

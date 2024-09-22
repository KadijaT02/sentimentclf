# Capstone Project - Sentiment Classification

### Table of content
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Description](#files)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Licensing, Authors and Acknowledgements](#licensing)

## Installation<a name="Installation"></a>

The project is a Python 3.6+ project and its library dependencies will 
need to be installed before it can be run.

It is recommended to install libraries in a dedicated virtual  
environement. The dependencies requirements can be found in 
**requirements.txt** file. For example, the dependencies can be 
installed using the following commands after activation of the virtual 
environement:

```
> cd /path/to/sentimentclf
> pip install -r requirements.txt
```

## Project Motivation<a name="motivation"></a>

The objective of this project was to build a classification model that 
predicts the sentiment of a given film review. More precisely, the 
classifier predict would predict if a given review is a positive one or 
a negative one. This project therefore combines concepts from Natural 
Language Processing (NLP) and Deep Learning.

The project idea took inspiration from the book Deep Learning Illustrated, A 
Visual, Interactive Guide to Artificial Intelligence (2020) by Jon Krohn 
with Grant Beyleveld and Aglaé Bassens [1]. In this book, the authors 
tie together concepts that were introduced in Chapter 11 - Natural 
Language Processing by experimenting with a sentiment classifier. Some 
of these concepts were new to me so this project idea felt like the 
perfect introduction to them.

The dataset we use is a collection of film reviews from the Internet 
Movie Database (IMDb). It was constructed by Andrew L. Maas *et al.* who 
then released it to the public [2]. The dataset is available via Keras' 
built-in small datasets [3].

## File Description<a name="files"></a>

The project can be broken down into 3 main directories:

1. **output** relates to the building of the model. It contains three 
subdirectories: **dense** contains the models trained during our first
training of the dense neural network (one per epoch). **experiments** 
stored the models built when running the experiments in an attempt to 
refine our initial solution. Lastly, **hyptuning** contains the results 
of the hyperparameter tuning along with the final solution 
`best_model.keras`.
2. **toolset** contains a custom class that was built to run the 
experiments.
3. **mywebapp** relates to the web application. It contains the Flask 
application factory, the blueprint, some utility functions, and the 
database. It also contains two subdirectories: **static** is where the
images that were added to the web application while **templates** is 
where the application's templates are stored.

**Important Note**: due to the size of some of the directories being more 
than 100MB, they were stored using the Git Large File Storage extension.

## Instructions

The Project Definition, Analysis, Methodology, Results, and Conclusion 
can all be found in the Jupyter notebook `sentimentclf.ipynb` located in 
the project's root directory.

To run the web application, the following command should be ran in the 
project's root directorty - the web app will then be available at 
http://127.0.0.1:5000/
```
> python -m flask --app mywebapp/app run
```
or
```
> flask --app mywebapp/app run
```

## Results<a name="results"></a>

The final solution is a dense neural network which showed great 
performance at predicting the sentiment of a given film review: it has 
high accuracy of 0.8690 and a high ROC AUC score of 94.35% on unsee test 
data.

However, it seems that the model can struggle detecting patterns of 
tokens that occur in sequences that would reveal the sentiment of a 
given review. This would be one aspect of the implementation that could 
be improved in the future. 

To remedy with this shortcoming, a potential solution could be to change
the architecture of the model for one that is more specialised in 
identifying patterns of tokens - for example, Recurrent Neural Networks 
(RNNs). 

## Licensing, Authors, and Acknowledgements<a name="licensing"></a>

This project would not have been possible without the helpful 
information found in the following resources:

[1] Jon Krohn with Grant Beyleveld and Aglaé Bassens (2020). Deep 
Learning Illustrated, *A Visual, Interactive Guide to Artificial 
Intelligence.

[2] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. 
Ng, and Christopher Potts. (2011). Learning Words Vectors for Sentiment 
Analysis, *The 49th Annual Meeting of the Association for Computational 
Linguistics (ACL 2011). 

[3] IMDB movie review sentiment classification dataset, Keras' built-in 
small datasets. Available at: http://keras.io/api/datasets/imdb/ 
(Accessed: 16 September 2024). 

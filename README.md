# Classification Between Docker Containers using Machine Learning Techniques

This repository contains the source code to perform the classification between the docker container. 

The application run insde the container produces logs. These are collected by the `Agentless Crawler`. The reference is `https://github.com/cloudviz/agentless-system-crawler`.

The logs of the container is collected and transformed into pictures. Then we apply fully connected vanilla neural network to perform the classification. 

Please refer to the `crawler metric processing.ipynb` and `Crawler Metric Processing - 2.ipynb` jupyter notebook file to explore the processign of the logs. The cleaned datased is available in `ubuntu_centos_classification.csv` file. 

The neural network model to show the classification between Ubuntu and Centos containers is given in `Ubunut_Centos_Classification.ipynb`. The suitability of the neural architecture is evaluated by autoencoder in `Autoencoders.ipynb` notebook file.

Instead of the neural network, we also show the use of `support vector machine` (SVM) in `OneclassSVM.ipynb` notebook.   

Instead of Docker container, one could use the Kubernetes platform, some ofthe useful commands to re-create the experiments are given in `Kubernetes Commands.docx`.

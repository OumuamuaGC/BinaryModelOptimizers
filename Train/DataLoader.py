import os
import numpy as np
import scipy.io as scio

from DataGenerator import NormalGenerator



def load_data(dataset):
    """
        Load a dataset and prepare the feature and label matrices (both training and testing set)
            Input: the dataset name
            Output: training feature, training label, testing feature, testing label   
    """
    dataset_path = '../Datasets/%s' % dataset
    if os.path.exists(dataset_path):
        train_feature_path = '%s/%s_train.mat' % (dataset_path, dataset)
        train_label_path = '%s/%s_train_label.mat' % (dataset_path, dataset)
        test_feature_path = '%s/%s_test.mat' % (dataset_path, dataset)
        test_label_path = '%s/%s_test_label.mat' % (dataset_path, dataset)
    else:
        raise Exception("Dateset Not Found!")
    
    train_feature = scio.loadmat(train_feature_path)['A'].toarray()    
    train_label = scio.loadmat(train_label_path)['b']

    if os.path.exists(test_feature_path):
        test_feature = scio.loadmat(test_feature_path)['A'].toarray()    
        test_label = scio.loadmat(test_label_path)['b']
    else:
        test_feature = None 
        test_label = None

    return train_feature, train_label, test_feature, test_label


def generate_data(c1, c2, sigma1, sigma2, size1, size2):
    """
        Generate a synthetic dataset that satisfies some distribution
            Input: 
            Output: training feature, training label, testing feature, testing label
    """
    generator = NormalGenerator(c1, c2, sigma1, sigma2, size1, size2)
    train_feature, train_label = generator.prepare()
    generator.normal()
    test_feature, test_label = generator.prepare()
    return train_feature, train_label, test_feature, test_label

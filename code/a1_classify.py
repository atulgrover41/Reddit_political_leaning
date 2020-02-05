import argparse
import os
import copy
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  
import numpy as np

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    cii = np.trace(C)
    cij = C.sum()
    print(cii/cij)
    return cii/cij

def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    CM = C.sum(axis=1)
    C00 = C[0][0] / CM[0]
    C11 = C[1][1] / CM[1]
    C22 = C[2][2] / CM[2]
    C33 = C[3][3] / CM[3]
    #print(C11, C22, C33, C00)
    return [C11, C22, C33, C00]


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''

    CM = C.sum(axis=0)

    C00 = C[0][0] / CM[0]
    C11 = C[1][1] / CM[1]
    C22 = C[2][2] / CM[2]
    C33 = C[3][3] / CM[3]
    #print(C11, C22, C33, C00)
    return [C11, C22, C33, C00]
    

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''

    SGD = SGDClassifier()
    GNB = GaussianNB()
    RFC = RandomForestClassifier(max_depth = 5,n_estimators = 10)
    MLP = MLPClassifier(alpha = 0.05)
    ABC = AdaBoostClassifier()



    SGD.fit(X_train,y_train)
    GNB.fit(X_train,y_train)
    RFC.fit(X_train,y_train)
    MLP.fit(X_train,y_train)
    ABC.fit(X_train,y_train)

    y_pred_SGD = SGD.predict(X_test)
    y_pred_GNB = GNB.predict(X_test)
    y_pred_RFC = RFC.predict(X_test)
    y_pred_MLP = MLP.predict(X_test)
    y_pred_ABC = ABC.predict(X_test)

    C_SGD = confusion_matrix(y_test, y_pred_SGD)
    C_GNB = confusion_matrix(y_test, y_pred_GNB)
    C_RFC = confusion_matrix(y_test, y_pred_RFC)
    C_MLP = confusion_matrix(y_test, y_pred_MLP)
    C_ABC = confusion_matrix(y_test, y_pred_ABC)

    Accuracy_SGD = accuracy(C_SGD)
    Accuracy_GNB = accuracy(C_GNB)
    Accuracy_RFC = accuracy(C_RFC)
    Accuracy_MLP = accuracy(C_MLP)
    Accuracy_ABC = accuracy(C_ABC)

    Recall_SGD = recall(C_SGD)
    Recall_GNB = recall(C_GNB)
    Recall_RFC = recall(C_RFC)
    Recall_MLP = recall(C_MLP)
    Recall_ABC = recall(C_ABC)

    Precision_SGD = precision(C_SGD)
    Precision_GNB = precision(C_GNB)
    Precision_RFC = precision(C_RFC)
    Precision_MLP = precision(C_MLP)
    Precision_ABC = precision(C_ABC)

    general_acc = {Accuracy_SGD: 0 ,Accuracy_GNB: 1,Accuracy_RFC: 2,Accuracy_MLP: 3,Accuracy_ABC: 4}
    iBest = general_acc[max(general_acc)]


    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        #     outf.write(f'Results for {classifier_name}:\n')  # Classifier name
        #     outf.write(f'\tAccuracy: {accuracy:.4f}\n')
        #     outf.write(f'\tRecall: {[round(item, 4) for item in recall]}\n')
        #     outf.write(f'\tPrecision: {[round(item, 4) for item in precision]}\n')
        #     outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
        pass

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    general_acc = [SGDClassifier(),GaussianNB(),RandomForestClassifier(max_depth = 5,n_estimators = 10),MLPClassifier(alpha = 0.05),AdaBoostClassifier()]
    iBst =general_acc[iBest]
    ABC_1k = copy.copy(iBst)
    ABC_5k = copy.copy(iBst)
    ABC_10k = copy.copy(iBst)
    ABC_15k = copy.copy(iBst)
    ABC_20k = copy.copy(iBst)
    rndm = np.random.randint(X_train.shape[0],size = 1000)
    X_1k = X_train[rndm]
    y_1k = y_train[rndm]
    rndm = np.random.randint(X_train.shape[0],size = 5000)
    X_5k = X_train[rndm]
    y_5k = y_train[rndm]
    rndm = np.random.randint(X_train.shape[0], size = 10000)
    X_10k = X_train[rndm]
    y_10k = y_train[rndm]
    rndm = np.random.randint(X_train.shape[0], size = 15000)
    X_15k = X_train[rndm]
    y_15k = y_train[rndm]
    rndm = np.random.randint(X_train.shape[0], size = 20000)
    X_20k = X_train[rndm]
    y_20k = y_train[rndm]
    ABC_1k.fit(X_1k,y_1k)
    ABC_5k.fit(X_5k,y_5k)
    ABC_10k.fit(X_10k,y_10k)
    ABC_15k.fit(X_15k, y_15k)
    ABC_20k.fit(X_20k, y_20k)

    y_pred_1k = ABC_1k.predict(X_test)
    y_pred_5k = ABC_5k.predict(X_test)
    y_pred_10k = ABC_10k.predict(X_test)
    y_pred_15k = ABC_15k.predict(X_test)
    y_pred_20k = ABC_20k.predict(X_test)

    C_1k = confusion_matrix(y_test,y_pred_1k)
    C_5k = confusion_matrix(y_test,y_pred_5k)
    C_10k = confusion_matrix(y_test,y_pred_10k)
    C_15k = confusion_matrix(y_test,y_pred_15k)
    C_20k = confusion_matrix(y_test,y_pred_20k)

    accuracy_1k = accuracy(C_1k)
    accuracy_5k = accuracy(C_5k)
    accuracy_10k = accuracy(C_10k)
    accuracy_15k = accuracy(C_15k)
    accuracy_20k = accuracy(C_20k)


    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        #     outf.write(f'{num_train}: {accuracy:.4f}\n'))
        pass

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print("This is  class 33 \n \n")
    general_acc = [SGDClassifier(),GaussianNB(),RandomForestClassifier(max_depth = 5,n_estimators = 10),MLPClassifier(alpha = 0.05),AdaBoostClassifier()]
    selector_32_5 = SelectKBest(f_classif,k=5)
    selector_1_5 = SelectKBest(f_classif, k=5)
    selector_1_50 = SelectKBest(f_classif, k=50)
    X_new_32_5 = selector_32_5.fit_transform(X_train,y_train)
    selector_32_50 = SelectKBest(f_classif,k=50)
    X_new_32_50 = selector_32_50.fit_transform(X_train,y_train)
    pp_5 = selector_32_5.pvalues_
    pp_50 = selector_32_50.pvalues_
    X_new_1_5 = selector_1_5.fit_transform(X_1k,y_1k)
    X_new_1_50 = selector_1_50.fit_transform(X_1k,y_1k)
    X_new_test_1_5 = selector_1_5.transform(X_test)
    X_new_test_1_50 = selector_1_50.transform(X_test)
    X_new_test_32_5 = selector_32_5.transform(X_test)
    X_new_test_32_50 = selector_32_50.transform(X_test)
    model_1_5 = copy.copy(general_acc[i])
    model_1_50 = copy.copy(general_acc[i])
    model_32_5 = copy.copy(general_acc[i])
    model_32_50 = copy.copy(general_acc[i])
    model_1_5.fit(X_new_1_5,y_1k)
    model_1_50.fit(X_new_1_50,y_1k)
    model_32_5.fit(X_new_32_5,y_train)
    model_32_50.fit(X_new_32_50,y_train)

    y_pred_1_5 = model_1_5.predict(X_new_test_1_5)
    y_pred_1_50 = model_1_50.predict(X_new_test_1_50)
    y_pred_32_5 = model_32_5.predict(X_new_test_32_5)
    y_pred_32_50 = model_32_50.predict(X_new_test_32_50)

    C_1_5 = confusion_matrix(y_test, y_pred_1_5)
    C_1_50 = confusion_matrix(y_test, y_pred_1_50)
    C_32_5 = confusion_matrix(y_test, y_pred_32_5)
    C_32_50 = confusion_matrix(y_test, y_pred_32_50)

    Acc_1_5 = accuracy(C_1_5)
    Acc_1_50 = accuracy(C_1_50)
    Acc_32_5 = accuracy(C_32_5)
    Acc_32_50 = accuracy(C_32_50)
    print("\n \n \n \n ")

    print(selector_1_5.get_support() , selector_32_5.get_support())
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        
        # for each number of features k_feat, write the p-values for
        # that number of features:
            # outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}')
        
        # outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        # outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        # outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        # outf.write(f'Top-5 at higher: {top_5}\n')
        pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        pass

def main(args):
    data = np.load(args.input)
    data = data['arr_0']
    print(data.shape)
    Y= data[:,173]
    X = data[:,:173]
    X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=40,train_size=0.8,test_size=0.2)
    iBest = class31(args.output_dir,X_train,X_test,y_train,y_test)
    X_1k,y_1k = class32(args.output_dir,X_train,X_test,y_train,y_test,iBest)
    class33(args.output_dir,X_train,X_test,y_train,y_test,iBest,X_1k,y_1k)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    main(args)
    # TODO: load data and split into train and test.
    # TODO : complete each classification experiment, in sequence.

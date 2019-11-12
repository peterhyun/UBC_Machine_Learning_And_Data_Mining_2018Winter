# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest
# from random_forest import RandomForest
from kmeans import Kmeans
from sklearn.cluster import DBSCAN


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]       
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        depths = np.arange(1,15) # depths to try       

        tr_errors = np.zeros(depths.size)
        te_errors = np.zeros(depths.size)
        
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            
            y_pred = model.predict(X)
            tr_errors[i] = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_errors[i] = np.mean(y_pred != y_test)
        
        plt.plot(depths, tr_errors, label="errorrate")
        plt.plot(depths, te_errors, label="errorrate")
        plt.xlabel("Depth")
        plt.ylabel("Error")

        plt.legend(['Training error', 'Testing error'], loc='upper right')

        fname = os.path.join("..", "figs", "q1_1_ErrorVsDepth.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape

        k = int(n/2)
        X_train = X[:k,:]
        y_train = y[:k]

        X_validate = X[k:,:]
        y_validate = y[k:]

        depths = np.arange(1,15) # depths to try       

        tr_errors = np.zeros(depths.size)
        te_errors = np.zeros(depths.size)
        
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_train)
            tr_errors[i] = np.mean(y_pred != y_train)

            y_pred = model.predict(X_validate)
            te_errors[i] = np.mean(y_pred != y_validate)

            print(max_depth)
            print(tr_errors[i])
            print(te_errors[i])
        
        plt.plot(depths, tr_errors, label="errorrate")
        plt.plot(depths, te_errors, label="errorrate")
        plt.xlabel("Depth")
        plt.ylabel("Error")

        plt.legend(['Training error', 'Testing error'], loc='upper right')

        fname = os.path.join("..", "figs", "q1_2_ErrorVsDepth.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]
        print(wordlist)
        print(wordlist[50])
        print(X[500, :])
        print(groupnames)
        arr = np.where(X[500, :] == 1)
        print(wordlist[arr])
        print(groupnames[y[500]])



    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        model = BernoulliNB()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (BernoulliNB) validation error: %.3f" % v_error)
    

    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        #print(Xtest)
        #print(y)
        #print(X)
        ytest = dataset['ytest']
        model = KNN(k=1)
        model.fit(X, y)
        y_pred = model.predict(Xtest)
        v_error = np.mean(y_pred != ytest)
        print("KNN testing error for k = 1: %.3f" % v_error)

        y_pred = model.predict(X)
        v_error = np.mean(y_pred != y)
        print("KNN training error for k = 1: %.3f" % v_error)


        model = KNN(k=3)
        model.fit(X, y)
        y_pred = model.predict(Xtest)
        v_error = np.mean(y_pred != ytest)
        print("KNN test error for k = 3: %.3f" % v_error)

        y_pred = model.predict(X)
        v_error = np.mean(y_pred != y)
        print("KNN training error for k = 3: %.3f" % v_error)

        
        model = KNN(k=10)
        model.fit(X, y)
        y_pred = model.predict(Xtest)
        v_error = np.mean(y_pred != ytest)
        print("KNN test error for k = 10: %.3f" % v_error)

        y_pred = model.predict(X)
        v_error = np.mean(y_pred != y)
        print("KNN training error for k = 10: %.3f" % v_error)
        
        model = KNN(k=1)
        model.fit(X, y)
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q_3_k1.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)

        print("Change model")
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X, y)
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q_3_k_1_scikit.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)


    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        print(X.shape)
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        print("RandomTree error")
        evaluate_model(RandomTree(max_depth=np.inf))
        print("RandomForest error")
        evaluate_model(RandomForest(max_depth = np.inf, num_trees = 70))
        print("scikit-learn's RandomForest error")
        evaluate_model(RandomForestClassifier(n_estimators=70))


    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']
        model = Kmeans(k=4)
        error = model.fit(X)
        minerror = error
        ideal_means = model.returnMeans()
        for i in range(49):
            model = Kmeans(k=4)
            error = model.fit(X)
            if error < minerror:
                ideal_means = model.returnMeans()
                minerror = error
                print('minerror changed to '+str(error))
        model.means = ideal_means
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_50_runs.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']
        model = Kmeans(k=4)
        error = model.fit(X)
        minerror = error
        ideal_means = model.returnMeans()
        for j in range(1, 11):
            for i in range(49):
                model = Kmeans(k=int(j))
                error = model.fit(X)
                if error < minerror:
                    ideal_means = model.returnMeans()
                    minerror = error
                    print('minerror changed to '+str(error))
        model.means = ideal_means
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_500_runs.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=13, min_samples=1)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))
        
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        plt.xlim(-25,25)
        plt.ylim(-15,30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)
        
    else:
        print("Unknown question: %s" % question)


# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)


    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0
        
        # YOUR CODE HERE FOR Q1.1.1
        x = np.sum(X, axis=0)
        print("Number of Stars: ", np.max(x))
        x_index = np.argmax(x);
        print(item_inverse_mapper[x_index]);

        # YOUR CODE HERE FOR Q1.1.2
        X_sum = np.sum(X_binary, axis = 1)
        print("Maximum number of reviews", np.max(X_sum))
        X_argmax = np.argmax(X_sum)
        print("Index of maximum", np.argmax(X_sum))
        print(user_inverse_mapper[X_argmax])

        # YOUR CODE HERE FOR Q1.1.3
        X_sum = np.sum(X_binary, axis = 1)
        print("Dimensions of X_sum", X_sum.shape)
        plt.xlabel("Users")
        plt.ylabel("Number of Ratings")
        plt.yscale('log', nonposy='clip')
        plt.hist(X_sum)
        plt.plot()
        fname = os.path.join("..", "figs", "q_1_1_3_plot1.png")
        plt.savefig(fname)
        plt.close()
        plt.gcf().clear()

        X_sum_2 = np.transpose(np.sum(X_binary, axis = 0))
        print("Dimensions of X_sum_2", X_sum_2.shape)
        plt.yscale('log', nonposy = 'clip')
        plt.xlabel("Items")
        plt.ylabel("Number of Ratings")
        plt.hist(X_sum_2)
        print(np.max(X_sum_2))
        fname = os.path.join("..", "figs", "q_1_1_3_plot2.png")
        plt.savefig(fname)
        plt.gcf().clear()

        unique_values, counts = np.unique(X.data, return_counts=True)
        plt.xlabel("Ratings")
        plt.ylabel("Frequency")
        plt.hist(np.transpose(X[np.nonzero(X)]), bins= [1, 2, 3, 4, 5, 6])
        fname = os.path.join("..", "figs", "q_1_1_3_plot3.png")
        plt.savefig(fname)

    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]

        print(url_amazon % grill_brush)

        # YOUR CODE HERE FOR Q1.2
        print("Euclidean Distance")
        nbrs0 = NearestNeighbors(n_neighbors=6).fit(np.transpose(X))
        distances0, indices0 = nbrs0.kneighbors(np.transpose(grill_brush_vec))
        print(distances0);
        print(np.array(indices0[0, :]));
        for i in range(0,6):
        	val = item_inverse_mapper[indices0[0, i]]
        	print(val)
        print()
        print("Normalize Euclidean Distance")
       	nbrs = NearestNeighbors(n_neighbors=6).fit(np.transpose(sklearn.preprocessing.normalize(X)))
        distances, indices = nbrs.kneighbors(np.transpose(grill_brush_vec))
        print(distances);
        print(np.array(indices[0, :]));
        for i in range(0,6):
        	val = item_inverse_mapper[indices[0, i]]
        	print(val)
        print()
        print("Metric Cosine")
        nbrs = NearestNeighbors(n_neighbors=6, metric="cosine").fit(np.transpose(X))
        distances, indices = nbrs.kneighbors(np.transpose(grill_brush_vec))
        print(distances);
        print(np.array(indices[0, :]));
        for i in range(0,6):
        	val = item_inverse_mapper[indices[0, i]]
        	print(val)

        # YOUR CODE HERE FOR Q1.3
        print()
        X_sum_2 = np.transpose(np.sum(X_binary, axis = 0))
        for i in range(0,6):
        	val = item_inverse_mapper[indices0[0, i]]
        	print(val, " - ", X_sum_2[indices0[0, i]][0, 0])
        for i in range(0,6):
        	val = item_inverse_mapper[indices[0, i]]
        	print(val, " - ", X_sum_2[indices[0, i]][0, 0])


    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # YOUR CODE HERE
        V = np.array([])
        for i in range(400):
            V = np.append(V,1)
        for i in range(100):
            V = np.append(V,0.1)
        V = np.diag(V)
        model = linear_model.WeightedLeastSquares()
        model.fit(X,y,V)

        utils.test_and_plot(model,X,y,title="Weighted Least Squares",filename="Weighted_least_squares_outliers.pdf")

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        model = linear_model.LeastSquaresBias()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, bias",filename="least_squares_bias.pdf")

        # YOUR CODE HERE

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p=%d" % p)
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)
            utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, bias",filename="least_squares_bias_"+str(p)+".pdf")


    else:
        print("Unknown question: %s" % question)


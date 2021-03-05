import numpy as np

class LinearRegressionScratch:
    def __init__(self):
        return None

    def fit(self, X, y, method, learning_rate=0.01, iterations=500, batch_size=32):
        """inputs :
            X : Ndarray : the data we want to train on
            y : Nadarray : the labeled data for regression
            method : string : ["ols","sgd"]. ols refers to Ordinary Least Square Estimators and sgd to Stochastic Gradient Descent
            learning : float : parameter for the sdg method
            iterations : integer : parameter for the sdg
            batch_size : integer : parameter for the sdg method

        """
        # x devient une matrice avec des 1 en bas avec la taille de y
        y = np.reshape(y, (len(y),1), order='C')
        X = np.concatenate([X, np.ones_like(y)], axis=1)
        # we gather X shape
        rows, cols = X.shape
        if method == 'ols':
            # if more raws than columns, we gather the rank of the matrix
            if rows >= cols == np.linalg.matrix_rank(X):
                # We use the famous formula for the OLS
                self.weights = np.matmul(
                    np.matmul(
                        np.linalg.inv(
                            np.matmul(
                                X.transpose(),
                                X)),
                        X.transpose()),
                    y)
            # if less raws than columns => impossible
            else:
                print('X has not full column rank. method=\'solve\' cannot be used.')

        # METHOD Stochastic gradient descent
        elif method == 'sgd':
            self.weights = np.random.normal(scale=1/cols, size=(cols, 1))
            for i in range(iterations):
                Xy = np.concatenate([X, y], axis=1)
                np.random.shuffle(Xy)
                X, y = Xy[:, :-1], Xy[:, -1:]
                for j in range(int(np.ceil(rows/batch_size))):
                    start, end = batch_size*j, np.min([batch_size*(j+1), rows])
                    Xb, yb = X[start:end], y[start:end]
                    gradient = 2*np.matmul(
                        Xb.transpose(),
                        (np.matmul(Xb,
                                   self.weights)
                         - yb))
                    self.weights -= learning_rate*gradient

        else:
            print('unknown method')

        return self

    def predict(self, X):
        """ function to predict the result with a test set based on what have learned the fit method
            X : NdArray: test set
        """

        # If no weights => no fit
        if not hasattr(self, 'weights'):
            print('Cannot predict. You should call the .fit() method first.')
            return


        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        if X.shape[1] != self.weights.shape[0]:
            #print(f'Shapes do not match. {X.shape[1]} != {self.weights.shape[0]}')
            return

        return np.matmul(X, self.weights)

    def rmse(self, X, y):
        # we use the fit method from bellow
        y_hat = self.predict(X).ravel()

        # if not prediction we quit
        if y_hat is None:
            return

        # otherwise we apply the formula
        return np.sqrt((np.mean((y_hat - y)**2)))

    def get_weights(self):
        return self.weights[:-1]

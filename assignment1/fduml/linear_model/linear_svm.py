"""
Linear SVM Classifier
"""
from builtins import range
import numpy as np

from .linear import LinearModel


class LinearSVMClassifier(LinearModel):
    """
    Linear SVM classifier with l2 regularization

    Parameters
    ----------
    learning_rate: (float) learning rate for optimization.

    reg: (float) regularization strength.

    num_iters: (integer) number of steps to take when optimizing

    batch_size: (integer) number of training examples to use at each step.

    verbose: (boolean) If true, print progress during optimization.

    loss_type: (string) naive version or vectorized version of softmax loss

    W: (array) parameter parameter matrix, 'naive' or 'vectorized'

    seed: (int) random seed
    """

    def __init__(self, learning_rate=1e-3, reg=1e-5, num_iters=50, batch_size=200,
                 verbose=False, loss_type='naive', seed=233):
        self.reg = reg
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_type = loss_type
        np.random.seed(seed)

        self.W = None

    def fit(self, X, y):
        """
        Train this softmax classifier using stochastic gradient descent.

        Parameters
        ----------
        X: A numpy array of shape (N, D) containing training data; there are N
            training samples each of dimension D.

        y: A numpy array of shape (N,) containing training labels; y[i] = c
            means that X[i] has label 0 <= c < C for C classes.

        Returns
        ----------
        self : object
            Fitted model with predicted self.coef_ and self.intercept_.
        """

        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes

        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        best_W = None
        min_loss = 1e9
        loss_history = []
        for it in range(self.num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, self.reg)
            loss_history.append(loss)
            if loss < min_loss:
                min_loss = loss
                best_W = self.W

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if self.verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" %
                      (it, self.num_iters, loss))

        self.W = best_W
        return self

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Parameters
        ----------
        X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns
        ----------
        y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """

        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Parameters
        ----------
        X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.

        y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        reg: (float) regularization strength.

        Returns
        ----------
        A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        if self.loss_type == 'naive':
            return self.svm_loss_naive(self.W, X_batch, y_batch, reg)
        elif self.loss_type == 'vectorized':
            return self.svm_loss_vectorized(self.W, X_batch, y_batch, reg)
        else:
            raise NotImplementedError

    def svm_loss_naive(self, W, X, y, reg):
        """
        Structured SVM loss function, naive implementation (with loops).
        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.
        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X: A numpy array of shape (N, D) containing a minibatch of data.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength
        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
        """
        dW = np.zeros(W.shape)  # initialize the gradient as zero

        # compute the loss and the gradient
        num_classes = W.shape[1]
        num_train = X.shape[0]
        loss = 0.0
        for i in range(num_train):
            scores = X[i].dot(W)
            correct_class_score = scores[y[i]]
            for j in range(num_classes):
                if j == y[i]:
                    continue
                margin = scores[j] - correct_class_score + 1  # note delta = 1
                if margin > 0:
                    loss += margin

                    # for incorrect classes (j != y[i]), gradient for class j is x * I(margin > 0)
                    # the transpose on the extracted input sample X[i] transforms it into a column vector
                    # for dw[:, j]
                    dW[:, j] += X[i].T

                    # for correct class (j = y[i]), gradient for class j is -x * I(margin > 0)
                    # the transpose on the extracted input sample X[i] transforms it into a column vector
                    # for dw[:, j]
                    dW[:, y[i]] += -X[i].T

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        loss /= num_train
        dW /= num_train

        # Add regularization to the loss.
        loss += reg * np.sum(W * W)

        #############################################################################
        # TODO:                                                                     #
        # Compute the gradient of the loss function and store it dW.                #
        # Rather than first computing the loss and then computing the derivative,   #
        # it may be simpler to compute the derivative at the same time that the     #
        # loss is being computed. As a result you may need to modify some of the    #
        # code above to compute the gradient.                                       #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, dW

    def svm_loss_vectorized(self, W, X, y, reg):
        """
        Structured SVM loss function, vectorized implementation.
        Inputs and outputs are the same as svm_loss_naive.
        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.
        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X: A numpy array of shape (N, D) containing a minibatch of data.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength
        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
        """
        loss = 0.0
        dW = np.zeros(W.shape)  # initialize the gradient as zero

        #############################################################################
        # TODO:                                                                     #
        # Implement a vectorized version of the structured SVM loss, storing the    #
        # result in loss.                                                           #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #############################################################################
        # TODO:                                                                     #
        # Implement a vectorized version of the gradient for the structured SVM     #
        # loss, storing the result in dW.                                           #
        #                                                                           #
        # Hint: Instead of computing the gradient from scratch, it may be easier    #
        # to reuse some of the intermediate values that you used to compute the     #
        # loss.                                                                     #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, dW

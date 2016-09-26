import numpy as np
from random import shuffle
import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores, axis=0)
    correct_class_score = scores[y[i]]
    sum = 0.0
    dsum = np.zeros_like(W)
    for j in xrange(num_classes):
      sum += np.exp(scores[j])
      dsum[:, j] += np.exp(scores[j]) * X[i]
    loss += -math.log(np.exp(correct_class_score) / sum)
    dW[:, y[i]] -= X[i]
    #print dsum
    dW +=  dsum / sum
     
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  scores -= np.reshape(np.max(scores, axis=1), (scores.shape[0], 1))
  correct_scores = scores[range(num_train), y]
  # get correct scores position
  mask = np.zeros_like(scores)
  mask[range(num_train), y] = 1.0
  # calculate probaility
  #pNumerator = np.exp(correct_scores)
  #pDenominator = np.sum(np.exp(scores), axis=1)
  p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  loss = np.sum(-np.log(p) * mask)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  #dP = -1.0 / p # N
  #dPNumerator = dP * (1 / pDenominator)
  #dPDenominator = dP * (-pNumerator / np.power(pDenominator, 2))
  dW += np.dot(X.T, p - mask) / num_train + reg * W 
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW


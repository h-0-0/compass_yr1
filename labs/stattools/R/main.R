# Some useful keyboard shortcuts for package authoring:
#
#   Install Package:           'Ctrl + Shift + B'
#   Check Package:             'Ctrl + Shift + E'
#   Test Package:              'Ctrl + Shift + T'

#' Create a model matrix
#'
#'Given data, D, and columns to leave out, r, creates the corresponding model matrix.
#'Note: doesn't notify you or make the model matrix full rank if its not.
#' @param D, data in the form of a data frame, matrix or vector
#' @param r, optional vector of column indices to omit from the model matrix
#' @return The corresponding model matrix X
#' @export
#' @examples
#' model_matrix(c(1,2,3,4))
#' model_matrix(cars)
#TODO: check matrix is full rank and if not make it so
model_matrix <- function(D,r) {
  if(!missing(r)){
    X <- data.matrix( D[, -r ] )
    colnames(X) <- rep(NULL, ncol(D))
  }
  else{
    X <- as.matrix(D)
  }

  if(is.vector(D)){
    ones <- rep(1, length(D))
  }
  else{
    ones <- rep(1, nrow(D))
  }

  X <- cbind(ones, X, deparse.level=0)
  X
}

#' Performs a polynomial feature transform
#'
#' Given data, x, and degree of polynomial transform to perform, b, computes and returns the polynomial feature transform of degree b on x, x_ft.
#' @param x, a numeric or vector
#' @param b, a numeric
#' @return x_ft, an array (either 1 or 2 dimensional)
#' @export
#' @examples
#' feat_trans(5,3)
#' feat_trans(c(1,2,3,4), 4)
#TODO: can you give this a matrix or dataframe?
feat_trans <- function(x,b){
  x_ft <- matrix(x, nrow=length(x), ncol=b, byrow=FALSE)
  for(i in 1:b){
    x_ft[,i] <- x_ft[,i]^i
  }
  x_ft
}

#' Linear Least Squares (LLS) estimator
#'
#' Given model matrix, X, and targets, y, returns the LLS estimator
#' @param X, a matrix
#' @param y, a numeric or vector
#' @return w, a numeric or vector
#' @export
#' @examples
#' X <- model_matrix(c(1,2,3,4))
#' y <- c(1,4,9,16)
#' LLS(X,y)
LLS <- function(X, y) {
  w <- solve(t(X) %*% X) %*% t(X)  %*% y
  w
}

#' Regularized Linear Least Squares (LLS-R) estimator
#'
#' Given model matrix, X, targets, y, and regularization rate, lambda, returns the LLS-R estimator.
#' Where our regularization term is "\eqn{\text{lambda} \cdot \mathbf{w}^{T} \mathbf{w}}"
#' @param X, a matrix
#' @param y, a numeric or vector
#' @param lambda, a numeric
#' @return w, a numeric or vector
#' @export
#' @examples
#' X <- model_matrix(c(1,2,3,4))
#' y <- c(1,4,9,16)
#' LLS_R(X,y, 1)
LLS_R <- function(X, y, lambda) {
    w <- solve(t(X) %*% X + lambda*diag(ncol(X))) %*% t(X)  %*% y
    w
  }

#' Cross validation
#'
#' This function is still under construction
# This function given a dataset (either as a dataframe or vector), D, and target variable, y, performs k-fold cross validation using a specified Regression Method, RM.
#TODO: change so we can pass RM functions with all arguments pre-determined except X and y
#TODO: add ability to change error function
regr_cross_val <- function(D, y, RM=LLS, k=10, ...){

  # We randomly shuffle the indices of the rows and using these randomized indices shuffle the rows of the dataset and the target variable (in the same order)
  if(is.vector(D)){
    ind <- sample(length(D))
    D_dash <- D[ind]
    y_dash <- y[ind]
  }
  else if(is.array(D)){
    ind <- sample(nrow(D))
    D_dash <- D[ind,]
    y_dash <- y[ind]
  }
  else{
    ind <- sample(nrow(D))
    D_dash <- D[paste(ind),]
    y_dash <- y[ind]
  }

  # We now create a list which indexes the rows in our dataset, we will use this to select groups of certain rows from the dataset.
  #   Hence what we have effectively done here is partition the dataset (randomly - as we shuffled the dataset previously) into groups.
  #   We split the dataset into 10 groups as we will be performing 10-fold cross validation.
  if(is.vector(D)){
    subsets <- cut(seq(1,length(D)), breaks = k, labels = FALSE)
  }
  else{
    subsets <- cut(seq(1,nrow(D)), breaks = k, labels = FALSE)
  }

  # We create a vector to store our cross-val errors
  errors <- c()

  # Loop for carrying out 10-fold cross-val
  #TODO: be able to adjust the number of folds
  for(i in 1:k){

    # We segment our data into testing, D.test, and training, D.train, datasets
    testIndexes <- which(subsets==i,arr.ind=TRUE)
    if(is.vector(D)){
      D.test <- D_dash[testIndexes]
      D.train <- D_dash[-testIndexes]
    }
    else{
      D.test <- D_dash[testIndexes, ,drop=FALSE]
      D.train <- D_dash[-testIndexes, ,drop=FALSE]
    }

    # We get the model matrix and predictor variables for the training data
    # and then we find the LS estimator
    X.train <- model_matrix(D.train)
    y.train <- y_dash[-testIndexes]
    w <- RM(X.train, y.train,...)

    # We get the model matrix and predictor variables for the testing data
    # and then we calculate the least squares testing error
    X.test <- model_matrix(D.test)
    y.test <- y_dash[testIndexes]
    test.error <- norm(y.test - X.test %*% w, type="2")**2
    # We add the testing error to our error vector
    errors[i] <- test.error

  }

  # We find and return the cross-val error
  error.CV <- sum(errors)/k
  error.CV
}


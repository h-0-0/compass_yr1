# Some useful keyboard shortcuts for package authoring:
#
#   Install Package:           'Ctrl + Shift + B'
#   Check Package:             'Ctrl + Shift + E'
#   Test Package:              'Ctrl + Shift + T'


# This function returns model matrix given the predictor variable (as a data frame or vector), D, and a vector of column indices which to leave omitted from the model matrix
model_matrix <- function(D,r) {
  #TODO: check matrix is full rank and if not make it so
  if(!missing(r)){
    X <- as.matrix(D[ -r ])
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

  X <- cbind(ones, X)
  X
}


# This function computes and returns the Linear Least Squares (LSS) solution, w, given the model matrix, X, and the target variables y.
LLS <- function(X, y) {
  w <- solve(t(X) %*% X) %*% t(X)  %*% y
  w
}

# This function computes and returns the Regularized Linear Least Squares (LSS) solution, w, given the model matrix, X, the target variables y and scalar lambda.
#   Where we use regularization term: lambda * t(w) %*% w
LLS_R <- function(X, y, lambda) {
    w <- solve(t(X) %*% X + lambda*diag(ncol(X))) %*% t(X)  %*% y
    w
  }


# This function given a dataset D performs 10-fold cross validation
#TODO: change so we can pass RM functions with all arguments pre-determined except X and y
#TODO: add ability to change error function
regr_cross_val <- function(D, y, RM=LLS, k=10, ...){

  # We randomly shuffle the indices of the rows and using these randomized indices shuffle the rows of the dataset and the target variable (in the same order)
  if(is.vector(D)){
    ind <- sample(length(D))
    D_dash <- D[ind]
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
  print(k)
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
      D.test <- D_dash[testIndexes, ]
      D.train <- D_dash[-testIndexes, ]
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

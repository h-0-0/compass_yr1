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
#TODO: check matrix is full rank and if not make it so IMPORTANT
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
#' If no x is given then it returns a function that will compute polynomial feature transform for x with degree set to b.
#' @param x, a numeric or vector (Optional)
#' @param b, a numeric
#' @return x_ft, an array (either 1 or 2 dimensional)
#' @export
#' @examples
#' poly_feat_trans(3,5)
#' # The following two are equivalent:
#' poly_feat_trans(4, c(1,2,3,4))
#' ---------------
#' pft <- poly_feat_trans(4)
#' pft(c(1,2,3,4))
#TODO: can you give this a matrix or dataframe?
#TODO: add testing
poly_feat_trans <- function(b, x=NULL){
  if(is.null(x)){
    f <- function(x){
      x_ft <- matrix(x, nrow=length(x), ncol=b, byrow=FALSE)
      for(i in 1:b){
        x_ft[,i] <- x_ft[,i]^i
      }
      x_ft
    }
    f
  }
  else{
    x_ft <- matrix(x, nrow=length(x), ncol=b, byrow=FALSE)
    for(i in 1:b){
      x_ft[,i] <- x_ft[,i]^i
    }
    x_ft
  }
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
#TODO: add testing
LLS <- function(X, y) {
  w <- solve(t(X) %*% X) %*% t(X)  %*% y
  w
}


#' Regularized Linear Least Squares (LLS-R) estimator
#'
#' Given model matrix, X, targets, y, and regularization rate, lambda, returns the LLS-R estimator.
#' If only the regularization rate is given will return function that computes LLS-R for the given regularization rate.
#' Where our regularization term is "\eqn{\text{lambda} \cdot \mathbf{w}^{T} \mathbf{w}}"
#' @param X, a matrix (Optional)
#' @param y, a numeric or vector (Optional)
#' @param lambda, a numeric
#' @return w, a numeric or vector
#' @export
#' @examples
#' X <- model_matrix(c(1,2,3,4))
#' y <- c(1,4,9,16)
#' # The following are equivalent
#' --------
#' f <- LLS_R(1)
#' f(X,y)
#' --------
#' LLS_R(1,X,y)
#TODO: add testing
LLS_R <- function(lambda, X=NULL, y=NULL) {
  f <- function(A, b){
    w <- solve(t(A) %*% A + lambda*diag(ncol(A))) %*% t(A)  %*% b
    w
  }
  if(is.null(X) || is.null(y)){
    return(f)
  }
  else{
    return(f(X,y))
  }
}

#' L2 Norm Error
#'
#' Given targets and predictions computes the L2 norm of their difference.
#' @param targ, a vector
#' @param pred, a vector
#' @return a numeric
#' @export
#' @examples
#' err <- E_l2( c(5,4,3,2,1), c(6,3,4,1,2) )
#TODO: add testing
E_l2 <- function(targ, pred){
  norm(targ - pred, type="2")**2
}


#' Cross validation
#'
#' This function is still under construction
# This function given a dataset (either as a dataframe or vector), D, and target variable, y, performs k-fold cross validation using a specified Regression Method, RM.
#TODO: Update documentation and add testing
CrossValidation <- setRefClass("CrossValidation",
                               fields=c( data="numeric",
                                         target="numeric",
                                         k="integer",
                                         Regr_method="ANY",
                                         Regr_method.name="character",
                                         E_fun = "ANY",
                                         E_fun.name = "character",
                                         Feat_trans = "ANY",
                                         Feat_trans.name = "character",
                                         k_test_errors = "numeric",
                                         estimators = "list",
                                         cv_error = "numeric",
                                         flag = "logical"
                                        )
                              )
CrossValidation$methods(
  initialize = function(data, target, k=as.integer(10), Regr_method=LLS, E_fun=E_l2 ) {
    .self$data <- data
    .self$target <- target
    .self$k <- k

    .self$setRegr_method(Regr_method)
    .self$Regr_method.name <- as.character(substitute(Regr_method)) # We set the name again when initializing otherwise will use argument name
    .self$setE_fun(E_fun)
    .self$E_fun.name <-as.character(substitute(E_fun)) # We set the name again when initializing otherwise will use argument name
    .self$Feat_trans <- NULL
    .self$Feat_trans.name <- ""
    .self$k_test_errors <- numeric(0)
    .self$estimators <- vector(mode = "list", length = 0)
    .self$cv_error <- numeric(0)
    .self$flag <- FALSE
  },

  # Carries out k-fold cross-validation for a regression problem
  regr_cv = function(){
    .self$flag <- TRUE
    # We randomly shuffle the indices of the rows and using these randomized indices shuffle the rows of the dataset and the target variable (in the same order)
    if(is.vector(.self$data)){
      ind <- sample(length(.self$data))
      D_dash <- .self$data[ind]
      y_dash <- .self$target[ind]
    }
    else if(is.array(.self$data)){
      ind <- sample(nrow(.self$data))
      D_dash <- .self$data[ind,]
      y_dash <- .self$target[ind]
    }
    else{
      ind <- sample(nrow(.self$data))
      D_dash <- .self$data[paste(ind),]
      y_dash <- .self$target[ind]
    }

    # We now create a list which indexes the rows in our dataset, we will use this to select groups of certain rows from the dataset.
    if(is.vector(.self$data)){
      subsets <- cut(seq(1,length(.self$data)), breaks = .self$k, labels = FALSE)
    }
    else{
      subsets <- cut(seq(1,nrow(.self$data)), breaks = .self$k, labels = FALSE)
    }

    # We assign empty vectors to fields k_test_errors and estimators so that we can assign them values in the upcoming for loop
    .self$k_test_errors <- vector(mode="numeric", length=.self$k)
    .self$estimators <- vector("list", length = .self$k)

    # Loop for carrying out k-fold cross-val
    for(i in 1:.self$k){
      # We segment our data into testing, D.test, and training, D.train, datasets
      testIndexes <- which(subsets==i,arr.ind=TRUE)
      if(is.vector(.self$data)){
        D.test <- D_dash[testIndexes]
        D.train <- D_dash[-testIndexes]
      }
      else{
        D.test <- D_dash[testIndexes, ,drop=FALSE]
        D.train <- D_dash[-testIndexes, ,drop=FALSE]
      }

      # We get the model matrix and predictor variables for the training data and then we find the estimator using our regression method
      if(is.null(.self$Feat_trans)){
        X.train <- model_matrix(D.train)
      }
      else{
        X.train <-model_matrix(.self$Feat_trans(D.train))
      }
      y.train <- y_dash[-testIndexes]
      w <- .self$Regr_method(X.train, y.train)
      .self$estimators[[i]] <- w

      # We get the model matrix and predictor variables for the testing data and then we calculate the error using our error function
      if(is.null(.self$Feat_trans)){
        X.test <- model_matrix(D.test)
      }
      else{
        X.test <-model_matrix(.self$Feat_trans(D.test))
      }
      y.test <- y_dash[testIndexes]
      test.error <- .self$E_fun(y.test, X.test %*% w)

      # We add the testing error to our error vector
      .self$k_test_errors[i] <- test.error

    }

    # We find and return the cross-val error
    .self$cv_error <- sum(.self$k_test_errors)/.self$k
    .self$cv_error
  },

  show = function() {
    cat("head(data)    =", head(.self$data), "\n", sep=" ")
    cat("head(target)  =", head(.self$target), "\n", sep=" ")
    cat("k             =", .self$k, "\n", sep=" ")
    cat("Regr_method   =", .self$Regr_method.name, "\n", sep=" ")
    cat("E_fun         =", .self$E_fun.name, "\n", sep=" ")
    if(.self$flag){
      cat("k_test_errors =", head(.self$k_test_errors), "\n", sep=" ")
      for (v in 1:length(head(.self$estimators))) {
        cat(paste("estimators[",as.character(v), "] =", sep=""), head(.self$estimators)[[v]], "\n", sep=" ")
      }
      cat("cv_error      =", .self$cv_error, "\n", sep=" ")
    }
  },

  # When called with problem type (at the moment only support regression:"r") computes the cv error and returns it
  getCv_error = function(t="r") {
    if (t == "r"){
      .self$regr_cv()
      return(.self$cv_error)
    }
    else{
      stop("Not given valid problem type")
    }
  },

  getEstimators = function() {
    return(.self$estimators)
  },

  getK_test_errors = function() {
    return(.self$k_test_errors)
  },

  # Allows user to change the data that cv will be performed on (user must give both data and targets)
  setData = function(data, target) {
    .self$data <- data
    .self$target <- target
  },

  # Checks that the function passed to it has closure
  check_closure = function(f){
    if (typeof(f) == "closure"){
      return(TRUE)
    }
    else{
      return(FALSE)
    }
  },

  # Sets the regression method passed to it to Regr_method if function is of type "closure"
  setRegr_method = function(Regr_method){
    if (.self$check_closure(Regr_method)){
      .self$Regr_method <- Regr_method
      .self$Regr_method.name <- as.character(substitute(Regr_method))
    }
    else{
      stop("Regr_method doesnt have type: closure")
    }
  },

  # Sets the error function passed to it to E_fun if function is of type "closure"
  setE_fun = function(E_fun){
    if (.self$check_closure(E_fun)){
      .self$E_fun <- E_fun
      .self$E_fun.name <-as.character(substitute(E_fun))
    }
    else{
      stop("E_fun doesnt have type: closure")
    }
  },

  # Sets the value of k and makes sure its an integer
  setk = function(i){
    if (is.integer(i)){
      .self$k <- i
    }
    else{
      stop("setk was not given an integer")
    }
  },

  setFeat_trans = function(Feat_trans){
    if (.self$check_closure(Feat_trans)){
      .self$Feat_trans <- Feat_trans
      .self$Feat_trans.name <-as.character(substitute(Feat_trans))
    }
    else{
      stop("Feat_trans doesnt have type: closure")
    }
  }
)

#### Here we will test both carrying out kernel regression with different kernel functions and using the CrossValidation class ####

#First we load in some data which we will use to carry out our tests
D <- read.table(url("https://hastie.su.domains/ElemStatLearn/datasets/prostate.data"))
# We create a CrossValidation object initialized with our data.
cv <- CrossValidation$new(D[ 1:(ncol(D)-2) ], D$lpsa, k=as.integer(15))

test_that("K_LLS_R works with k_linear as kernel function using cross validation", {
  set.seed(9012023)
  cv$setRegr_method(K_LLS_R(k_linear,1), km=TRUE)
  awns <- signif(cv$getCv_error(), 4)
  expect_equal(awns, 3.441)
})

test_that("K_LLS_R works with k_poly as kernel function using cross validation", {
  set.seed(9012023)
  cv$setRegr_method(K_LLS_R(k_poly(2) ,1), km=TRUE)
  awns <- signif(cv$getCv_error(), 4)
  expect_equal(awns, 5.415)
})

test_that("K_LLS_R works with k_RBF as kernel function using cross validation", {
  set.seed(9012023)
  cv$setRegr_method(K_LLS_R(k_RBF(3) ,1), km=TRUE)
  awns <- signif(cv$getCv_error(), 4)
  expect_equal(awns, 24.43)
})

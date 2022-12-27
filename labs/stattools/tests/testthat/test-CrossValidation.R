data(longley)
D <- longley[-ncol(longley)]
y <- as.numeric(longley$Employed)
cv <- CrossValidation$new(D, y, k=as.integer(4))

test_that("setRegr_method and getCv_error works", {
  set.seed(9012023)
  cv$setRegr_method(LLS)
  err <- signif(cv$getCv_error(),4)
  expect_equal(err, 1.24)

  cv$setRegr_method(K_LLS_R(k_linear,1), km=TRUE)
  err <- signif(cv$getCv_error(),4)
  expect_equal(err, 1.285)
})

test_that("setData works", {
  D <- read.table(url("https://hastie.su.domains/ElemStatLearn/datasets/prostate.data"))
  cv$setData(D[ 1:(ncol(D)-2) ], D$lpsa)
  expect_equal(cv$data, D[ 1:(ncol(D)-2) ])
  expect_equal(cv$target, D$lpsa)
})

test_that("setE_fun works", {
  e_fun <- function(targ, pred){sum((targ - pred)**2)}
  cv$setE_fun(e_fun)
  expect_equal(cv$E_fun, e_fun)
})

test_that("setk works", {
  k <- as.integer(2)
  cv$setk(k)
  expect_equal(cv$k, k)
})

test_that("setFeat_trans works", {
  set.seed(9012023)
  cv$setFeat_trans(poly_feat_trans(3))
  err <- signif(cv$getCv_error(),4)
  expect_equal(err, 32.89)
})

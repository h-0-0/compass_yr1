# We create a dataset to carry out testing with
set.seed(9012023)
x_start <- -4
x_end <- 4
x <- seq(x_start, x_end, length.out=200)
e <- rnorm(200, mean=0, sd=0.64)
y <- exp(1.5*x -1) + e

test_that("LLS works", {
  X <- model_matrix(x)
  w_LS <- LLS(X,y)
  w_LS <- signif(w_LS, 3)

  awns <- c(12.6, 7.86)
  dim(awns) <- c(2, 1)

  expect_equal(w_LS, awns)
})

test_that("LLS_R works", {
  X <- model_matrix(x)
  w_LS_R <- LLS_R(2, X, y)
  w_LS_R <- signif(w_LS_R, 3)

  awns <- c(12.5, 7.84)
  dim(awns) <- c(2, 1)

  expect_equal(w_LS_R, awns)
})

test_that("LLS_R works when X and y not given at first", {
  X <- model_matrix(x)
  f <- LLS_R(2)
  w_LS_R <- f(X,y)
  w_LS_R <- signif(w_LS_R, 3)

  awns <- c(12.5, 7.84)
  dim(awns) <- c(2, 1)

  expect_equal(w_LS_R, awns)
})

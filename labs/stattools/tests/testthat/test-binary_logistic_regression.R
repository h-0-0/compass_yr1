test_that("Binary logistic regression on simple dataset", {
  # First we create some data to carry out the tests with
  x <- c(1:10)
  y <- c(c(rep(0,5), rep(1,5)))

  results <- optim(par = c(0,0), fn = binlr_nll, D=x, y=y)
  params <- signif(results$par,4)
  expect_equal(params, c(-45.160, 8.189))

  p1 <- round(prediction(5, results$par))
  p2 <- round(prediction(6, results$par))
  expect_equal(p1, 0)
  expect_equal(p2, 1)
})

test_that("Binary logistic regression on more complex dataset", {
  diabetes <- read.csv(url('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'))
  x <- diabetes[,-ncol(diabetes)]
  y <- diabetes[,ncol(diabetes)]
  results <- optim(par = rep(0, ncol(x)+1), fn = binlr_nll, D=x, y=y)
  params <- signif(results$par,4)
  expect_equal(params, c(-0.6973, 0.1271, 0.009711, -0.03852, -0.003901,  0.0007273,  0.04765, 0.3615, -0.01526))
})

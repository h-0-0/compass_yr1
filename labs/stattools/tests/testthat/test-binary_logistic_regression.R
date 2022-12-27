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
  library("mlbench")
  data(Sonar)
  x <- Sonar[,1:ncol(Sonar)-1]
  y <- ifelse(Sonar[,ncol(Sonar)] == 'R', 0, 1)
  results <- optim(par = rep(1,ncol(x)+1), fn = binlr_nll, D=x, y=y)
  params <- signif(results$par,4)
  expect_equal(params, c(0.7492, 1.0320, 1.0290, 1.0260, 1.0190, 1.0150, 1.0030, 0.9980, 1.0870, 0.9585, 0.9749, 1.1940,
                         1.0140, 1.0110, 1.0090, 1.0060, 1.0010, 0.9959, 0.9947, 0.9177, 0.9829, 0.9709, 0.9684, 0.9624,
                         0.9591, 0.9565, 0.9481, 0.9507, 0.9540, 0.9735, 0.9668, 0.9798, 0.9934, 0.9972, 0.9959, 0.9922,
                         0.9910, 0.9984, 1.0030, 1.0050, 1.0080, 1.0070, 1.0100, 1.0130, 1.0150, 0.9814, 0.9929, 1.0070,
                         1.0110, 1.0220, 1.0340, 1.0370, 1.0390, 1.0430, 1.0440, 1.0460, 1.0480, 1.0440, 0.9698, 1.0470,
                         1.0500))
})

test_that("model_matrix works when just supplied with vector", {
  x <- c(1,2,3,4)
  expect_equal(
    stattools::model_matrix(x)
  , cbind(c(1,1,1,1) , c(1,2,3,4) )
  )
})

test_that("model_matrix works when just supplied with matrix", {
  x <- matrix(c(1:16), 4,4)
  expect_equal(
    stattools::model_matrix(x)
    , cbind(c(1,1,1,1) , matrix(c(1:16), 4,4) )
  )
})

test_that("model_matrix works when supplied with matrix and vector of columns to ommit", {
  x <- matrix(c(1:16), 4,4)
  r <- c(3,4)
  expect_equal(
    stattools::model_matrix(x,r)
    , cbind(c(1,1,1,1) , matrix(c(1:8), 4,2) )
  )
})

test_that("model_matrix works when supplied with data-frame and vector of columns to ommit", {
  x_1 <- c(1:4)
  x_2 <- c(5:8)
  x_3 <- c(9:12)
  x_4 <- c(13:16)
  x <- data.frame(x_1, x_2, x_3, x_4)
  r <- c(3,4)

  expect_equal(
    stattools::model_matrix(x,r)
    , cbind(c(1,1,1,1) , x_1, x_2 , deparse.level=0)
  )
})

test_that("feat_trans works when given a single value and a degree larger than 2", {
  x <- 5
  b <- 3
  expect_equal(
    feat_trans(x,b)
    , t(matrix(cbind(x, x^2, x^3)))
    )
})

test_that("feat_trans works when given a vector and a degree larger than 2", {
  x <- c(1,2,3,4)
  b <- 3
  expect_equal(
    feat_trans(x,b)
    , cbind(x, x^2, x^3, deparse.level=0)
  )
})

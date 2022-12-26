test_that("feat_trans works when given a single value and a degree larger than 2", {
  x <- 5
  b <- 3
  expect_equal(
    poly_feat_trans(b,x)
    , t(matrix(cbind(x, x^2, x^3)))
  )
})

test_that("feat_trans works when given a vector and a degree larger than 2", {
  x <- c(1,2,3,4)
  b <- 3
  expect_equal(
    poly_feat_trans(b,x)
    , cbind(x, x^2, x^3, deparse.level=0)
  )
})

test_that("feat_trans works when given a degree less than 2", {
  x <- c(1,2,3,4)
  b <- 1
  expect_equal(
    poly_feat_trans(b,x)
    , cbind(x, deparse.level=0)
  )
})

test_that("feat_trans works if x null", {
  x <- c(1,2,3,4)
  b <- 3
  f <- poly_feat_trans(b)
  expect_equal(
    f(x)
    , cbind(x, x^2, x^3, deparse.level=0)
  )
})

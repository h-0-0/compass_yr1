test_that("Check E_l2 works", {
  err <- E_l2( c(5,4,3,2,1), c(6,3,4,1,2) )
  expect_equal(err, 5)
})

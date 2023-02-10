#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export(name = "moving_avg")]]
NumericVector moving_avg(const NumericVector y, const int lag)
{
  int ni = y.size();
  NumericVector ys(ni);
  
  // Define sum to store curent sum of window we are computing moving average for
  double sum = 0;

  // We now compute the moving average
  for(int i = 1; i < ni; i++){
      // Add newest sample
      sum += y[i] ;
          
      if( i >= lag ){
          // Subtract oldest sample
          sum -= y[i - lag] ;
          ys[i] = sum / lag ;
      }
      else{
          // Just let value be 0 if we aren't passed lag yet
          ys[i] = 0.0;
      }
  }
  return ys;
}
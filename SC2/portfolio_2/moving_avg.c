#include <R.h>
#include <Rinternals.h>
// ^ Headers needed for interfacing with R

// We define our function that returns type SEXP:
SEXP moving_avg(SEXP y, SEXP lag)
{
    // We start by declaring our variable types
    int ni;
    double *xy, *xys;
    int lagi;
    SEXP ys;

    // We initialize y (type SEXP),
    // coerceVector() is used to coerce y to type REALSXP,
    // PROTECT() protects the output from being cleaned up by R's garbage collector.
    y = PROTECT(coerceVector(y, REALSXP));

    // We define ni (length of y).
    ni = length(y);

    // We define ys,
    // we use allocVector() to allocate memory for type REALSXP of length ni,
    // we again use PROTECT() to prevent R from garbage collecting.
    ys = PROTECT(allocVector(REALSXP, ni));

    // We define lagi (the lag of the moving average),
    // REAL() is used to asses the object it takes as input 
    // and return a double pointer to its real part,
    // we then use [0] to access the first value of this real part
    lagi = REAL(lag)[0];

    // We define xy (points to real part of y).
    xy = REAL(y); 

    // We define xys (points to real part of ys).
    xys = REAL(ys);


    // Define sum to store curent sum of window we ae computing moving average for
    double sum = 0;

    // We now compute the moving average
    for(int i = 1; i < ni; i++){
        // Add newest sample
        sum += xy[i] ;
            
        if( i >= lagi ){
            // Subtract oldest sample
            sum -= xy[i - lagi] ;
            xys[i] = sum / lagi ;
        }
        else{
            // Just let value be 0 if we aren't passed lag yet
            xys[i] = 0.0;
        }
    
    }

    // We use this command to "unprotect" two objects,
    // it's important we unprotect as many objects as we protected,
    // otherwise we will cause memory leakage.
    UNPROTECT(2);

    // We return the moving averages
    return ys;
}

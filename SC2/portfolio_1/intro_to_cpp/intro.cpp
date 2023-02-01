#include <iostream>
#include <string>
// We will explain what the above mean shortly

// This chunk of code describes how to define a function aswell as some useful syntax, 
// read this once you have gotten to where a_function is called in main!
// Again everything in c++ is typed including functions
// if we didn't want to return anything we could set the type to "void"
double a_function(int arg) 
{
    // We can condition using the following syntax and print out something useful about arg
    if (arg <0){
        std::cout << arg << " is negative." << std::endl;
    }
    else if (arg < 20){
        std::cout << arg << " is between 0 and 20." << std::endl;
    }
    else{
        std::cout << arg << " is bigger than 20" << std::endl;
    }
    
    // We can carry out a for loop as follows which prints out the numbers from 10 to 1
    // We could also write i=i-1 as i-- or i-=1
    for (int i=10; i>0; i=i-1){
        std::cout << i << std::endl;
    }
    
    double out = arg * 0.01;
    return out;
}

// We now define a function main and put all our code inside, 
// everything inside main will then be run when we run this code
int main()
{
    // In C++ everything is typed and so when we define anything we must make sure we type it, 
    // eg. we can define a variable that is the integer 1 as follows:
    int a = 1 ;

    // Notice that we put a ";" after as this is how c++ defines line breaks, 
    // if we wanted to define a string we could do the following:
    std::string our_string = "Hello!";
    std::cout << our_string << std::endl;

    // Here we define a variable our_string that is of type string (std::string) 
    // and then print the output. 
    // We use "#include <string>" to include the string header file 
    // which allows us to use the string type. 
    // We also type "#include <iostream>" which gives us the functionality
    // of getting inputs and giving outputs, 
    // for example if we want to print to the console we need to include this header file. 
    // We can then use std::cout and std::endl as above to print to the console. 
    // Other things we can do with this header file is throw exceptions and return errors, 
    // we will write how to do this below but comment them out 
    // so we don't get any errors when running this code:
    // std::cerr << "Some error" << std::endl; 
    // throw std::runtime_error("Unknown exception");

    // We now call the function "a_function" which we define above. 
    // Go read that portion of code now!
    a_function(10) ;

    return 0;
}
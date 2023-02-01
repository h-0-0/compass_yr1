// We first include header files we will need in this script
#include <iostream>
#include <string>

int main(){
    // Let's first define a variable of type double
    double var = 10.5;
    std::cout << "Var as a double: " << var << std::endl;

    // We can change the type of the variable as follows:
    int var1 = var;
    std::cout << "Var now converted to int: " << var1 << std::endl;

    // If you check the output of running this file 
    // you can see we have lost information 
    // So must be careful when converting types

    // We also have auto:
    // if it is obvious what type a var or fun should have, 
    // you can use "auto" eg,
    auto var2 = var1 ;
    // ^ Here it is "obvious" that var2 should be of type int
    std::cout << "Var assigned using auto: " << var2 << std::endl;
}
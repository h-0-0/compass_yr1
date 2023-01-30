#include <iostream>

#include "timestable.h"

int timestable(int num, int max)
{
    for(int i=1 ; i <= max ; i++)
    {
        std::cout << i*num << std::endl;
    }
    return 0;
}
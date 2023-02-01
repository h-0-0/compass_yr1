// Don't read this for now ...
// ---------------------------------------
#include <iostream>
#include <vector> // We need this to use vectors
#include <map> // We need this to use dictionaries (maps in C++)

// This function creates a vector of vectors (a matrix)
std::vector<std::vector<int>> get_nested_vector(){
    std::vector< std::vector<int> > M;

    // We create a vector of vectors
    // ie. a matrix
    for (int i=1; i<=3; ++i)
    {
        // We first create a row
        std::vector<int> row;

        for (int j=1; j<=3; ++j)
        {
            row.push_back( i * j );
        }

        // Now we save the row to the matrix
        M.push_back(row);
    }
    return M;
}

// ---------------------------------------
// START READING HERE!
int main(){
    // First we will talk vectors,
    // a vector is a container that can store multiple values of only one type

    // We can create a vector that can hold integers as so:
    std::vector<int> v;

    // We can add values onto the end as so:
    v.push_back( 4 );
    v.push_back( 2.5 ); 
    // Note for the second command 2.5 will be converted to an integer
    // Let's now print out our vector,
    // note we use a range-based for loop here (needs C++11 or later)
    std::cout << "Our vector: { " ;
    for (auto i: v)
        std::cout << i << ' ';
    std::cout << "}" << std::endl;

    // We can get the size of the vector
    int length = v.size();
    std::cout << "The vector has length: " << length << std::endl;

    // Is also possible to nest vectors inside vectors
    auto M = get_nested_vector();
    // go to the get_nested_vector to see how we create and fill a matrix!
    std::cout << "A nested vector: " << std::endl;
    // this prints the matrix out
    for (int i=0; i<3; ++i)
    {
        for (int j=0; j<3; ++j)
        {
            std::cout << M[i][j] << " ";
        }

        std::cout << std::endl;
    }

    // ----------
    // Now lets talk dictionaries,
    // dictionaries in C++ are called maps,
    // they are containers that store key value pairs,
    // note that keys must be of the same type and values must be of the same type

    // We create a map that stores strings and where the keys are strings
    std::map<std::string, std::string> dict;

    // we can add some items to the map
    dict["key_1"] = "value_1";
    dict["key_2"] = "value_2";

    // Now we can loop through all of the key-value pairs
    // in the map and print them out
    for ( auto item : dict )
    {
        //item.first is the key
        std::cout << "Getting the key using .first :" 
        << item.first << " , ";

        //item.second is the value
        std::cout << "Getting the key using .second :" 
        << item.second << std::endl;
    }

    // Finally we can lookup values by key
    std::cout << "What's the value associated to key_1?: " << dict["key_1"] 
              << std::endl;


    return 0; 
}

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <regex>
#include "part1.h"

 
// A function that reads a text file and returns it as a string
// we will use this to read in the odyssey text file
std::string read_file(const std::string &filename)
{
    std::ifstream file(filename);
    std::string result;

    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            // Replace punctuation with spaces
            std::replace(line.begin(), line.end(), ',', ' ');
            std::replace(line.begin(), line.end(), '.', ' '); 
            std::replace(line.begin(), line.end(), ';', ' ');
            std::replace(line.begin(), line.end(), ':', ' ');
            std::replace(line.begin(), line.end(), '!', ' ');
            std::replace(line.begin(), line.end(), '?', ' ');
            std::replace(line.begin(), line.end(), '"', ' ');
            std::replace(line.begin(), line.end(), '\'', ' ');
            std::replace(line.begin(), line.end(), '(', ' ');
            std::replace(line.begin(), line.end(), ')', ' ');
            std::replace(line.begin(), line.end(), '-', ' ');
            line = std::regex_replace(line, std::regex("—"), " ");
            result += line;
        }
    }

    return result;
}

// A function that takes a long string and turns it into a vector of strings
// we will use this to split the odyssey text file into a vector of words
std::vector<std::string> split_string(const std::string &str)
{
    std::vector<std::string> result;
    std::string word;
    bool prev_space = false;
    for (auto &c : str)
    {
        if (c == ' ')
        {
            // this makes sure we don't add empty strings to the vector
            if (!prev_space)
            {
                result.push_back(word);
                word = "";
            }
            prev_space = true;
        }
        else
        {
            word += c;
            prev_space = false;
        }
    }
    result.push_back(word);
    return result;
}

// A function that checks if the word given is "the"
int contains_the(const std::string &word)
{
    int out = (word == "the") || (word == "The") || (word == "THE");
    return out;
} 

// A function that check if the first word is "the" and the second is "gods"
int contains_the_gods(const std::string &word1, const std::string &word2)
{
    int bool1 = (word1 == "the") || (word1 == "The") || (word1 == "THE");
    int bool2 = (word2 == "gods") || (word2 == "GODS") || (word2 == "Gods");
    int out = (bool1 && bool2);
    return out;
}

// A generic function that given two words will check if the next two words are the same
// we will use this to show how to use lambda functions
int contains(const std::string &word1, const std::string &word2, const std::string &word3, const std::string &word4)
{
    // Let's turn all the words to lower case
    std::string word1_lower = word1;
    std::string word2_lower = word2;
    std::string word3_lower = word3;
    std::string word4_lower = word4;
    std::transform(word1_lower.begin(), word1_lower.end(), word1_lower.begin(), ::tolower);
    std::transform(word2_lower.begin(), word2_lower.end(), word2_lower.begin(), ::tolower);
    std::transform(word3_lower.begin(), word3_lower.end(), word3_lower.begin(), ::tolower);
    std::transform(word4_lower.begin(), word4_lower.end(), word4_lower.begin(), ::tolower);
    int out = ((word1_lower == word3_lower) && (word2_lower == word4_lower));
    return out;
}

// We will use all these contains functions in conjunction 
// with our map function when we get to main()

// We will now define a generic version of map,
// this map function can map a function to an arbitrary number of vectors.
// The first thing to explain is templating,
// templating is a way to define a function that can work with any type,
// we will use this to define a map function that can work with any type,
// with the template here we are saying that the function we define
// can take some arbitrary type FUNC and some arbitrary number of vectors of type ARGS,
// we build in this functionality by using the "..." variadic template operator.
// Map will return a vector of the type returned by the function we pass in,
// where it will apply the function to the first element of each vector,
// then the second element of each vector, etc.
// note: the function we pass in must take the same number 
// of arguments as the number of vectors we pass in
// We will put this function into action later on in main()
template<class FUNC, class... ARGS>
auto map(FUNC func, const std::vector<ARGS>&... args)
{
    // Here we define the type of the return value of the function we pass in
    typedef typename std::result_of<FUNC(ARGS...)>::type RETURN_TYPE;
    // Here we use a function defined in the file part1
    // what it does is get the minimum size of the vectors we pass in
    int nargs=part1::detail::get_min_container_size(args...);
    // Here we define the vector we will return
    // with the same type as the return type of func
    std::vector<RETURN_TYPE> result(nargs);

    // We now loop over the vectors and apply func to each element
    for (size_t i=0; i<nargs; ++i)
    {
        result[i] = func(args[i]...);
    }

    return result;
}

// A function which takes two int's and returns their sum
// we will use this with reduce in a moment
int sum(int x, int y)
{
    return x+y;
}

// Here we define the reduce function,
// reduce takes a function and a vector of values,
// it then applies the function to the first two elements of the vector,
// then applies the function to the result of the previous function 
// and the third element of the vector,
// and so on until it has applied the function to all the elements of the vector.
// To implement reduce again we use templating,
// allowing us to define a function that can work with any type,
// here we have to arbitrary types FUNC and T,
// where the function takes in an argument func of type FUNC
// and an argument values of type std::vector<T> (a vector of type T)
template<class FUNC, class T>
T reduce(FUNC func, const std::vector<T> &values)
{
    // Deal with the case where the vector is empty
    if (values.empty())
    {
        return T();
    }
    // If the vector is not empty...
    else
    {
        // We start with the first element of the vector
        T result = values[0];
        // We then loop over the rest of the vector
        // and apply the function to the result and the next element
        for (size_t i=1; i<values.size(); ++i)
        {
            result = func(result, values[i]);
        }

        return result;
    }
}

// Let's now use the functions we explained above
int main(int argc, char **argv)
{
    // First we are going to read in the odyssey text file 
    // and we are going to turn it into a vector of strings
    auto words = read_file("odyssey.txt");
    auto vec_words = split_string(words);
    // We are now going to print out the first 10 words
    std::cout << "The First 10 words in the Odyssey:" << std::endl;
    for (int i=0; i<10; ++i)
    {
        std::cout << vec_words[i] << std::endl;
    }

    // Let's now use our map function to find every occurence of the word "the"
    // this will return a vector of bools
    auto has_the = map(contains_the, vec_words);
    // We can now print out the first 10 bools
    std::cout << "" << std::endl;
    std::cout << "Which of the first ten words contain the word the?" << std::endl;
    for (int i=0; i<10; ++i)
    {
        std::cout << has_the[i] << std::endl;
    }

    // What if we now wanted to find out all the places where the word the is followed by gods?
    // we can do this by using our map function again 
    // First let's create a vector where the words are shifted by one to the left
    std::string temp = vec_words[0];
    std::vector<std::string> vec_words_shift(vec_words.size());
    for (int i=0; i<vec_words.size()-1; ++i)
    {
        vec_words_shift[i] = vec_words[i+1];
    }
    vec_words_shift[vec_words.size()-1] = temp;

    // Now we can use our map function
    std::vector<int> has_the_gods = map(contains_the_gods, vec_words, vec_words_shift);
    // We can now print out the first 10 bools
    std::cout << "" << std::endl;
    std::cout << "Which of the first ten words contain the word the followed by gods?" << std::endl;
    for (int i=0; i<10; ++i)
    {
        std::cout << has_the_gods[i] << std::endl;
    }

    // Now let's do the same but we will use a lambda function
    // a lambda function allows us to define a function on the fly
    // recall the generic contain function we defined above
    // let's write a lambda function that does the same thing as contains_the_gods
    auto contains_the_gods_lambda = [](const std::string &word1, const std::string &word2)
    {
        return contains(word1, word2, "the", "gods");
    };
    // Here we name the lambda function contains_the_gods_lambda
    // we can also use it in-line so we don't have to name it
    // let's do that now
    has_the_gods = map([](const std::string &word1, const std::string &word2)
    {
        return contains(word1, word2, "the", "gods");
    }, vec_words, vec_words_shift);
    // Let's check this is the same as before:
    std::cout << "" << std::endl;
    std::cout << "Which of the first ten words contain the word the followed by gods? this time we used a lambda function" << std::endl;
    for (int i=0; i<10; ++i)
    {
        std::cout << has_the_gods[i] << std::endl;
    }
    // Note that there is something called capture in lambda functions
    // this allows us to capture variables from the outside scope
    // the three main ways to capture are by value (writing [=]), 
    // by reference (writing [&]), and to not capture at all! (writing [])

    // Let's now find out the number of times the phrase "the gods" appears in the oddyssey
    // we can do this by using our reduce function
    int n_the_gods = reduce(sum, has_the_gods);
    // Let's print out the result
    std::cout << "" << std::endl;
    std::cout << "The number of times the phrase the gods appears in the oddyssey is: " << n_the_gods << std::endl;

    return 0;
}

/*

This is a program designed to calculate a simple 
statistical analysis on two columns of data.

*/


#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace std;

// Function to get the sum of one vector
double bostonSum(vector<double> b)
{
    // Initializing sum variable
    double sum = 0;

    // Adding everything in the vector
    for (int i = 0; i < b.size(); i++)
    {
        sum += b[i];
    }

    // Return output of sum
    return sum;
}

// Funtion to get the mean of one vector
double bostonMean(vector<double> b)
{
    // Initializing sum variable
    double sum = 0;

    // Adding everything in the vector
    for (int i = 0; i < b.size(); i++)
    {
        sum += b[i];
    }

    // Calculating mean
    double mean = sum / b.size();

    // Returning calculated mean
    return mean;
}

// Function to get median of a vector
double bostonMedian(vector<double> b)
{
    // Intializing med (median) variable
    double med = 0;

    // Sort vector from smallest to biggest
    sort(b.begin(), b.end());


    // Checking if the vector is even. If it is, 
    // mean the two middle numbers and assign it to median.
    // If not just show the median.
    if (b.size() % 2 == 0)
    {
        med = (b[(b.size() / 2) - 1] + b[b.size() / 2]) / 2;
    }

    else
    {
        med = b[b.size() / 2];
    }

    // Return calculated median
    return med;
}

// Funtion to get range of a vector
double bostonRange(vector<double> b)
{
    // Sort the vector smallest to biggest
    sort(b.begin(), b.end());

    // Return the difference between the first and last elements in the vector
    return b[b.size() - 1] - b[0];
}

// Funtion to find the minimum of a vector
double bostonMin(vector<double> b)
{
    // Sort the vector from smallest to biggest
    sort(b.begin(), b.end());

    // Return the first value
    return b[0];
}

// Funtion to find the maximum of a vector
double bostonMax(vector<double> b)
{
    // Sort the vector from smallest to biggest
    sort(b.begin(), b.end());

    // Return the last value
    return b[b.size() - 1];
}

// Function to calculate covariance of 2 vectors
double bostonCovar(vector<double> b, vector<double> c)
{
    // Calculate mean of x and y
    // Initialize numerator, calculate denominator
    double xm = bostonMean(b);
    double ym = bostonMean(c);
    double num = 0, den = b.size() - 1;

    // Calculating numerator
    for (int i = 0; i < b.size(); i++)
    {
        num += (b[i] - xm) * (c[i] - ym);
    }

    // Calculating covariance
    double cov = num / den;

    // Return covariance
    return cov;
}

// Calculate correlation of 2 vectors
double bostonCor(vector<double> b, vector<double> c)
{
    // Calculate variance of x and y separately
    // Calculate sigma of vectors x and y
    double varx = bostonCovar(b, b);
    double vary = bostonCovar(c, c);
    double sigx = sqrt(varx);
    double sigy = sqrt(vary);

    // Calculate correlation
    double corr = bostonCovar(b, c) / (sigx * sigy);

    // Return correlation
    return corr;
}

// Main function to open and close file, 
// and to also call each function to come up with stats.
int main()
{
    // Initializing input and strings used for opening files
    // Initializing vectors to be used for functions
    ifstream inFS;
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    // Opening "Boston.csv" file
    cout << "Opening file..." << endl << endl;

    inFS.open("Boston.csv");

    // If the file isn't open let the end user know
    if (!inFS.is_open())
    {
        cout << "Couldn't open file." << endl;
        return 1;
    }

    // Read line 1 and prepare the heading
    cout << "Read line 1" << endl;
    getline(inFS, line);

    cout << "heading: " << line << endl;
  
    // Calculate number of observations from file
    // and input each column value into the vector
    int numObsv = 0;
    while (inFS.good())
    {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObsv) = stof(rm_in);
        medv.at(numObsv) = stof(medv_in);

        numObsv++;
    }

    // Size vectors to match the number of observations
    rm.resize(numObsv);
    medv.resize(numObsv);
    
    // Let the end user know the new length
    cout << "new length " << rm.size() << endl << endl;

    // Close file
    cout << "Closing file..\n\n\n";
    inFS.close();


    // Output the statistical results
    // Side note: I think it's important to include the max and min for the range
    cout << "Number of records: " << numObsv << endl;

    cout << "\n***STATS FOR RM***" << endl << endl;
    cout << "Sum: " << bostonSum(rm) << endl;
    cout << "Mean: " << bostonMean(rm) << endl;
    cout << "Median: " << bostonMedian(rm) << endl;
    cout << "Range: " << bostonRange(rm) << "    (Max: " << bostonMax(rm) << "  Min: " << bostonMin(rm) << ")" << endl;

    cout << "\n***STATS FOR MEDV***" << endl << endl;
    cout << "Sum: " << bostonSum(medv) << endl;
    cout << "Mean: " << bostonMean(medv) << endl;
    cout << "Median: " << bostonMedian(medv) << endl;
    cout << "Range: " << bostonRange(medv) << "    (Max: " << bostonMax(medv) << "  Min: " << bostonMin(medv) << ")" << endl;

    cout << "\n\nCovariance = " << bostonCovar(rm, medv) << endl;

    cout << "Correlation = " << bostonCor(rm, medv) << endl << endl << endl;

    // End Program
    cout << "\nProgram terminated.\n";

    return 0;
}


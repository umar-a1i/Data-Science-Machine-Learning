
/*
	Umar Ali-Salaam, Cory Pekkala
 
	This is a program to calculate Naive Bayes Classification on survival data from the Titanic

*/

#include<iostream>
#include<string>
#include<fstream>
#include<cmath>
#include<vector>
#include<algorithm>
#include<numeric>
#include<iomanip>
#include<chrono>

using namespace std;
using namespace std::chrono;

// w/o header there are 1,046 rows in .csv
#define MAX_LEN 1046

// Calculating prior probabilities
template <typename T>
double prior_prob(vector<T> survive)
{
	// Prep variables
	double surv = 0;
	double survN = 0;

	// Count who survived and who didn't
	for (int i = 0; i < survive.size(); i++)
	{

		if (survive[i] == 1)
		{
			surv++;
		}

		if (survive[i] == 0)
		{
			survN++;
		}
	}

	// Outputting Data
	cout << "\nSurvived" << setw(16) << "Not Survived\n"
		<< (surv / survive.size()) << setw(16) << (survN / survive.size()) << endl;

	return 0;
}

// Function for conditional class probability
template <typename T>
double class_prob(vector<T> survive, vector<T> pclass)
{
	// Prep variables
	double n = 0, m = 0, p = 0, q = 0;
	double a = 0, b = 0, c = 0, d = 0;

	double class1 = 0, class2 = 0, class3 = 0;
	double class1n = 0, class2n = 0, class3n = 0;

	// Count survive and not survive for all 3 classes
	for (int i = 0; i < survive.size(); i++)
	{

		if (survive[i] == 1)
		{
				
			if (pclass[i] == 1)
			{
				n++;
			}
			if (pclass[i] == 2)
			{
				m++;
			}
			if (pclass[i] == 3)
			{
				p++;
			}
			q++;
		}

		if (survive[i] == 0)
		{

			if (pclass[i] == 1)
			{
				a++;
			}
			if (pclass[i] == 2)
			{
				b++;
			}
			if (pclass[i] == 3)
			{
				c++;
			}
			d++;
		}
		
	}

	// Calculate class survival and output the data
	class1 = n / q;
	class2 = m / q;
	class3 = p / q;

	class1n = a / d;
	class2n = b / d;
	class3n = c / d;

	cout << "\n\nClass\n" << setw(25) << "Class 1" << setw(16) << "Class 2" << setw(17) << "Class 3\n";

	cout << "Survived: " << setw(16) << class1 << setw(16) << class2 << setw(16) << class3 << endl;
	cout << "Not Survived: " << setw(12) << class1n << setw(16) << class2n << setw(16) << class3n << endl;

	return 0;

}

// Function for conditional sex probability
template <typename T>
double sex_prob(vector<T> survive, vector<T> sex)
{
	// Prep variables for tests
	double n = 0, m = 0, p = 0;
	double a = 0, b = 0, c = 0;

	double female = 0, male = 0;
	double femaleN = 0, maleN = 0;

	// Count females and males who survived and didn't survive
	for (int i = 0; i < survive.size(); i++)
	{

		if (survive[i] == 1)
		{

			if (sex[i] == 0)
			{
				n++;
			}
			if (sex[i] == 1)
			{
				m++;
			}
			p++;
		}

		if (survive[i] == 0)
		{

			if (sex[i] == 0)
			{
				a++;
			}
			if (sex[i] == 1)
			{
				b++;
			}
			c++;
		}

	}

	// Calculate and output female and male data
	female = n / p;
	male = m / p;

	femaleN = a / c;
	maleN = b / c;

	cout << "\n\nSex\n" << setw(24) << "Female" << setw(15) << "Male\n";

	cout << "Survived: " << setw(16) << female << setw(16) << male << endl;
	cout << "Not Survived: " << setw(12) << femaleN << setw(16) << maleN << endl;

	return 0;

}


// Function for conditional age probability
template <typename T>
double age_prob(vector<T> survive, vector<T> age)
{
	// Prep variables for tests
	double n = 0, p = 0, sum = 0, sumN = 0, sd = 0, sdN = 0;

	// Collect sums for survived and not survived
	// Count survived and not survived
	for (int i = 0; i < survive.size(); i++)
	{

		if (survive[i] == 1)
		{
			n++;
			sum += age[i];
		}

		if (survive[i] == 0)
		{
			p++;
			sumN += age[i];
		}

	}

	// Calculate mean for survived and not survived
	double mean = sum / n;
	double meanN = sumN / p;

	// Calculating numerator for standard deviation of survived and not survived
	for (int i = 0; i < survive.size(); i++)
	{

		if (survive[i] == 1)
		{
			sd += (age[i] - mean) * (age[i] - mean);
		}

		if (survive[i] == 0)
		{
			sdN += (age[i] - meanN) * (age[i] - meanN);
		}

	}

	// Calculating standard deviation as if it were sample data not population
	double std = sqrt(sd / (n));
	double stdN = sqrt(sdN / (p));

	// Outputting data
	cout << "\n\nAge\n" << setw(22) << "Mean" << setw(21) << "Std. Dev\n";

	cout << "Survived: " << setw(16) << mean << setw(16) << std << endl;
	cout << "Not Survived: " << setw(12) << meanN << setw(16) << stdN << endl;

	return 0;
}

// Function for raw probability
template <typename T>
double raw_prob(vector<T> survive, vector<T> pclass, vector<T> sex, vector<T> age)
{
	// Prep variables
	double surv = 0;
	double survN = 0;

	// Count who survived and who didn't
	for (int i = 0; i < survive.size(); i++)
	{

		if (survive[i] == 1)
		{
			surv++;
		}

		if (survive[i] == 0)
		{
			survN++;
		}
	}

	double surv1 = (surv / survive.size()), surv2 = (survN / survive.size());

	// Class

	// Prep variables
	double n = 0, m = 0, p = 0, q = 0;
	double a = 0, b = 0, c = 0, d = 0;

	double class1 = 0, class2 = 0, class3 = 0;
	double class1n = 0, class2n = 0, class3n = 0;

	// Count survive and not survive for all 3 classes
	for (int i = 0; i < survive.size(); i++)
	{

		if (survive[i] == 1)
		{

			if (pclass[i] == 1)
			{
				n++;
			}
			if (pclass[i] == 2)
			{
				m++;
			}
			if (pclass[i] == 3)
			{
				p++;
			}
			q++;
		}

		if (survive[i] == 0)
		{

			if (pclass[i] == 1)
			{
				a++;
			}
			if (pclass[i] == 2)
			{
				b++;
			}
			if (pclass[i] == 3)
			{
				c++;
			}
			d++;
		}

	}

	// Calculate class survival and output the data
	class1 = n / q;
	class2 = m / q;
	class3 = p / q;

	class1n = a / d;
	class2n = b / d;
	class3n = c / d;

	// Gender

	// Prep variables for tests
	double n1 = 0, m1 = 0, p1 = 0;
	double a1 = 0, b1 = 0, c1 = 0;

	double female = 0, male = 0;
	double femaleN = 0, maleN = 0;

	// Count females and males who survived and didn't survive
	for (int i = 0; i < survive.size(); i++)
	{

		if (survive[i] == 1)
		{

			if (sex[i] == 0)
			{
				n1++;
			}
			if (sex[i] == 1)
			{
				m1++;
			}
			p1++;
		}

		if (survive[i] == 0)
		{

			if (sex[i] == 0)
			{
				a1++;
			}
			if (sex[i] == 1)
			{
				b1++;
			}
			c1++;
		}

	}

	// Calculate and output female and male data
	female = n1 / p1;
	male = m1 / p1;

	femaleN = a1 / c1;
	maleN = b1 / c1;

	// Age

	// Reprepping variables for tests
	double n2 = 0, p2 = 0, sum = 0, sumN = 0, sd = 0, sdN = 0;

	// Collect sums for survived and not survived
	// Count survived and not survived
	for (int i = 0; i < survive.size(); i++)
	{

		if (survive[i] == 1)
		{
			n2++;
			sum += age[i];
		}

		if (survive[i] == 0)
		{
			p2++;
			sumN += age[i];
		}

	}

	// Calculate mean for survived and not survived
	double mean = sum / n2;
	double meanN = sumN / p2;

	// Calculating numerator for standard deviation of survived and not survived
	for (int i = 0; i < survive.size(); i++)
	{

		if (survive[i] == 1)
		{
			sd += (age[i] - mean) * (age[i] - mean);
		}

		if (survive[i] == 0)
		{
			sdN += (age[i] - meanN) * (age[i] - meanN);
		}

	}

	// Calculating standard deviation as if it were sample data not population
	double std = (sd / (n2 - 1));
	double stdN = (sdN / (p2 - 1));

	double pi = 3.14159265358979323846;
	double e = 2.718;


		// Raw probability
		double calc_age = (1 / sqrt(2 * pi * std)) * pow(e, 0((sd) / (2 * std)));
		double calc_ageN = (1 / sqrt(2 * pi * stdN)) * pow(e, 0((sdN) / (2 * stdN)));

		double numS = ((class1 * class2 * class3) * (male * female) * (calc_age) * surv1);
		double numN = ((class1n * class2n * class3n) * (maleN * femaleN) * (calc_ageN) * surv2);
		double den = (numS)+(numN);

		// Outputting Data
		cout << "\nSurvived" << setw(17) << "Not Survived\n";
		cout << (numS / den) << setw(16) << (numN / den) << endl;



	return 0;
}



int main() {

	ifstream infile;
	string line;

	// Prep vectors for each class, 800 for train 246 for test
	vector<int> survivedTrain(800);
	vector<int> survivedTest(246);
	vector<int> sexTrain(800);
	vector<int> sexTest(246);
	vector<int> ageTrain(800);
	vector<int> ageTest(246);
	vector<int> classTrain(800);
	vector<int> classTest(246);

	// Create temporary vector of size 5 since there are 5 columns
	vector<string> temp(5);

	// Open file
	infile.open("titanic_project.csv");

	// Read in headers first
	getline(infile, line);

	// Fill each vector for each class
	int row = 0;
	char delim;
	while (infile.good() && row < MAX_LEN) {
		for (int i = 0; i < 5; i++) {
			delim = ((i == 4) ? '\n' : ',');
			getline(infile, temp.at(i), delim);
		}
		if (row < 800) {
			// Fill training vectors for each class
			classTrain.at(row) = stoi(temp.at(1));
			survivedTrain.at(row) = stoi(temp.at(2));
			sexTrain.at(row) = stoi(temp.at(3));
			ageTrain.at(row) = stoi(temp.at(4));
		}
		else {
			// Fill testing vectors for each class
			classTest.at(row % 800) = stoi(temp.at(1));
			survivedTest.at(row % 800) = stoi(temp.at(2));
			sexTest.at(row % 800) = stoi(temp.at(3));
			ageTest.at(row % 800) = stoi(temp.at(4));
		}
		row++;

	}

	auto start = high_resolution_clock::now();

	// Outputting probabilities with train data
	// And calling each probability function
	cout << "\n\nTrain Data:\n\n===========================================================\n\nPriori Probabilities:\n";

	prior_prob(survivedTrain);

	cout << "\n\nConditional Probability:";

	class_prob(survivedTrain, classTrain);

	sex_prob(survivedTrain, sexTrain);

	age_prob(survivedTrain, ageTrain);

	cout << "\n\nRaw Probability:\n";

	raw_prob(survivedTrain, classTrain, sexTrain, ageTrain);

	cout << "===========================================================\n\n\n\n\n";



	// Outputting probabilities with test data
	// And calling each probability function
	cout << "Test Data:\n\n===========================================================\n\nPriori Probabilities:\n";

	prior_prob(survivedTest);

	cout << "\n\nConditional Probability:";

	class_prob(survivedTest, classTest);

	sex_prob(survivedTest, sexTest);

	age_prob(survivedTest, ageTest);

	cout << "\n\nRaw Probability:\n";

	raw_prob(survivedTest, classTest, sexTest, ageTest);

	cout << "===========================================================\n\n\n\n\n";

	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<seconds>(stop - start);

	cout << "Time taken to compute for algorithm to run:\t" << duration.count() << " seconds" << endl;

	// Close file
	infile.close();


	return 0;
}
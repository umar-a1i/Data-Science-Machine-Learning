/*
	Umar Ali-Salaam, Cory Pekkala
	
	This is a program to calculate the Logistic Regression classfication on the Titanic data set.

*/

#include<iostream>
#include<string>
#include<fstream>
#include<cmath>
#include<vector>
#include<typeinfo>
#include<map>
#include<algorithm>
#include<chrono>

using namespace std;
using namespace std::chrono;

// macro for sigmoid function ~ returns 1/1+e^-z value per instance
#define SIGMOID(z) (1/(1+exp(-1*z))) 

// w/o header there are 1,046 rows in .csv
#define MAX_LEN 1046

/*===========================================================================================*/

//debug: write all vectors from input now handled to console ~ all will be of same length
template <typename T>
void displayVector(vector<T> v) {
	for (T e : v) {
		cout << e << endl;
	}
}

template <typename T>
void displayMatrix(vector<vector<T> > M) {
	for (int r = 0; r < M.size(); r++) {
		for (int c = 0; c < M[r].size(); c++)
			cout << M[r][c] << " ";
		cout << endl;
	}
}

//debug: write all vectors from input now handled to console ~ all will be of same length
void displayVectors(
	vector<int> v1,
	vector<int> v2
) {
	for (int i = 0; i < v1.size(); i++)
		cout << v1.at(i) << "," << v2.at(i) << endl;
}


/*NOTE: LOOK INTO MAZIDI ML BOOK PAGES 116-118 for R-CODE EXAMPLES OF LOGISTIC REGRESSION*/

// computes the matrix dot product between the train matrix & weight matrix
template <typename T>
vector<double> MATRIX_DOT_PRODUCT(vector<vector<int> > m1, vector<T> m2) {
	// m1 is the train matrix
	// m2 is the weight matrix
	// train is nx2 and weight is 2x1 ~  nx2 * 2x1  will give nx1 matrix ~ n-lengthed vector
	vector<double> r;
	double rowprod;
	int nrow = m1.size();
	for (int i = 0; i < nrow; i++) {
		rowprod = (m1[i].at(0) * m2.at(0)) + (m1[i].at(1) * m2.at(1));
		r.push_back(rowprod);
	}
	return r;
}


// computes the matrix transpose for the logtrain matrix
vector<vector<int> > MATRIX_TRANSPOSE(vector<vector<int> > logtrainMatrix) {
	// input is nx2 ~ input^T is 2xn so return should have 2 rows, each w/n-cols
	vector<vector<int> > T;
	T.push_back(vector<int>());
	T.push_back(vector<int>());
	for (int i = 0; i < logtrainMatrix.size(); i++) {
		T[0].push_back(logtrainMatrix[i].at(0));
		T[1].push_back(logtrainMatrix[i].at(1));
	}
	return T;
}

// takes the sigmoid of all values in a vector
vector<double> SIGMOID_VECTOR(vector<double> v) {
	vector<double> n;
	for (int x : v) n.push_back(SIGMOID(x));
	return n;
}

// vector operation ~ add values of two vectors ~ bias towards size of first input vector
template <typename T, typename Q>
vector<double> VECTOR_ADD(vector<T> v1, vector<Q> v2) {
	vector<double> r;
	for (int i = 0; i < v1.size(); i++) r.push_back(v1[i] + v2[i]);
	return r;
}

// vector operation ~ subtract values of two vectors ~ bias towards size of first input vector
template <typename T, typename Q>
vector<double> VECTOR_SUB(vector<T> v1, vector<Q> v2) {
	vector<double> r;
	for (int i = 0; i < v1.size(); i++) r.push_back(v1[i] - v2[i]);
	return r;
}

// vector operation ~ multiply values of two vectors ~ bias towards size of first input vector
template <typename T, typename Q>
vector<double> VECTOR_PRODUCT(vector<T> v1, vector<Q> v2) {
	vector<double> r;
	for (int i = 0; i < v1.size(); i++) r.push_back(v1[i] * v2[i]);
	return r;
}

// vector operation ~ scale values of a vector by scalar 
template <typename T, typename Q>
vector<double> VECTOR_SCALE(vector<T> v, Q scalar) {
	vector<double> r;
	for (T x : v) r.push_back(x * scalar);
	return r;
}

// calculates probability vector - as the matrix product between the logtrainMatrix & weight matrices, then takes sigmoid of all values.
vector<double> PROBABILITY_VECTOR(vector<vector<int> > logtrainMatrix, vector<double> weights) {
	// logtrain ~ dim nx2 
	// weights ~  dim 2x1 
	// logtrain * weights ~ nx1 matrix (n-lengthed vector for proba)
	return SIGMOID_VECTOR(MATRIX_DOT_PRODUCT(logtrainMatrix, weights));
}

vector<double> LOG_ODDS_VECTOR(vector<vector<int> > logtestMatrix, vector<double> optimalWeights) {
	// logtestMatrix ~ dim nx2 
	// weights ~  dim 2x1 
	// logtestMatrix * weights ~ nx1 matrix (n-lengthed vector for log odds)
	return MATRIX_DOT_PRODUCT(logtestMatrix, optimalWeights);
}

// calculate error vector ~  for i in y_actual,y_pred -  error_i = y_actual_i - y_pred_i
vector<double> ERROR_VECTOR(vector<int> y_actual, vector<double> y_pred) {
	return VECTOR_SUB(y_actual, y_pred);
}

// gradient descent ~ returns an optimal weight vector after n_iter amount of iterations
vector<double> GRADIENT_DESCENT(vector<vector<int> > logtrainMatrix, vector<int> survived, vector<double> weights, int n_iter=1000) {
	double learning_rate = 0.001;
	vector<double> prob_vector;
	vector<double> error;
	for (int i = 0; i < n_iter; i++) {
		prob_vector = PROBABILITY_VECTOR(logtrainMatrix, weights);
		error = ERROR_VECTOR(survived, prob_vector);
		weights = VECTOR_ADD(VECTOR_SCALE(MATRIX_DOT_PRODUCT(MATRIX_TRANSPOSE(logtrainMatrix), error), learning_rate), weights);
	}
	return weights;
}

// converts log-odds to probabilities 
vector<double> proba_test(vector<double> logOddsTest) {
	vector<double> probs;
	for (double lo : logOddsTest)
		probs.push_back((exp(lo)) / (1.0 + exp(lo)));
	return probs;
}

// returns a vector of predictions based on each value in test for sex ~ ypred_i=w0+w1(xi)
vector<int> predict(vector<double> sexTestProbabilities) {
	vector<int> sexTestPredictions;
	double px;
	int pred;
	for (int i = 0; i < sexTestProbabilities.size(); i++) {
		px = sexTestProbabilities[i];
		pred = ((px > 0.5) ? 1 : 0);
		sexTestPredictions.push_back(pred);
	}
	return sexTestPredictions;
}

// calculate the accuracy between what was predicted based on weights applied to sex test data [see predict function] & the ACTUAL survived test values
double accuracy(vector<int> sexTestPredictions, vector<int> survivedTest) {
	int total = survivedTest.size();
	int actual, predicted;
	double score = 0.0;
	for (int i = 0; i < total; i++) {
		actual = survivedTest[i];
		predicted = sexTestPredictions[i];
		score += (actual==predicted);
	}
	return score / total;
}

// calculate the sensitivity between what was predicted based on weights applied to sex test data [see predict function] & the ACTUAL survived test values
// TP/(TP+FN)
double sensitivity(vector<int> sexTestPredictions, vector<int> survivedTest) {
	double tp = 0.0, fn = 0.0;
	int pred, act;
	for (int i = 0; i < sexTestPredictions.size(); i++) {
		pred = sexTestPredictions[i];
		act = survivedTest[i];
		if (pred == 1 && act == 1)
			tp += 1.0;
		if (pred == 0 && act == 1)
			fn += 1.0;
	}
	return tp / (tp+fn);
}

// calculate the specificity between what was predicted based on weights applied to sex test data [see predict function] & the ACTUAL survived test values
// TN/(TN+FP)
double specificity(vector<int> sexTestPredictions, vector<int> survivedTest) {
	double tn = 0.0, fp = 0.0;
	int pred, act;
	for (int i = 0; i < sexTestPredictions.size(); i++) {
		pred = sexTestPredictions[i];
		act = survivedTest[i];
		if (pred == 0 && act==0)
			tn += 1.0;
		if (pred == 1 && act == 0)
			fp += 1.0;
	}
	return tn / (tn+fp);
}

void classification_report(vector<int> sexTestPredictions, vector<int> survivedTest, vector<double> optimalWeights) {
	cout << "w0 = " << optimalWeights[0] << "\tw1 = " << optimalWeights[1] << endl << endl;
	double acc = accuracy(sexTestPredictions, survivedTest);
	double sens = sensitivity(sexTestPredictions, survivedTest);
	double spec = specificity(sexTestPredictions, survivedTest);
	cout << "accuracy = " << acc << endl;
	cout << "sensitivity = " << sens << endl;
	cout << "specificity = " << spec << endl;
}


int main() {
	
	ifstream infile;
	string line;
	
	// we're only using sex to predict survived ~ so we'll only use these columns
	vector<int> survivedTrain(800); 
	vector<int> survivedTest(246);
	vector<int> sexTrain(800); 
	vector<int> sexTest(246);
	
	// create temporary vector of size 5 since there are 5 columns
	vector<string> temp(5);
	
	// try to open the file 
	infile.open("titanic_project.csv");
	
	// read in headers first
	getline(infile, line);
	
	// read in each column value in csv format & fill vectors
	int row = 0;
	char delim;
	while(infile.good() && row < MAX_LEN) {
		for(int i=0; i<5; i++) {
			delim=((i==4) ? '\n' : ',');
			getline(infile,temp.at(i),delim);
		}
		if (row < 800) {
			// fill training vectors for survived & sex
			survivedTrain.at(row) = stoi(temp.at(2));
			sexTrain.at(row) = stoi(temp.at(3));
		} 
		else {
			// fill testing vectors for survived & sex
			survivedTest.at(row % 800) = stoi(temp.at(2));
			sexTest.at(row % 800) = stoi(temp.at(3));
		}
		row++;
	}

	/*
	Logistic regression set up steps
	1st set w1=w0=1
	2nd create matrix (2d vector) - 1 vector is 'sex' (train), the other is all 1's & w/same nrows
	3rd isolate the 'survived' train
	4th isolate weights vector
	5th define learning 
	*/
	vector<vector<int> > logtrainMatrix;
	int ntrows = sexTrain.size();
	vector<int> ones(ntrows, 1);
	for (int i = 0; i < ntrows; i++) {
		logtrainMatrix.push_back(vector<int>());
		logtrainMatrix[i].push_back(ones.at(i));
		logtrainMatrix[i].push_back(sexTrain.at(i));
		//logtrainMatrix[i].push_back(ones.at(i));
	}

	// create augmented test matrix (w/sex test ~ 1|test)
	vector<vector<int> > logtestMatrix;
	int nterows = sexTest.size();
	for (int i = 0; i < nterows; i++) {
		logtestMatrix.push_back(vector<int>());
		logtestMatrix[i].push_back(ones.at(i));
		logtestMatrix[i].push_back(sexTest.at(i));
	}

	vector<double> weights{ 1.0,1.0 };

	// START CLOCK
	auto start = high_resolution_clock::now();

	// perform gradient descent @ 5000 iterations
	vector<double> optimalWeights = GRADIENT_DESCENT(logtrainMatrix, survivedTrain, weights,5000);

	// get logOdds vector for optimal weights
	vector<double> logOddsTest = LOG_ODDS_VECTOR(logtestMatrix, optimalWeights);

	// get probabilities
	vector<double> probas = proba_test(logOddsTest);

	// get predictions
	vector<int> sexTestPredictions = predict(probas);

	// STOP CLOCK
	auto stop = high_resolution_clock::now();

	// calculate duration
	auto duration = duration_cast<seconds>(stop - start);

	// show classification report for: optimal weights, accuracy, sensitivity, and specificity
	classification_report(sexTestPredictions, survivedTest, optimalWeights);

	// show duration for train/test & gradient descent algorithms
	cout << endl;
	cout << "time taken to compute for algorithm to run (besides metrics reporting):\t" << duration.count() << " seconds" << endl;

	infile.close();
	
	
	return 0;
}

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include "KDT.h"
#include <unordered_map>

using namespace std;

vector<Point> readData(const char* fileName, bool withLabel) {
    vector<Point> result;
    string line;
    int numDim = 0;
    double data = 0;

    ifstream in;
    in.open(fileName, ios::binary);

    // count number of dimensions
    getline(in, line);
    stringstream lineStream(line);
    while (lineStream >> data) numDim++;
    if (withLabel) numDim--;

    //Resets the stream to beginning of file
    in.seekg(0, ios_base::beg);

    // read the data and convert them to data points
    while (!in.eof()) {
        vector<double> features;
        int label = 0;
        for (int i = 0; i < numDim; i++) {
            in >> data;
            if (!in.fail()) {
                features.push_back(data);
            }
        }
        if (withLabel) {
            in >> label;
        }
        if (!in.fail()) {
            result.push_back(Point(features, label));
        }
    }

    in.close();
    return result;
}

// Find the most frequent label in the given set of points
int mostFreqLabel(vector<Point>& points) {
    // Initializes Map of Labels
    unordered_map<int, int> map;
    // Loops through vector
    unsigned int i;
    for (i = 0; i < points.size(); i++) {
        map[points[i].label] = map[points[i].label] + 1;
    }

    // Label and Count for largest counted label
    int largestLabel = 0;
    int largestCount = -1;

    // Check all counts in array to see what is largest
    for (auto& label : map) {
        // Only replaces largestLabel if larger count in current array index
        if (largestCount < label.second) {
            largestLabel = label.first;
            largestCount = label.second;
            
        }
        else if (label.second == largestCount) {
            if (largestLabel > label.first) {
                largestLabel = label.first;
            }       
        }
    }

    // return most popular label
    return largestLabel;
}

// Tests data against itself using KNN for k = 1, 3, 5, 9, 15
void testKNNAgainstSelf(vector<Point> data) {
    KDT kdtree;
    kdtree.build(data);
    int kValues[5] = { 1, 3, 5, 9, 15 };
    int predicted = 0;
    double totalMissedLabels;
    for (int k : kValues) {
        cout << "K: " << k << endl;
        predicted = 0;
        totalMissedLabels = 0;
        for (Point& p : data) {
            vector<Point> knn = kdtree.findKNearestNeighbors(p, k);
            predicted = mostFreqLabel(knn);
            if (predicted != p.label) totalMissedLabels++;
        }
        cout << "Validation Error : " << totalMissedLabels / (float)data.size() << endl;
    }
}

// Tests data against other data using KNN
void testKNNAgainstOther(vector<Point> data, vector<Point> other, int k) {
    KDT kdtree;
    kdtree.build(data);
    int predicted = 0;
    double totalMissedLabels;
    cout << "K: " << k << endl;
    predicted = 0;
    totalMissedLabels = 0;
    for (Point& p : other) {
        vector<Point> knn = kdtree.findKNearestNeighbors(p, k);
        predicted = mostFreqLabel(knn);
        if (predicted != p.label) totalMissedLabels++;
    }
    cout << "Validation Error : " << totalMissedLabels / (float)other.size() << endl;
}

// Multiplys data by projection
vector<Point> project(vector<Point> data, vector<Point> projection) {
    Point point = Point(vector<double>(projection[0].features.size(), 0), 0);
    vector<Point> p = vector<Point>(data.size(), point);
    if (data[0].features.size() != projection.size()) {
        cerr << "Matrix Size not Equal" << endl;
        exit(-1);
    }
    for (unsigned int i = 0; i < data.size(); i++) {
        for (unsigned int j = 0; j < projection[0].features.size(); j++) {
            p[i].label = data[i].label;
            for (unsigned int k = 0; k < projection.size(); k++) {
                p[i].features[j] += data[i].features[k] * projection[k].features[j];
            }
        }
    }
    return p;
}
int main() {
    vector<Point> train = readData("PA1train.txt", true);
    vector<Point> test = readData("PA1test.txt", true);
    vector<Point> validate = readData("PA1validate.txt", true);
    vector<Point> projection = readData("projection.txt", false);

    // Part 1
    cout << "Testing PA1train.txt against itself..." << endl;
    testKNNAgainstSelf(train);

    cout << "Testing PA1train.txt against PA1validate.txt..." << endl;
    testKNNAgainstOther(train, validate, 1);
    testKNNAgainstOther(train, validate, 5);
    testKNNAgainstOther(train, validate, 9);
    testKNNAgainstOther(train, validate, 15);

    cout << "Testing PA1train.txt against PA1test.txt..." << endl;
    testKNNAgainstOther(train, test, 1);
    
    cout << "Projecting training data..." << endl;
    vector<Point> trainProjected = project(train, projection);

    cout << "Projecting validation data..." << endl;
    vector<Point> validateProjected = project(validate, projection);

    cout << "Projecting test data..." << endl;
    vector<Point> testProjected = project(test, projection);

    // Part 2
    cout << "Testing PA1train.txt (projected) against itself..." << endl;
    testKNNAgainstSelf(trainProjected);

    cout << "Testing PA1train.txt (projected) against PA1validate.txt (projected)..." << endl;
    testKNNAgainstOther(trainProjected, validateProjected, 1);
    testKNNAgainstOther(trainProjected, validateProjected, 5);
    testKNNAgainstOther(trainProjected, validateProjected, 9);
    testKNNAgainstOther(trainProjected, validateProjected, 15);

    cout << "Testing PA1train.txt (projected) against PA1test.txt (projected)..." << endl;
    testKNNAgainstOther(trainProjected, testProjected, 15);

    return 0;
}

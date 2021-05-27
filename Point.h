
#ifndef Point_h
#define Point_h

#include <math.h>    // pow, abs
#include <vector>    // vector<typename>
#include <string>

using namespace std;

#define DELTA 0.00005

/** The data point with given features and label **/
class Point {
public:
    
    vector<double> features;
    int label;
    int numDim;
    double squareDistToQuery;
    
    /** Constructor that defines a data point with features and certain 
    label */
    Point(vector<double> features, int label) : features(features),
        label(label), numDim(features.size()), squareDistToQuery(0) {}
    
    /** Set the square distance to current query point */
    void setSquareDistToQuery(const Point& queryPoint) {
        squareDistToQuery = 0;
        for (unsigned int i = 0; i < features.size(); i++) {
            squareDistToQuery += ((features[i] - queryPoint.features[i])
                                * (features[i] - queryPoint.features[i]));
        }
    }
    
    /** Equals operator */
    bool operator == (const Point& other) const {
        if (numDim != other.numDim) return false;
        for (int i = 0; i < numDim; i++) {
            if (abs(features[i] - other.features[i]) > DELTA) {
                return false;
            }
        }
        return true;
    }
    
    /** Not-equals operator */
    bool operator != (const Point& other) const {
        return !((*this) == other);
    }
    
};

std::ostream& operator << (std::ostream& out, const Point& data) {
    std::string s = "(";
    for (int i = 0; i < data.numDim - 1; i++) {
        s += to_string(data.features[i]) + ", ";
    }
    s += to_string(data.features[data.numDim - 1]) + ") : " 
         + to_string(data.label);
    out << s;
    return out;
}

/** The comparator used in sorting points */
struct CompareValueAt {
    CompareValueAt() {}
    bool operator() (const Point & p1, const Point & p2) {
        return p1.squareDistToQuery < p2.squareDistToQuery;
    }
};

#endif /* Point_h */

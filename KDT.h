#ifndef KDT_H
#define KDT_H

#include <algorithm> // sort, max, min
#include <math.h>    // pow, abs
#include <vector>    // vector<typename>
#include <deque>     // deque<typename>
#include "Point.h"

using namespace std;

/** A KD tree that can output K nearest neighbors of given query point */
class KDT {
    
protected:
    
    /** Inner class that defines a KDNode with certain data point 
     *  and pointers to its children and parent
     */
    class KDNode {
        
    public:
        
        KDNode * left;
        KDNode * right;
        KDNode * parent;
        Point point;
        
        KDNode(Point point) : point(point) {
            left = right = parent = nullptr;
        }
    };
    
    
    KDNode * root; // root of KD tree
    unsigned int numDim; // number of dimension of data points
    unsigned int k; // number of nearest neighbors to find
    double threshold; // largest distance to query point in current KNN
    unsigned int isize;
    unsigned int iheight;
    
    // a data structure to store current K nearest neighbors
    deque<Point> knn; // double sided stack

public:
    
    /** Default constructor of KD tree */
    KDT() : root(0), numDim(0), k(0), threshold(1000000), isize(0), iheight(0) {}
    
    /** Build the KD tree from the given vector of Point references */
    void build(vector<Point>& points) {
        // Checks size of points
        if (points.size() == 0) return;
        else {
            // Sets isize to size of points
            isize = points.size();
            // sets dimension to points dimension
            numDim = points[0].numDim;
            // Make root from build subtree
            root = buildSubtree(points, 0, points.size(), 0, 0);
        }
    }
    
    /** Find k nearest neighbors of the given query point */
    vector<Point> findKNearestNeighbors(Point queryPoint, unsigned int k) {
        knn.clear();
        if (isize == 0) return vector<Point>(knn.begin(), knn.end());
        this->k = k;
        threshold = 1000000;
        findKNNHelper(root, queryPoint, 0);
        vector<Point> v(knn.begin(), knn.end());
        return v;
    }
    
    // KDT Destructor
    virtual ~KDT() {
        deleteAll(root);
    }
    /** Return the size of the KD tree */
    unsigned int size() {
        return isize;
    }
    
    /** Return the height of the KD tree */
    unsigned int height() {
        return iheight;
    }
    
private:
    
    /** Helper method to recursively build the subtree of KD tree */
    KDNode * buildSubtree(vector<Point>& points, unsigned int start, 
                    unsigned int end, unsigned int d, unsigned int height) {
        // Checks if worth iterating through
        if (end == start) return 0;
        // if branch is taller than iheight change iheight to height
        if (height > iheight) {
            iheight = height;
        }
        // sets median value
        unsigned int median = (start + end)/2;
        // Sorts through vector
        sort(points.begin() + start, points.begin() + end, CompareValueAt());
        // Creates new node to be placed in tree
        KDNode * newNode = new KDNode(points[median]);
        newNode->left = newNode->right = newNode->parent = 0;
        // Checks if worth adding children to current node
        if (end - start == 0) return newNode;
        newNode->left = buildSubtree(points, start, median, (d + 1) % numDim, height + 1);
        newNode->right = buildSubtree(points, median + 1, end, (d + 1) % numDim, height + 1);
        if (newNode->left)
            newNode->left->parent = newNode;
        if (newNode->right)
            newNode->right->parent = newNode;
        // return new node
        return newNode;
    }
    
    /** Helper method to recursively find the K nearest neighbors */
    void findKNNHelper(KDNode* node, const Point& queryPoint, unsigned int d) {
        bool rightChecked = false;
        bool leftChecked = false;
        // base case if no children
        if (!node->left && !node->right) {
            // calculate square distance from query to current node
            node->point.setSquareDistToQuery(queryPoint);

            // add current node to knn
            if (knn.size() < k || node->point.squareDistToQuery < threshold)
                updateKNN(node->point);

            return;
        }

        // calculate square distance of current node
        double squareDist =
            (queryPoint.features[d] - node->point.features[d]) *
            (queryPoint.features[d] - node->point.features[d]);

        if (queryPoint.features[d] < node->point.features[d]) {
            leftChecked = true;
            if (node->left)
                findKNNHelper(node->left, queryPoint, (d + 1) % numDim);
        }
        else {
            rightChecked = true;
            if (node->right)
                findKNNHelper(node->right, queryPoint, (d + 1) % numDim);
        }

        // check current node's square distance from query
        node->point.setSquareDistToQuery(queryPoint);

        if (knn.size() < k || node->point.squareDistToQuery < threshold)
            updateKNN(node->point);

        if (!leftChecked) {
            if (node->left && squareDist < threshold)
                findKNNHelper(node->left, queryPoint, (d + 1) % numDim);
        }

        if (!rightChecked) {
            if (node->right && squareDist <= threshold)
                findKNNHelper(node->right, queryPoint, (d + 1) % numDim);
        }
    }
    
    /** Helper method to update your data structure storing KNN using 
     *  the given point.
     */
    void updateKNN(Point& point) {
        // insert if current knn size less than k
        if (knn.size() < k) {
            knn.push_back(point);

            // sort knn
            if (knn.size() == k) {
                sort(knn.begin(), knn.end(), CompareValueAt());
                threshold = knn[knn.size() - 1].squareDistToQuery;
            }
        }
        else {
            if (point.squareDistToQuery < knn[knn.size() - 1].squareDistToQuery) {
                knn.pop_back();
                knn.push_back(point);
                sort(knn.begin(), knn.end(), CompareValueAt());
                threshold = knn[knn.size() - 1].squareDistToQuery;
            }
        }
    }
    
    /** Helper method to destroy all nodes */
    static void deleteAll(KDNode* node) {
        if (node == NULL) return;
        deleteAll(node->left);
        deleteAll(node->right);
        delete(node);
    }
};


#endif // KDT_H

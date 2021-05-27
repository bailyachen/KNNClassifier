# Makefile for KDT and KNNClassifier

CC=g++
CXXFLAGS=-std=c++11 -g
LDFLAGS=-g

default: classify

classify: KNNClassifier

KNNClassifier: KNNClassifier.o

KNNClassifier.o:  KNNClassifier.cpp

clean:
	rm -f KDT KNN *.o core*
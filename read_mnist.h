#ifndef _READ_MNIST_
#include <iostream>  
#include <fstream>  
#include <vector>  
#include "./eigen-3.4.0/Eigen/Dense"

std::vector<std::vector<float>> readImages(const std::string& filename);
std::vector<int> readLabels(const std::string& filename);
Eigen::MatrixXf oneHotEncode(const std::vector<int>& labels, int numClasses);
std::vector<int> findMaxIndicesPerRow(const Eigen::MatrixXf& mat);


#endif












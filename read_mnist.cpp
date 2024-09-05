
#include "read_mnist.h"

void swapEndian32(int* input) {  
    // 交换字节顺序
    int value = *input;
    value = ((value << 24) & 0xff000000) |  
            ((value <<  8) & 0x00ff0000) |  
            ((value >>  8) & 0x0000ff00) |  
            ((value >> 24) & 0x000000ff);  
    *input = value;
} 


const int IMAGE_WIDTH = 28;  
const int IMAGE_HEIGHT = 28;  
const int NUM_LABELS = 10;  
  
// 函数用于读取MNIST的idx3-ubyte图像文件  
std::vector<std::vector<float>> readImages(const std::string& filename) {  
    std::ifstream file(filename, std::ios::binary);  
    if (!file) {  
        throw std::runtime_error("无法打开文件: " + filename);  
    }  
  
    // 读取文件头信息  
    int magicNumber;  
    int numImages;  
    int rows;  
    int cols;  
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    // std::cout << magicNumber << std::endl;
    swapEndian32(&magicNumber);
    // std::cout << magicNumber << std::endl;
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));  
    // std::cout << numImages << std::endl;
    swapEndian32(&numImages);
    // std::cout << numImages<< std::endl;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));  
    // std::cout << rows<< std::endl;
    swapEndian32(&rows);
    // std::cout << rows<< std::endl;
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));  
    // std::cout << cols<< std::endl;
    swapEndian32(&cols);
    // std::cout << cols<< std::endl;
  
    if (magicNumber != 2051) {  
        throw std::runtime_error("不正确的文件类型");  
    }  
  
    if (rows != IMAGE_HEIGHT || cols != IMAGE_WIDTH) {  
        throw std::runtime_error("不正确的图像尺寸");  
    }  
  
    std::vector<std::vector<float>> images(numImages, std::vector<float>(IMAGE_WIDTH * IMAGE_HEIGHT));  
    for (int i = 0; i < numImages; ++i) {  
        for (int j = 0; j < IMAGE_HEIGHT * IMAGE_WIDTH; ++j) {  
            unsigned char pixel;  
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));  
            images[i][j] = static_cast<float>(pixel) / 255.0f;  
        }  
    }  
  
    file.close();  
    return images;  
}  
  
// 函数用于读取MNIST的idx1-ubyte标签文件  
std::vector<int> readLabels(const std::string& filename) {  
    std::ifstream file(filename, std::ios::binary);  
    if (!file) {  
        throw std::runtime_error("无法打开文件: " + filename);  
    }  
  
    // 读取文件头信息  
    int magicNumber;  
    int numLabels;  
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));  
    swapEndian32(&magicNumber);
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));  
    swapEndian32(&numLabels);
  
    if (magicNumber != 2049) {  
        throw std::runtime_error("不正确的文件类型");  
    }  
  
    std::vector<int> labels(numLabels);  
    for (int i = 0; i < numLabels; ++i) {  
        unsigned char label;  
        file.read(reinterpret_cast<char*>(&label), sizeof(label));  
        labels[i] = static_cast<int>(label);  
    }  
  
    file.close();  
    return labels;  
}  


Eigen::MatrixXf oneHotEncode(const std::vector<int>& labels, int numClasses) {  
    int numRows = labels.size();  
    Eigen::MatrixXf oneHotMatrix(numRows, numClasses);  
    oneHotMatrix.setZero(); // 初始化全零矩阵  
  
    for (int i = 0; i < numRows; ++i) {  
        int label = labels[i];  
        if (label >= 0 && label < numClasses) {  
            oneHotMatrix(i, label) = 1.f; // 设置对应位置为1  
        } else {  
            std::cerr << "Label " << label << " is out of range [0, " << numClasses - 1 << "]." << std::endl;  
        }  
    }  
    // for (int i = 0; i < numRows; ++i) {  
    //     int label = labels[i];  
    //     if (label >= 0 && label < numClasses)
    //     {  
    //         oneHotMatrix(i, label) = 1.f; 
    //     }
    // }  
  
    return oneHotMatrix;  
}  


std::vector<int> findMaxIndicesPerRow(const Eigen::MatrixXf& mat) {  
    std::vector<int> maxIndices(mat.rows()); // 初始化一个vector来保存每行最大值的索引  
  
    // 遍历每一行，找到最大值的列索引  
    for (int i = 0; i < mat.rows(); ++i) {  
        int maxIndexCol = 0; // 初始化最大值的列索引为0  
        double maxValue = mat(i, 0); // 假设第一列是最大值  
        for (int j = 1; j < mat.cols(); ++j) {  
            if (mat(i, j) > maxValue) {  
                maxValue = mat(i, j);  
                maxIndexCol = j;  
            }  
        }  
        maxIndices[i] = maxIndexCol; // 将最大值的列索引保存到vector中  
    }  
  
    return maxIndices; // 返回包含每行最大值的列索引的vector  
}  
  

// std::string trainImagesFilename = "train-images-idx3-ubyte";  
// std::string trainLabelsFilename = "train-labels-idx1-ubyte";  
// std::string testImagesFilename = "t10k-images-idx3-ubyte";  
// std::string testLabelsFilename = "t10k-labels-idx1-ubyte";  

// std::vector<std::vector<float>> trainImages = readImages(trainImagesFilename);  
// std::vector<int> trainLabels = readLabels(trainLabelsFilename);  
// std::vector<std::vector<float>> testImages = readImages(testImagesFilename);  
// std::vector<int> testLabels = readLabels(testLabelsFilename);  
  






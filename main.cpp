
#include "nn.h"
#include "nn_cuda.cuh" 
#include <iostream>
#include <chrono>
#include "cuda_opt.cuh"
#include "read_mnist.h"


// void print(auto & any)
// {
//     std::cout << any << std::endl;
// }

int main()
{
    auto start = std::chrono::high_resolution_clock::now();  
    std::string trainImagesFilename = ".\\mnist\\train-images.idx3-ubyte";
    std::string trainLabelsFilename = ".\\mnist\\train-labels.idx1-ubyte";
    std::string testImagesFilename = ".\\mnist\\t10k-images.idx3-ubyte";
    std::string testLabelsFilename = ".\\mnist\\t10k-labels.idx1-ubyte";
    std::vector<std::vector<float>> trainImages = readImages(trainImagesFilename);
    std::vector<int> trainLabels = readLabels(trainLabelsFilename);
    Eigen::MatrixXf train_x(trainImages.size(), trainImages[0].size());
    for (int i = 0; i < trainImages.size(); ++i)
    {
        for (int j = 0; j < trainImages[0].size(); ++j)
        {
            train_x(i, j) = trainImages[i][j];
        }
    }
    Eigen::MatrixXf train_y = oneHotEncode(trainLabels, 10);
    Eigen::MatrixXf train_x_t = train_x.transpose();
    Eigen::MatrixXf train_y_t = train_y.transpose();


    std::vector<std::vector<float>> testImages = readImages(testImagesFilename);
    std::vector<int> testLabels = readLabels(testLabelsFilename);
    Eigen::MatrixXf test_x(testImages.size(), testImages[0].size());
    for (int i = 0; i < testImages.size(); ++i)
    {
        for (int j = 0; j < testImages[0].size(); ++j)
        {
            test_x(i, j) = testImages[i][j];
        }
    }
    Eigen::MatrixXf test_y = oneHotEncode(testLabels, 10);
    Eigen::MatrixXf test_x_t = test_x.transpose();
    Eigen::MatrixXf test_y_t = test_y.transpose();

    auto end = std::chrono::high_resolution_clock::now();  
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);  
    std::cout << "read data time:" << duration.count() / 1000 << std::endl;  
  

    // nn参数
    int x_dims = train_x_t.rows();
    // print(x_dims);
    std::cout << x_dims << std::endl;
    int y_dims = train_y_t.rows();
    // print(y_dims);
    std::cout << y_dims << std::endl;
    int hidden_nums = 32;
    float lr = 1e-2;
    int epochs = 10;
    int batch = 32;

    // nn实例化
    NNCuda nn(x_dims, y_dims, hidden_nums, lr, epochs, batch);
    // NN nn(x_dims, y_dims, hidden_nums, lr, epochs, batch);
    // start = std::chrono::high_resolution_clock::now();  
    nn.train(train_x_t, train_y_t);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);  
    std::cout << "train time:" << duration.count() / 1000 << std::endl;  

    int predict_number = 10;

    Eigen::MatrixXf train_pred_y_t = nn.predict(train_x_t.block(0, 0, train_x_t.rows(), predict_number));
    Eigen::MatrixXf train_pred_y = train_pred_y_t.transpose();

    std::vector<int> trainLabels_10(trainLabels.begin(), trainLabels.begin() + predict_number);
    for (int value : trainLabels_10) {  
        std::cout << value << " ";  
    }  
    std::cout << std::endl;  
    std::vector<int> trainPredLabels_10 = findMaxIndicesPerRow(train_pred_y);
    for (int value : trainPredLabels_10) {  
        std::cout << value << " ";  
    }  
    std::cout << std::endl;  

    Eigen::MatrixXf test_pred_y_t = nn.predict(test_x_t.block(0, 0, test_x_t.rows(), predict_number));
    Eigen::MatrixXf test_pred_y = test_pred_y_t.transpose();

    std::vector<int> testLabels_10(testLabels.begin(), testLabels.begin() + predict_number);
    for (int value : testLabels_10) {  
        std::cout << value << " ";  
    }  
    std::cout << std::endl;  
    std::vector<int> testPredLabels_10 = findMaxIndicesPerRow(test_pred_y);
    for (int value : testPredLabels_10) {  
        std::cout << value << " ";  
    }  
    std::cout << std::endl;  
    // print(testPredLabels);
    return 0;
}

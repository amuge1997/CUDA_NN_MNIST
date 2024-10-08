
#ifndef _NN_CUDA_
#define _NN_CUDA_

#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include "./eigen-3.4.0/Eigen/Dense"


// nn网络
class NNCuda
{
public:
    NNCuda(int x_dims, int y_dims, int hidden_dims, float lr, int epochs, int batch)
    {
        this->x_dims = x_dims;           // 输入维度
        this->y_dims = y_dims;           // 输出维度
        this->hidden_dims = hidden_dims; // 中间维度
        this->lr = lr;                   // 学习率
        this->epochs = epochs;           // 轮次
        this->batch = batch;             // 批数量
    }

    // 预测
    Eigen::MatrixXf predict(const Eigen::MatrixXf &x)
    {
        int nums = x.cols();

        Eigen::MatrixXf bias(1, nums);                           // 偏置向量
        Eigen::MatrixXf x_bias(x_dims + 1, nums);                // 添加偏置后的输入矩阵
        Eigen::MatrixXf hidden1_sig_bias(hidden_dims + 1, nums); // 添加偏置后的中间矩阵
        Eigen::MatrixXf hidden2_sig_bias(hidden_dims + 1, nums); // 添加偏置后的中间矩阵

        Eigen::MatrixXf hidden1;
        Eigen::MatrixXf hidden2;
        Eigen::MatrixXf hidden1_sig;
        Eigen::MatrixXf hidden2_sig;
        Eigen::MatrixXf out;

        bias.setOnes();
        x_bias << x, bias;

        hidden1 = weights1 * x_bias;
        hidden1_sig = 1.0 / (1.0 + (-hidden1.array()).exp());
        hidden1_sig_bias << hidden1_sig, bias;

        hidden2 = weights2 * hidden1_sig_bias;
        hidden2_sig = 1.0 / (1.0 + (-hidden2.array()).exp());
        hidden2_sig_bias << hidden2_sig, bias;

        out = weights3 * hidden2_sig_bias;
        return out;
    }

    // 训练
    void train(const Eigen::MatrixXf &x_ori, const Eigen::MatrixXf &y_ori)
    {
        int nums = x_ori.cols(); // 样本数量

        Eigen::MatrixXf bias(1, batch);                           // 偏置向量
        Eigen::MatrixXf x(x_dims, batch);                         // 输入层
        Eigen::MatrixXf y(y_dims, batch);                         // 标签
        Eigen::MatrixXf x_bias(x_dims + 1, batch);                // 输入偏置层
        Eigen::MatrixXf hidden1_sig_bias(hidden_dims + 1, batch); // 中间偏置层
        Eigen::MatrixXf hidden2_sig_bias(hidden_dims + 1, batch); // 中间偏置层

        Eigen::MatrixXf hidden1(hidden_dims, batch);     // 隐藏层1
        Eigen::MatrixXf hidden2(hidden_dims, batch);     // 隐藏层2
        Eigen::MatrixXf hidden1_sig(hidden_dims, batch); // sigmoid激活层1
        Eigen::MatrixXf hidden2_sig(hidden_dims, batch); // sigmoid激活层2
        Eigen::MatrixXf out(y_dims, batch);         // 输出层
        Eigen::MatrixXf error(y_dims, batch);       // 误差
        float error2;                // 总误差

        Eigen::MatrixXf dEdO(y_dims, batch);  // 误差对输出梯度
        Eigen::MatrixXf dEdW3(y_dims, hidden_dims + 1); // 误差对W3梯度

        Eigen::MatrixXf dEdH2sig(hidden_dims + 1, batch);  // 误差对sigmoid激活层2梯度
        Eigen::MatrixXf dEdH2bias(hidden_dims + 1, batch); // 误差对含偏置隐藏层2梯度
        Eigen::MatrixXf dEdH2(hidden_dims, batch);     // 误差对隐藏层2梯度
        Eigen::MatrixXf dEdW2(hidden_dims, hidden_dims + 1);     // 误差对W2梯度

        Eigen::MatrixXf dEdH1sig(hidden_dims + 1, batch);  // 误差对sigmoid激活层1梯度
        Eigen::MatrixXf dEdH1bias(hidden_dims + 1, batch); // 误差对含偏置隐藏层1梯度
        Eigen::MatrixXf dEdH1(hidden_dims, batch);     // 误差对隐藏层1梯度
        Eigen::MatrixXf dEdW1(hidden_dims, x_dims + 1);     // 误差对W1梯度

        // 初始化参数矩阵
        weights1 = Eigen::MatrixXf::Random(hidden_dims, x_dims + 1).array() / sqrt(hidden_dims + x_dims + 1);
        weights2 = Eigen::MatrixXf::Random(hidden_dims, hidden_dims + 1).array() / sqrt(hidden_dims + hidden_dims + 1);
        weights3 = Eigen::MatrixXf::Random(y_dims, hidden_dims + 1).array() / sqrt(y_dims + hidden_dims + 1);

        
        // 偏置向量
        bias.setOnes();

        
        Eigen::MatrixXf xt(batch, x_dims);                         // 输入层
        Eigen::MatrixXf yt(batch, y_dims);                         // 标签

        Eigen::MatrixXf weights1t = weights1.transpose();
        Eigen::MatrixXf weights2t = weights2.transpose();
        Eigen::MatrixXf weights3t = weights3.transpose();
        // 这里将 weights1t, weights2t，weights3t, bias   的内存复制到 d_weights1t, d_weights2t，d_weights3t, d_bias


        // 序号列表 用于样本批次抽样
        std::srand(unsigned(std::time(nullptr)));
        std::vector<int> numbers;
        for (int i = 0; i < nums; i++)
        {
            numbers.push_back(i);
        }
        for (int i = 0; i < epochs; i++)
        {
            // 每批次随机抽样
            std::random_shuffle(numbers.begin(), numbers.end());
            int batch_sum = nums / batch;
            for (int j = 0; j < batch_sum; j++)
            {
                for (int bi = 0; bi < batch; bi++)
                {
                    x.col(bi) = x_ori.col(numbers[bi + j * batch]);
                    y.col(bi) = y_ori.col(numbers[bi + j * batch]);
                }
                xt = x.transpose();
                yt = y.transpose();
                // 这里将 xt, yt 的内存复制到 d_x, d_y

                // 添加偏置
                x_bias << x, bias; // 使用核函数将x,bias赋值给x_bias                        concatXBiasKernel(float* A, float* B, float* C, int Crows, int Ccols)

                // h1 = sigmoid( w1 * x )    使用sigmoid激活
                hidden1 = weights1 * x_bias;                          // cuda 矩阵乘法      matMulKernel(float *A, float *B, float *C, int Arows, int Acols, int Bcols)
                hidden1_sig = 1.0 / (1.0 + (-hidden1.array()).exp()); // cuda 直接计算      sigmoidKernel(float *A, float *B, int Acols, int Acols)
                hidden1_sig_bias << hidden1_sig, bias;                // cuda 内存转换      concatXBiasKernel(float* A, float* B, float* C, int Crows, int Ccols)

                // h2 = sigmoid( w2 * h1 )    使用sigmoid激活
                hidden2 = weights2 * hidden1_sig_bias;                // cuda 矩阵乘法      matMulKernel(float *A, float *B, float *C, int Arows, int Acols, int Bcols)
                hidden2_sig = 1.0 / (1.0 + (-hidden2.array()).exp()); // cuda 直接计算      sigmoidKernel(float *A, float *B, int Acols, int Acols)
                hidden2_sig_bias << hidden2_sig, bias;                // cuda 内存转换      concatXBiasKernel(float* A, float* B, float* C, int Crows, int Ccols)

                // 为了能全域映射,不进行激活直接线性映射得到输出
                // o = w3 * h2
                out = weights3 * hidden2_sig_bias; // cuda 矩阵乘法                         matMulKernel(float *A, float *B, float *C, int Arows, int Acols, int Bcols)

                // 误差
                // e = (Y - O)^2
                error = y - out;                               // cuda 直接计算             matSubKernel(float *A, float *B, float *C, int Crows, int Ccols)
                error2 = error.array().square().sum() / batch; // cuda 直接计算

                // 链式法则求各矩阵梯度
                // 输出层梯度
                dEdO = -2. / batch * error.array();          // cuda 直接计算               matErrorKernel(float *A, float *B, int Brows, int Bcols, int batch)
                dEdW3 = dEdO * hidden2_sig_bias.transpose(); // cuda 矩阵转置+矩阵乘法      matMulBTKernel(float *A, float *B, float *C, int Arows, int Acols, int Brows)
                dEdH2sig = weights3.transpose() * dEdO;      // cuda 矩阵转置+矩阵乘法      matMulATKernel(float *A, float *B, float *C, int Acols, int Arows, int Brows)

                // 第二层梯度
                dEdH2bias = dEdH2sig.array() * hidden2_sig_bias.array() * (1.0 - hidden2_sig_bias.array()); // cuda 直接计算            sigmoidDiffKenel(float *A, float *B, float *C, int Arows, int Acols)
                dEdH2 = dEdH2bias.topRows(dEdH2bias.rows() - 1);                                            // cuda 去除bias            copyNoBiasKernel(float *A, float *B, int Brows, int Bcols)
                dEdW2 = dEdH2 * hidden1_sig_bias.transpose();                                               // cuda 矩阵转置+矩阵乘法   matMulBTKernel(float *A, float *B, float *C, int Arows, int Acols, int Brows)
                dEdH1sig = weights2.transpose() * dEdH2;                                                    // cuda 矩阵转置+矩阵乘法   matMulATKernel(float *A, float *B, float *C, int Acols, int Arows, int Brows)

                // 第一层梯度
                dEdH1bias = dEdH1sig.array() * hidden1_sig_bias.array() * (1.0 - hidden1_sig_bias.array()); // cuda 直接计算            sigmoidDiffKenel(float *A, float *B, float *C, int Arows, int Acols)
                dEdH1 = dEdH1bias.topRows(dEdH1bias.rows() - 1);                                            // cuda 去除bias            copyNoBiasKernel(float *A, float *B, int Brows, int Bcols)
                dEdW1 = dEdH1 * x_bias.transpose();                                                         // cuda 矩阵转置+矩阵乘法   matMulBTKernel(float *A, float *B, float *C, int Arows, int Acols, int Brows)

                // 更新参数矩阵
                weights1 = weights1 - lr * dEdW1; // cuda 直接计算      weightUpdateKernel(float* A, float* B, int Arows, int Acols, float lr)
                weights2 = weights2 - lr * dEdW2; // cuda 直接计算      weightUpdateKernel(float* A, float* B, int Arows, int Acols, float lr)
                weights3 = weights3 - lr * dEdW3; // cuda 直接计算      weightUpdateKernel(float* A, float* B, int Arows, int Acols, float lr)
            }
            if (i % 1 == 0)
            {
                printf("???%d/%d error %f\n", i + 1, epochs, error2);
            }
        }

        // 这里将
    }

private:
    int x_dims;
    int y_dims;
    int hidden_dims;
    float lr;
    int epochs;
    int batch;
    Eigen::MatrixXf weights1;
    Eigen::MatrixXf weights2;
    Eigen::MatrixXf weights3;
};

#endif

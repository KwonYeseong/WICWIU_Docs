#ifndef SOFTMAXCROSSENTROPY_H_
#define SOFTMAXCROSSENTROPY_H_    value

#include "../LossFunction.h"

/*!
@class
@details
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE>
class SoftmaxCrossEntropy : public LossFunction<DTYPE>{
private:
    Tensor<DTYPE> *m_aSoftmaxResult; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    DTYPE m_epsilon;  // for backprop ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

    int m_timesize; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

    DTYPE **sum; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    DTYPE **max; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

public:
    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    SoftmaxCrossEntropy(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, DTYPE epsilon, std::string pName = "NO NAME") : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator, epsilon);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    SoftmaxCrossEntropy(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName = "NO NAME") : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator, 1e-6f);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    virtual ~SoftmaxCrossEntropy() {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::~SoftmaxCrossEntropy()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon) {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_timesize = timesize;

        sum = new DTYPE *[timesize];
        max = new DTYPE *[timesize];

        for (int i = 0; i < timesize; i++) {
            sum[i] = new DTYPE[batchsize];
            max[i] = new DTYPE[batchsize];
        }

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        m_aSoftmaxResult = new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize);

        m_epsilon = epsilon;

        return TRUE;
    }

    #ifdef __CUDNN__
        /*!
        @brief
        @details
        @param
        @return
        @todo GPU
        */
        // 문서 작성자 : , 작성 날짜 : 2018-
        void InitializeAttributeForGPU(unsigned int idOfDevice) {
            m_aSoftmaxResult->SetDeviceGPU(idOfDevice);
        }

    #endif  // if __CUDNN__

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    virtual void Delete() {
        if (m_aSoftmaxResult) {
            delete m_aSoftmaxResult;
            m_aSoftmaxResult = NULL;
        }

        if (sum) {
            for (int i = 0; i < m_timesize; i++) {
                delete[] sum[i];
                sum[i] = NULL;
            }
            delete[] sum;
        }

        if (max) {
            for (int i = 0; i < m_timesize; i++) {
                delete[] max[i];
                max[i] = NULL;
            }
            delete[] max;
        }
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo E_Train
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE>* ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input         = this->GetTensor();
        Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
        Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;
        Tensor<DTYPE> *result        = this->GetResult();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {  // thread
            sum[ti][ba] = 0.f;
            max[ti][ba] = 0.f;
        }


        int numOfOutputDim = 0;

        int capacity = colsize;

        int start = 0;
        int end   = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            max[ti][ba] = Max(input, start, end);
        }

        DTYPE temp = 0.f;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                temp += (exp((*input)[i] - max[ti][ba]) + m_epsilon);
            }
            sum[ti][ba] = temp;
            temp        = 0.f;
        }

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*softmaxresult)[i] = (exp((*input)[i] - max[ti][ba]) + m_epsilon) / sum[ti][ba];

                (*result)[ti * batchsize + ba] += -(*label)[i] * log((*softmaxresult)[i] + m_epsilon);
            }
        }

        return result;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo E_Train
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE>* BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
        Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;

        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchsize = input_delta->GetBatchSize();
        int colsize   = input_delta->GetColSize();

        int capacity = colsize;

        int start = 0;
        int end   = 0;

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*input_delta)[i] = ((*softmaxresult)[i] - (*label)[i]) / batchsize;
            }
        }

        return NULL;
    }

#ifdef __CUDNN__

    /*!
    @brief
    @details
    @param
    @return
    @todo GPU
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *input         = this->GetTensor();
        Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
        Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;
        Tensor<DTYPE> *result        = this->GetResult();

        int batchsize = input->GetBatchSize();
        int colsize   = input->GetColSize();

        float alpha = 1.f;
        float beta  = 0.f;

        cudnnTensorDescriptor_t pInputDesc   = input->GetDescriptor();
        cudnnTensorDescriptor_t pSoftMaxDesc = softmaxresult->GetDescriptor();

        DTYPE *pDevInput   = input->GetGPUData(pTime);
        DTYPE *pDevSoftMax = softmaxresult->GetGPUData(pTime);

        checkCUDNN(cudnnSoftmaxForward(this->GetCudnnHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                       &alpha, pInputDesc, pDevInput,
                                       &beta, pSoftMaxDesc, pDevSoftMax));

        int start = 0;
        int end   = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (pTime * batchsize + ba) * colsize;
            end   = start + colsize;

            for (int i = start; i < end; i++) {
                (*result)[pTime * batchsize + ba] += -(*label)[i] * log((*softmaxresult)[i] + m_epsilon);
            }
        }

        return result;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo GPU
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0) {
        return this->BackPropagate(pTime);
    }

#endif  // __CUDNN__


    /*!
    @brief
    @details
    @param
    @return
    @todo E_Train
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    DTYPE Max(Tensor<DTYPE> *input, int start, int end) {
        DTYPE max = (*input)[start];

        for (int i = start + 1; i < end; i++) {
            if ((*input)[i] > max) max = (*input)[i];
        }

        return max;
    }
};

#endif  // SOFTMAXCROSSENTROPY_H_

#ifndef GRADIENTDESCENTOPTIMIZER_H_
#define GRADIENTDESCENTOPTIMIZER_H_    value

#include "../Optimizer.h"

/*!
@class
@details
@todo 우선순위
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> class GradientDescentOptimizer : public Optimizer<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_ppParameter; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    Container<Tensor<DTYPE> *> *m_aaVelocity; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-

    int m_numOfParameter; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    float m_momentum; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-

public:
    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    GradientDescentOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc();
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    GradientDescentOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float momentum, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc(momentum);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    GradientDescentOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float momentum, float weightDecayRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, weightDecayRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc(momentum);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    ~GradientDescentOptimizer() {
        #ifdef __DEBUG__
        std::cout << "GradientDescentOptimizer::~GradientDescentOptimizer()" << '\n';
        #endif  // __DEBUG__
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int Alloc() {
        m_ppParameter    = this->GetTrainableTensor();
        m_numOfParameter = this->GetTrainableTensorDegree();

        return TRUE;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int Alloc(float momentum) {
        Alloc();
        m_aaVelocity = new Container<Tensor<DTYPE> *>();

        Shape *pParameterGradShape = NULL;

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();

            // std::cout << (*m_ppParameter)[i]->GetName() << '\n';
            // std::cout << pParameterGradShape << '\n';

            m_aaVelocity->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            pParameterGradShape = NULL;
        }

        m_momentum = momentum;

        return TRUE;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        if (m_momentum != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                (*m_aaVelocity)[i]->SetDeviceGPU(idOfDevice);
            }
        }
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    virtual int UpdateParameter() {
        if (m_momentum == 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameter((*m_ppParameter)[i]);
            }
        } else {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameter((*m_ppParameter)[i], (*m_aaVelocity)[i]);
            }
        }

        return TRUE;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int UpdateParameter(Operator<DTYPE> *pParameter) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float learning_rate   = this->GetOptimizeDirection() * this->GetLearningRate();
        float weightDecayRate = this->GetWeightDecayRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*trainable_data)[i] += (learning_rate * (*gradient)[i] + weightDecayRate * (*trainable_data)[i]);
        }

        return TRUE;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float learning_rate   = this->GetOptimizeDirection() * this->GetLearningRate();
        float weightDecayRate = this->GetWeightDecayRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*pVelocity)[i]      = m_momentum * (*pVelocity)[i] + learning_rate * (*gradient)[i];
            (*trainable_data)[i] += ((*pVelocity)[i] + weightDecayRate * (*trainable_data)[i]);
        }

        return TRUE;
    }

#ifdef __CUDNN__


    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    virtual int UpdateParameterOnGPU() {
        if (m_momentum == 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i]);
            }
        } else {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaVelocity)[i]);
            }
        }

        return TRUE;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        cudnnTensorDescriptor_t dataDesc = trainable_data->GetDescriptor();
        cudnnTensorDescriptor_t gradDesc = gradient->GetDescriptor();

        DTYPE *m_pDevData = trainable_data->GetGPUData();
        DTYPE *m_pDevGrad = gradient->GetGPUData();

        float learning_rate   = this->GetOptimizeDirection() * this->GetLearningRate();
        float weightDecayRate = this->GetWeightDecayRate();

        float alpha = 1.f + learning_rate * weightDecayRate;
        float beta  = learning_rate;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &beta, gradDesc, m_pDevGrad,
                                  &alpha, dataDesc, m_pDevData));

        return TRUE;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        cudnnTensorDescriptor_t dataDesc = trainable_data->GetDescriptor();
        cudnnTensorDescriptor_t gradDesc = gradient->GetDescriptor();
        cudnnTensorDescriptor_t veloDesc = pVelocity->GetDescriptor();

        DTYPE *m_pDevData = trainable_data->GetGPUData();
        DTYPE *m_pDevGrad = gradient->GetGPUData();
        DTYPE *m_pDevVelo = pVelocity->GetGPUData();

        float learning_rate   = this->GetOptimizeDirection() * this->GetLearningRate();
        float weightDecayRate = this->GetWeightDecayRate();

        float alpha = 1.f + learning_rate * weightDecayRate;
        // std::cout << "weight decay  : "<< weightDecayRate << '\n';
        // std::cout << "alpha : " << alpha << '\n';
        float beta  = learning_rate;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &beta, gradDesc, m_pDevGrad,
                                  &m_momentum, veloDesc, m_pDevVelo));

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &alpha, veloDesc, m_pDevVelo,
                                  &alpha, dataDesc, m_pDevData));

        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // GRADIENTDESCENTOPTIMIZER_H_

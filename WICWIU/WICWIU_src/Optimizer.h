#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_    value

#include "LossFunction_utils.h"

/*!
@brief
@details
*/
// 문서 작성자 : , 작성 날짜 : 2018-
enum OptimizeDirection {
    MAXIMIZE,
    MINIMIZE
};

template<typename DTYPE> class Optimizer {
private:
    float m_LearningRate; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_OptimizeDirection;  // 1 or -1 ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    float m_weightDecayRate; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

    Container<Operator<DTYPE> *> *m_ppTrainableTensors; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_TrainableTensorDegree; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

    int m_idOfDevice = -1; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

#ifdef __CUDNN__
    cudnnHandle_t m_pCudnnHandle; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
#endif  // if __CUDNN__

public:
    Optimizer(Operator<DTYPE> **pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection);
    Optimizer(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection);
    Optimizer(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, float pWeightDecayRate, OptimizeDirection pOptimizeDirection);

    virtual ~Optimizer();

    int                           Alloc(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection);
    int                           Alloc(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, float pWeightDecayRate, OptimizeDirection pOptimizeDirection);
    int                           Delete();

    virtual int                   UpdateParameter();
    virtual int                   UpdateParameter(Operator<DTYPE> *pTrainableTensor) = 0;

    void                          SetLearningRate(float pLearningRate);
    void                          SetTrainableTensorDegree(int pTrainableTensorDegree);
    void                          SetWeightDecayRate(int pWeightDecayRate);

    float                         GetLearningRate() const;
    int                           GetOptimizeDirection() const;
    Container<Operator<DTYPE> *>* GetTrainableTensor();
    int                           GetTrainableTensorDegree() const;
    float                         GetWeightDecayRate() const;

    int                           ResetParameterGradient();

#ifdef __CUDNN__
    void                          SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice);
    virtual void                  InitializeAttributeForGPU(unsigned int idOfDevice) = 0;
    virtual void                  SetCudnnHandle(cudnnHandle_t& pCudnnHandle);
    virtual int                   UpdateParameterOnGPU();
    virtual int                   UpdateParameterOnGPU(Operator<DTYPE> *pTrainableTensor) = 0;

    cudnnHandle_t& GetCudnnHandle();
    int            GetDeviceID();

#endif  // if __CUDNN__
};

#endif  // OPTIMIZER_H_

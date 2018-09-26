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
    float m_LearningRate; ///< 신경망에 대한 최적화(Optimizing) 수식의 학습률(learning Rate)에 해당하는 변수
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-25
    int m_OptimizeDirection;  // 신경망에 대한 최적화(Optimizing) 수식의 Gradient의 부호에 해당하는 변수. 경사 상승 시 1, 경사 하강 시 -1
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-25
    float m_weightDecayRate; ///< 신경망에 대한 최적화(Optimizing) 수식의 가중치 감소율(Weight Decay Rate)에 해당하는 변수 //// Legularization, Overfitting을 막음
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-25

    Container<Operator<DTYPE> *> *m_ppTrainableTensors; ///< NeuralNetwork의 operator들 중 학습이 가능한 Tensor, 즉 parameter들을 변수로 갖는 Operator들의 포인터들을 원소로 갖는 Container에 대한 포인터
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-25
    int m_TrainableTensorDegree; ///< TrainableTensor Container에 포함되어 있는 텐서들의 개수  @todo 확인 요
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-25

    int m_idOfDevice = -1; ///< GPU 사용 시, 사용하려는 GPU의 번호. CPU의 경우 -1
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-25

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

    virtual int                   UpdateParameter(); //// 보툥 사용하는 메소드
    virtual int                   UpdateParameter(Operator<DTYPE> *pTrainableTensor) = 0; //// 잘 사용하지 않음

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

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Optimizer_utils.h"

/*!
@class NeuralNetwork 뉴럴 네트워크 모델 생성, 학습 및 평가를 총괄하는 클래스
@details Operator, Module, Loss Function, Optimizer 클래스를 생성 및 활용해 뉴럴 네트워크를 구성하고 학습시킨다
*/
template<typename DTYPE> class NeuralNetwork {
private:
    Container<Operator<DTYPE> *> *m_aaOperator; ///< 신경망의 전체 Operator들의 포인터를 담고 있는 Container의 포인터 멤버 변수  //// Excutable + Input + Parameter
    Container<Operator<DTYPE> *> *m_apExcutableOperator; ///< 순전파 시 연산을 수행하는 Operator들의 포인터를 담고 있는 Container의 포인터 멤버 변수  //// 실행시키는 Operator 리스트
    Container<Operator<DTYPE> *> *m_apInput; ///< 신경망의 최초 Input이 되는 Operator들의 포인터를 담고 있는 Container의 포인터 멤버 변수  //// setInput, input임을 알려줘야 함
    Container<Operator<DTYPE> *> *m_apParameter; ///< 신경망의 학습이 가능한 파라미터에 해당하는 Operator들의 포인터를 담고 있는 Container의 포인터 멤버 변수   //// Optimizer를 추가할 때, Optimizer에게 파라미터 리스트를 넘겨줌

    int m_Operatordegree; ///< 해당 클래스의 Operator Container 멤버 변수의 Element의 개수    ////  Operator Element의 개수, num of Operator STL 구현 전에 구현해 둠, 안정성을 위해 남겨둠
    int m_ExcutableOperatorDegree; ///< 해당 클래스의 Excutable Operator Container 멤버 변수의 Element의 개수    ////  STL 구현 전에 구현해 둠, 안정성을 위해 남겨둠
    int m_InputDegree; ///< 해당 클래스의 Input Container 멤버 변수의 Element의 개수    ////  STL 구현 전에 구현해 둠, 안정성을 위해 남겨둠
    int m_ParameterDegree; ///< 해당 클래스의 Parameter Container 멤버 변수의 Element의 개수    ////  STL 구현 전에 구현해 둠, 안정성을 위해 남겨둠

    LossFunction<DTYPE> *m_aLossFunction; ///< 신경망의 손실함수에 해당하는 LossFunction의 포인터 멤버 변수  //// 하나만 갖도록 구현
    Optimizer<DTYPE> *m_aOptimizer; ///< 신경망의 Optimizer에 해당하는 Optimizer의 포인터 멤버 변수  //// 하나만 갖도록 구현

    Device m_Device; ///< 장치 사용 구분자, CPU 또는 GPU, Device 참고
    int m_idOfDevice = -1;  // 추후 수정  ///< GPU 사용 시, 사용하려는 GPU의 번호. CPU의 경우 -1

#ifdef __CUDNN__
    cudnnHandle_t m_cudnnHandle; ///< cudnn handler
#endif  // if __CUDNN__

private:
    int  Alloc();
    void Delete();

#ifdef __CUDNN__
    int  AllocOnGPU();
    void DeleteOnGPU();
#endif  // if __CUDNN__

public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    Operator<DTYPE>             * SetInput(Operator<DTYPE> *pInput);
    int                           SetInput(int pNumOfInput, ...);
    int                           IsInput(Operator<DTYPE> *pOperator);

    int                           IsValid(Operator<DTYPE> *pOperator); // Graph 분석 시 node에 추가할 것인지 확인한다.

    Operator<DTYPE>             * AnalyzeGraph(Operator<DTYPE> *pResultOperator);
    LossFunction<DTYPE>         * SetLossFunction(LossFunction<DTYPE> *pLossFunction);
    Optimizer<DTYPE>            * SetOptimizer(Optimizer<DTYPE> *pOptimizer);
    int                           FeedInputTensor(int pNumOfInput, ...);
    // =======

    Container<Operator<DTYPE> *>* GetInputContainer();

    Operator<DTYPE>             * GetResultOperator();
    Operator<DTYPE>             * GetResult();

    Container<Operator<DTYPE> *>* GetExcutableOperatorContainer();

    Container<Operator<DTYPE> *>* GetParameterContainer();
    Container<Operator<DTYPE> *>* GetParameter();

    LossFunction<DTYPE>         * GetLossFunction();

    Optimizer<DTYPE>            * GetOptimizer();

    int                           ForwardPropagate(int pTime = 0);
    int                           BackPropagate(int pTime = 0);

    void                          SetDeviceCPU();

    void                          SetModeTraining();
    void                          SetModeAccumulating();
    void                          SetModeInferencing();

    int                           Training();
    int                           Testing();

    int                           TrainingOnCPU();
    int                           TestingOnCPU();

    int                           TrainingOnGPU();
    int                           TestingOnGPU();

    float                         GetAccuracy(int numOfClass = 10);
    int                           GetMaxIndex(Tensor<DTYPE> *data, int ba, int ti, int numOfClass);

    float                         GetTop5Accuracy(int numOfClass = 10);
    int*                          GetTop5Index(Tensor<DTYPE> *data, int ba, int ti, int numOfClass);


    float                         GetLoss();

    void                          PrintGraphInformation();

    int                           ResetOperatorResult();
    int                           ResetOperatorGradient();

    int                           ResetLossFunctionResult();
    int                           ResetLossFunctionGradient();

    int                           ResetParameterGradient();

    Operator<DTYPE>             * SerchOperator(std::string pName);

    int                           Save(FILE * fileForSave);
    int                           Load(FILE * fileForLoad);

#ifdef __CUDNN__
    int                           ForwardPropagateOnGPU(int pTime = 0);
    int                           BackPropagateOnGPU(int pTime = 0);

    void                          SetDeviceGPU(unsigned int idOfDevice);
    int                           SetDeviceID(unsigned int idOfDevice);
#endif  // __CUDNN__
};

#endif  // NEURALNETWORK_H_

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Optimizer_utils.h"

/*!
@class
@details
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> class NeuralNetwork {
private:
    Container<Operator<DTYPE> *> *m_aaOperator; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    Container<Operator<DTYPE> *> *m_apExcutableOperator; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    Container<Operator<DTYPE> *> *m_apInput; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    Container<Operator<DTYPE> *> *m_apParameter; ///<   @todo Variable   //// Optimizer를 추가할 때, Optimizer에게 파라미터 리스트를 넘겨줌
    // 문서 작성자 : , 작성 날짜 : 2018-

    int m_Operatordegree; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_ExcutableOperatorDegree; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_InputDegree; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_ParameterDegree; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

    LossFunction<DTYPE> *m_aLossFunction; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    Optimizer<DTYPE> *m_aOptimizer; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

    Device m_Device; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_idOfDevice = -1;  // 추후 수정  ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

#ifdef __CUDNN__
    cudnnHandle_t m_cudnnHandle; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
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

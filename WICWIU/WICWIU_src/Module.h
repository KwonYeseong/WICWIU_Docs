#ifndef __MODULE_H_
#define __MODULE_H_    value

#include "Operator_utils.h"

/*!
@class
@details
@todo 우선순위
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> class Module : public Operator<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_aaExcutableOperator; ///<   @todo 우선순위,
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_numOfExcutableOperator; ///<   @todo 우선순위,
    // 문서 작성자 : , 작성 날짜 : 2018-

    Operator<DTYPE> *m_pLastOperator; ///<   @todo 우선순위,
    // 문서 작성자 : , 작성 날짜 : 2018-

    Device m_Device; ///<   @todo 우선순위,
    // 문서 작성자 : , 작성 날짜 : 2018-
    unsigned int m_idOfDevice = 0; ///<   @todo 우선순위,
    // 문서 작성자 : , 작성 날짜 : 2018-

private:
    int  Alloc();
    void Delete();

public:
    Module(std::string pName = "No Name");
    virtual ~Module();

    Operator<DTYPE>                   * SetInput(Operator<DTYPE> *pInput);
    int                                 SetInput(int pNumOfInput, ...);

    int                                 IsInput(Operator<DTYPE> *pOperator);

    int                                 IsValid(Operator<DTYPE> *pOperator); // Graph 분석 시 node에 추가할 것인지 확인한다.

    Operator<DTYPE>                   * AnalyzeGraph(Operator<DTYPE> *pResultOperator);

    Container<Operator<DTYPE> *>      * GetExcutableOperatorContainer();
    int                                 GetNumOfExcutableOperator();

    virtual Tensor<DTYPE>             * GetResult() const;
    virtual Container<Tensor<DTYPE> *>* GetResultContainer();

    virtual Tensor<DTYPE>             * GetGradient() const;
    virtual Container<Tensor<DTYPE> *>* GetGradientContainer();

    virtual Tensor<DTYPE>             * GetDelta() const;
    virtual Container<Tensor<DTYPE> *>* GetDeltaContainer();

    int                                 SetModeTraining();
    int                                 SetModeAccumulating();
    int                                 SetModeInferencing();

    int                                 ForwardPropagate(int pTime = 0);
    int                                 BackPropagate(int pTime = 0);

    int                                 ResetResult();
    int                                 ResetGradient();

    void                                PrintInformation();

    void                                SetDeviceCPU();


    // int                                 SetResultOnCPU();
    // int                                 SetGradientOnCPU();
#ifdef __CUDNN__
    // int                                 SetResultOnGPU();
    // int                                 SetGradientOnGPU();

    // void SetDeviceGPU(unsigned int idOfDevice);
    void SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice);
    // void InitializeAttributeForGPU(unsigned int idOfDevice);

    int  ForwardPropagateOnGPU(int pTime = 0);
    int  BackPropagateOnGPU(int pTime = 0);
#endif  // if __CUDNN__
};

#endif  // ifndef __MODULE__

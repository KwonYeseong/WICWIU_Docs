#ifndef LossFunction_H_
#define LossFunction_H_

#include "Module_utils.h"

/*!
@class LossFunction
@details
Not Operator
Network에 속하지 않음, Graph에 포함되지 않는 구조
Network graph, operators : undirected, LossFunction : directed
Network의 output과 Label을 비교해서 계산하는 function
개념적으로 혼동할 수 있음
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> class LossFunction {
private:
    Tensor<DTYPE> *m_aResult; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE> *m_aGradient; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

    Operator<DTYPE> *m_pInputOperator; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE> *m_pInputTensor; ///<   @todo Variable  ////Tensor 꺼내는 작업 Overhead가 크기 때문에 미리 빼 놓음
    // 문서 작성자 : , 작성 날짜 : 2018-

    Operator<DTYPE> *m_pLabel; ///<   @todo Variable  //// 매번 바뀜
    // 문서 작성자 : , 작성 날짜 : 2018-

    std::string m_name; ///<   @todo Variable  //// no name
    // 문서 작성자 : , 작성 날짜 : 2018-

    Device m_Device; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_idOfDevice = -1; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

#ifdef __CUDNN__
    cudnnHandle_t m_pCudnnHandle; ///<   @todo GPU
    // 문서 작성자 : , 작성 날짜 : 2018-
#endif  // if __CUDNN__

public:
    LossFunction(std::string pName = "NO NAME");
    LossFunction(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName = "NO NAME");

    virtual ~LossFunction();

    virtual int            Alloc(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel);
    virtual void           Delete();

    void                   SetResult(Tensor<DTYPE> *pTensor);
    void                   SetGradient(Tensor<DTYPE> *pTensor);

    Tensor<DTYPE>        * GetResult() const;
    Tensor<DTYPE>        * GetGradient() const;
    Operator<DTYPE>      * GetOperator() const;
    Tensor<DTYPE>        * GetTensor() const;
    Operator<DTYPE>      * GetLabel() const;
    std::string            GetName() const;
    virtual Device         GetDevice();
    virtual int            GetDeviceID();

    // For Propagate
    virtual Tensor<DTYPE>* ForwardPropagate(int pTime = 0);
    virtual Tensor<DTYPE>* BackPropagate(int pTime = 0); //// Forward & Backward로 수정 예정, Overhead 방지, speed를 위해

#ifdef __CUDNN__
    virtual Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0);
    virtual Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0);
#endif  // if __CUDNN__

    DTYPE                & operator[](unsigned int index);

    virtual void           SetDeviceCPU();
#ifdef __CUDNN__

    // Setting Supporter
    virtual int    SetResultOnCPU();
    virtual int    SetGradientOnCPU();

    // virtual void   SetDeviceGPU(unsigned int idOfDevice);
    virtual void   SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice);
    virtual void   InitializeAttributeForGPU(unsigned int idOfDevice);

    cudnnHandle_t& GetCudnnHandle();

    // Setting Supporter
    virtual int    SetResultOnGPU(unsigned int idOfDevice);
    virtual int    SetGradientOnGPU(unsigned int idOfDevice);

#endif  // if __CUDNN__

    // reset value
    int ResetResult();
    int ResetGradient();
};

#endif  // LossFunction_H_

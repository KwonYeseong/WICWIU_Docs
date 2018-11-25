#ifndef LossFunction_H_
#define LossFunction_H_

#include "Module_utils.h"

/*!
@class LossFunction 뉴럴 네트워크의 손실 함수를 계산하는 클래스
@details 뉴럴 네트워크의 순전파를 통해 계산된 출력 Tensor와 레이블 값을 비교해 손실 함수를 계산한다.
*/
template<typename DTYPE> class LossFunction {
private:
    Tensor<DTYPE> *m_aResult; ///< LossFunction에서 얻어진 결과 값을 저장하는 Tensor에 대한 포인터
    Tensor<DTYPE> *m_aGradient; ///< LossFunction에서 얻어진 결과 값의 Gradient를 저장하는 Tensor에 대한 포인터

    Operator<DTYPE> *m_pInputOperator; ///< LossFunction의 Input에 해당하는 Operator, 즉 NeuralNetwork의 Output에 해당하는 Operator에 대한 포인터
    Tensor<DTYPE> *m_pInputTensor; ///< NeuralNetwork의 Output에 해당하는 Operator의 Result Tensor에 대한 포인터  ////Tensor 꺼내는 작업 Overhead가 크기 때문에 미리 빼 놓음

    Operator<DTYPE> *m_pLabel; ///< 학습 데이터에 대한 Label 값에 대한 포인터  //// 매번 바뀜

    std::string m_name; ///< LossFunction의 이름을 저장하는 string  //// no name

    Device m_Device; ///< 장치 사용 구분자, CPU 또는 GPU, Device 참고
    int m_idOfDevice = -1; ///< GPU 사용 시, 사용하려는 GPU의 번호. CPU의 경우 -1


#ifdef __CUDNN__
    cudnnHandle_t m_pCudnnHandle; ///< cudnn handler
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

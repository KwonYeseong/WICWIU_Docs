#ifndef __MODULE_H_
#define __MODULE_H_    value

#include "Operator_utils.h"

/*!
@class Module Operator들을 그래프로 구성해 모듈화하는 클래스
@details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 단일 Operator로서 할 수 없는 기능들을 수행하게 한다
@details Module은 하나의 Operator처럼 뉴럴 네트워크 안에서 작동한다
*/
template<typename DTYPE> class Module : public Operator<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_aaExcutableOperator; ///< Module을 구성하는 Operator들 중, 연산에 참여하는 Operator들의 포인터를 저장하는 Container 멤버 변수
    int m_numOfExcutableOperator; ///< Module을 구성하는 Operator들 중, 연산에 참여하는 Operator들의 개수

    Operator<DTYPE> *m_pLastOperator; ///< Module을 구성하는 Operator들 중, 순전파 순서 상 마지막에 해당하는 operator의 포인터

    Device m_Device; ///< 장치 사용 구분자, CPU 또는 GPU, Device 참고
    unsigned int m_idOfDevice = 0; ///< GPU 사용 시, 사용하려는 GPU의 번호. CPU의 경우 -1

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

    virtual Tensor<DTYPE>             * GetResult() const; ////Getter가 virtual로 선언 됨, 마지막 operator의 result를 가져오기 위해서
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

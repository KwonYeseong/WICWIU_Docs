#ifndef OPERATOR_H_
#define OPERATOR_H_

#include "Tensor_utils.h"
#include "Container.h"

/*!
@brief Operator의 현재 상태를 나타내는  enum class
@details TRAINING:학습 중, ACCUMULATING:, INFERENCING:accuracy를 구하는 중
@todo EXTRA
*/
enum Mode {
    TRAINING,
    ACCUMULATING,
    INFERENCING,
};

/*!
@class Operator class
@details 본 프래임워크의 가장 작은 연산 단위.
*/
template<typename DTYPE> class Operator {
private:
    Container<Operator<DTYPE> *> *m_apOutput; ///< Operator의 m_aaResult값을 사용할 Operator들의 주소 값.
    Container<Operator<DTYPE> *> *m_apInput; ///< Operator에 input으로  들어오는 Operator들의 주소 값.
    Container<Tensor<DTYPE> *> *m_aaResult; ///< Operator의 결과 값.
    Container<Tensor<DTYPE> *> *m_aaGradient; ///< Operator의 Gradiuent값들의 Array.
    std::string m_name; ///< Operator에 사용자가 부여한 이름.
    Device m_Device; ///< Operator가 사용하고 있는 Device, 해당 Device의 메모리에 Operator가 있다.
    int m_idOfDevice = -1; ///< m_Device가 GPU일 경우 사용하는 GPU번호.
    Mode m_Mode; ///< Operator의 Mode.
    int m_isParameter; ///< Operator가 파라미터인지 알려주는 값.
    int m_isTrainable; ///< Operator가 학습가능한 Operator인지 알려주는 값.
    // 문서 작성자 : 권예성, 작성 날짜 : 2018-9-22

#ifdef __CUDNN__
    cudnnHandle_t m_pCudnnHandle; ///< cudnn 라이브러리를 가리키는 포인터.
#endif  // __CUDNN__

private:
    int  Alloc();
    int  Alloc(int numInput, ...);
    void Delete();

    int  AddInputEdge(Operator<DTYPE> *pInput);
    int  AddOutputEdge(Operator<DTYPE> *pOutput);


#ifdef __CUDNN__

#endif  // __CUDNN__

public:
    Operator(std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput, std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, Operator<DTYPE> *pInput2, std::string pName = "NO NAME");
    virtual ~Operator();

    int                                   AddEdgebetweenOperators(Operator<DTYPE> *pInput);
    int                                   AddEdgebetweenOperators(int numInput, ...);
    int                                   AddResult(Tensor<DTYPE> *pTensor);
    int                                   AddGradient(Tensor<DTYPE> *pTensor);
    int                                   AddDelta(Tensor<DTYPE> *pTensor);
    int                                   SetResult(Tensor<DTYPE> *pTensor);     // 0 or 1 일 때만 진행 가능
    int                                   SetGradient(Tensor<DTYPE> *pTensor);
    int                                   SetDelta(Tensor<DTYPE> *pTensor);

    int                                   SetDevice(Device pDevice);
    int                                   SetDeviceID(unsigned int idOfDevice);

    int                                   SetIsTensorholder(int pIsParameter);
    int                                   SetIsTrainable(int pIsTrainable);

    virtual int                           SetModeTraining();
    virtual int                           SetModeAccumulating();
    virtual int                           SetModeInferencing();

    virtual Operator<DTYPE>            ** GetOutput();
    virtual Container<Operator<DTYPE> *>* GetOutputContainer();
    virtual Operator<DTYPE>            ** GetInput();
    virtual Container<Operator<DTYPE> *>* GetInputContainer();
    virtual Tensor<DTYPE>               * GetResult() const;
    virtual Container<Tensor<DTYPE> *>  * GetResultContainer();
    virtual Tensor<DTYPE>               * GetGradient() const;
    virtual Container<Tensor<DTYPE> *>  * GetGradientContainer();
    virtual Tensor<DTYPE>               * GetDelta() const;
    virtual Container<Tensor<DTYPE> *>  * GetDeltaContainer();

    std::string                           GetName() const;
    virtual Device                        GetDevice();
    virtual int                           GetDeviceID();
    int                                   GetIsTensorholder();
    int                                   GetIsTrainable();

    virtual int                           ForwardPropagate(int pTime = 0);
    virtual int                           BackPropagate(int pTime = 0);

    // reset value
    virtual int                           ResetResult();
    virtual int                           ResetGradient();

    virtual void                          PrintInformation();

    virtual void                          SetDeviceCPU();

    virtual int                           SetResultOnCPU();
    virtual int                           SetGradientOnCPU();

    int                                   Save(FILE *fileForSave);
    int                                   Load(FILE *fileForLoad);
#ifdef __CUDNN__
    int                                   SetCudnnHandle(cudnnHandle_t& pCudnnHandle);
    virtual int                           SetResultOnGPU(unsigned int idOfDevice);
    virtual int                           SetGradientOnGPU(unsigned int idOfDevice);

    // virtual void                          SetDeviceGPU(unsigned int idOfDevice);
    virtual void                          SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice);
    virtual void                          InitializeAttributeForGPU(unsigned int idOfDevice);

    cudnnHandle_t& GetCudnnHandle();

    virtual int    ForwardPropagateOnGPU(int pTime = 0);
    virtual int    BackPropagateOnGPU(int pTime = 0);


#endif  // if __CUDNN__
};

#endif  // OPERATOR_H_

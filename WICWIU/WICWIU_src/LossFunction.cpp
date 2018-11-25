#include "LossFunction.h"

template class LossFunction<int>;
template class LossFunction<float>;
template class LossFunction<double>;

/*!
@brief LossFunction 클래스 생성자
@details LossFunction의 멤버 변수 포인터들을 NULL값으로 초기화하고, 매개변수로 받은 스트링을 m_name에 저장하고, m_Device를 CPU로 초기화한다.
@param pName m_name에 할당할 LossFunction의 이름, 값을 전달하지 않을 시 "NO NAME"으로 초기화 됨
@return 없음
*/
template<typename DTYPE> LossFunction<DTYPE>::LossFunction(std::string pName) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::LossFunction()" << '\n';
    #endif  // __DEBUG__
    m_aResult        = NULL;
    m_aGradient      = NULL;
    m_pInputOperator = NULL;
    m_pInputTensor   = NULL;
    m_pLabel         = NULL;
    m_name           = pName;
    m_Device         = CPU;
}

/*!
@brief LossFunction 클래스 생성자
@details LossFunction의 멤버 변수 포인터들을 NULL값으로 초기화하고, 매개변수로 받은 스트링을 m_name에 저장하고, m_Device를 CPU로 초기화한다.
@details pOperator와 pLabel을 매개변수로 LossFunction<DTYPE>::Alloc(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel) 메소드를 호출한다.
@param pOperator Alloc 메소드의 매개변수로 전달할 LossFunction의 입력에 해당하는 Operator
@param pLabel Alloc 메소드의 매개변수로 전달할 LossFunction의 입력에 해당하는 레이블
@param pName m_name에 할당할 LossFunction의 이름, 값을 전달하지 않을 시 "NO NAME"으로 초기화 됨
@return 없음
*/
template<typename DTYPE> LossFunction<DTYPE>::LossFunction(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::LossFunction()" << '\n';
    #endif  // __DEBUG__
    m_aResult        = NULL;
    m_aGradient      = NULL;
    m_pInputOperator = NULL;
    m_pInputTensor   = NULL;
    m_pLabel         = NULL;
    m_name           = pName;
    m_Device         = CPU;
    Alloc(pOperator, pLabel);
}

/*!
@brief LossFunction 클래스 소멸자
@details LossFunction<DTYPE>::Delete() 메소드를 호출하고 클래스를 소멸시킨다.
@return 없음
*/
template<typename DTYPE> LossFunction<DTYPE>::~LossFunction() {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::~LossFunction()" << '\n';
    #endif  // __DEBUG__
    this->Delete();
}

/*!
@brief LossFunction의 입력과 레이블을 지정하는 메소드
@details 매개변수로 전달받은 Operator와 Operator의 Result 포인터 값과 레이블 값을 저장한다.
@param pOperator LossFunction의 입력이 되는 Operator
@param plabel LossFunction의 입력이 되는 레이블
@return TRUE
*/
template<typename DTYPE> int LossFunction<DTYPE>::Alloc(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    #endif  // __DEBUG__

    m_pInputOperator = pOperator;
    m_pInputTensor   = m_pInputOperator->GetResult();

    m_pLabel = pLabel;
    return TRUE;
}

/*!
@brief 동적으로 할당받은 LossFunction의 멤버 변수들을 할당 해제하는 메소드
@details Result와 Gradient에 해당하는 Tensor들의 메모리를 할당 해제한다.
@return 없음
*/
template<typename DTYPE> void LossFunction<DTYPE>::Delete() {
    if (m_aResult) {
        delete m_aResult;
        m_aResult = NULL;
    }

    if (m_aGradient) {
        delete m_aGradient;
        m_aGradient = NULL;
    }
}

template<typename DTYPE> void LossFunction<DTYPE>::SetResult(Tensor<DTYPE> *pTensor) {
    m_aResult = pTensor;
}

template<typename DTYPE> void LossFunction<DTYPE>::SetGradient(Tensor<DTYPE> *pTensor) {
    m_aGradient = pTensor;
}

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::GetResult() const {
    return m_aResult;
}

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::GetGradient() const {
    return m_aGradient;
}

template<typename DTYPE> Operator<DTYPE> *LossFunction<DTYPE>::GetOperator() const {
    return m_pInputOperator;
}

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::GetTensor() const {
    return m_pInputTensor;
}

template<typename DTYPE> Operator<DTYPE> *LossFunction<DTYPE>::GetLabel() const {
    return m_pLabel;
}

template<typename DTYPE> std::string LossFunction<DTYPE>::GetName() const {
    return m_name;
}

template<typename DTYPE> Device LossFunction<DTYPE>::GetDevice() {
    return m_Device;
}

template<typename DTYPE> int LossFunction<DTYPE>::GetDeviceID() {
    return m_idOfDevice;
}

/*!
@brief LossFunction의 순전파를 수행하는 메소드
@param pTime 학습 데이터 텐서의 Time 인덱스, 값을 전달하지 않을 시 0으로 초기화 됨
@return NULL
*/
template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::ForwardPropagate(int pTime) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::ForwardPropagate(int pTime)" << '\n';
    std::cout << this->GetName() << '\n';
    #endif  // __DEBUG__
    return NULL;
}

/*!
@brief LossFunction의 역전파를 수행하는 메소드
@param pTime 학습 데이터 텐서의 Time 인덱스, 값을 전달하지 않을 시 0으로 초기화 됨
@return NULL
*/
template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::BackPropagate(int pTime) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::BackPropagate(int pTime)" << '\n';
    std::cout << this->GetName() << '\n';
    #endif  // __DEBUG__
    return NULL;
}

#ifdef __CUDNN__

/*!
@brief
@details
@param
@return
@todo 기술 예정
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return NULL;
}

/*!
@brief
@details
@param
@return
@todo 기술 예정
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::BackPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return NULL;
}

#endif  // __CUDNN__

/*!
@brief [] 연산자 오버로딩
@details 매개변수로 전달받은 index 값 매개변수로 전달하여 Result 텐서에서 []연산자 메소드를 호출한다.
@param index Tensor의 [] 연산자 메소드에 매개변수로 전달할 인덱스 값
@return (*m_aResult)[index]
@see Tensor<DTYPE>::operator[](unsigned int index)
*/
template<typename DTYPE> DTYPE& LossFunction<DTYPE>::operator[](unsigned int index) {
    return (*m_aResult)[index];
}

template<typename DTYPE> void LossFunction<DTYPE>::SetDeviceCPU() {
    m_Device = CPU;

#ifdef __CUDNN__
    this->SetResultOnCPU();
    this->SetGradientOnCPU();
#endif  // __CUDNN__
}


#ifdef __CUDNN__
/*!
@brief Result 텐서의 Device 멤버 변수를 CPU로 설정하는 메소드
@details Result 텐서가 정상적으로 할당되어 있는 경우, Result 텐서의 Device 멤버 변수를 CPU로 설정한다.
@return TRUE
*/
template<typename DTYPE> int LossFunction<DTYPE>::SetResultOnCPU() {
    if (m_aResult) m_aResult->SetDeviceCPU();

    return TRUE;
}

/*!
@brief Gradient 텐서의 Device 멤버 변수를 CPU로 설정하는 메소드
@details Gradient 텐서가 정상적으로 할당되어 있는 경우, Result 텐서의 Device 멤버 변수를 CPU로 설정한다.
@return TRUE
*/
template<typename DTYPE> int LossFunction<DTYPE>::SetGradientOnCPU() {
    if (m_aGradient) m_aGradient->SetDeviceCPU();

    return TRUE;
}

// template<typename DTYPE> void LossFunction<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
// m_Device = GPU;
// this->SetResultOnGPU(idOfDevice);
// this->SetGradientOnGPU(idOfDevice);
// }

/*!
@brief
@details
@param
@return
@todo 기술 예정
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> void LossFunction<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) {
    checkCudaErrors(cudaSetDevice(idOfDevice));
    m_Device       = GPU;
    m_idOfDevice   = idOfDevice;
    m_pCudnnHandle = pCudnnHandle;
    this->SetResultOnGPU(idOfDevice);
    this->SetGradientOnGPU(idOfDevice);
    this->InitializeAttributeForGPU(idOfDevice);
}

/*!
@brief
@return 없음
@todo 기술 예정
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> void LossFunction<DTYPE>::InitializeAttributeForGPU(unsigned int idOfDevice) {}

/*!
@brief
@details
@param
@return
@todo 기술 예정
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int LossFunction<DTYPE >::SetResultOnGPU(unsigned int idOfDevice) {
    if (m_aResult) m_aResult->SetDeviceGPU(idOfDevice);

    return TRUE;
}

/*!
@brief
@details
@param
@return
@todo 기술 예정
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int LossFunction<DTYPE>::SetGradientOnGPU(unsigned int idOfDevice) {
    if (m_aGradient) m_aGradient->SetDeviceGPU(idOfDevice);

    return TRUE;
}

template<typename DTYPE> cudnnHandle_t& LossFunction<DTYPE>::GetCudnnHandle() {
    return m_pCudnnHandle;
}

#endif  // __CUDNN__


/*!
@brief Result 텐서의 ELement를 0으로 초기화하는 메소드
@details Result 텐서의 Device 멤버 변수가 CPU인 경우 CPU 메모리에서 초기화하고, CPU인 경우 GPU 메모리에서 초기화한다.
@return Result 텐서의 Device 멤버 변수가 Invalid한 값을 가지고 있는 경우 FALSE를 그 외의 경우 TRUE를 반환한다.
*/
template<typename DTYPE> int LossFunction<DTYPE>::ResetResult() {
    if (m_Device == CPU) {
        if (m_aResult) m_aResult->Reset();
    }

#ifdef __CUDNN__
    else if (m_Device == GPU) {
        if (m_aResult) m_aResult->Reset(this->GetCudnnHandle());
    }
#endif  // if __CUDNN__

    else return FALSE;

    return TRUE;
}

/*!
@brief Gradient 텐서의 ELement를 0으로 초기화하는 메소드
@details Gradient 텐서의 Device 멤버 변수가 CPU인 경우 CPU 메모리에서 초기화하고, CPU인 경우 GPU 메모리에서 초기화한다.
@return Gradient 텐서의 Device 멤버 변수가 Invalid한 값을 가지고 있는 경우 FALSE를 그 외의 경우 TRUE를 반환한다.
*/
template<typename DTYPE> int LossFunction<DTYPE>::ResetGradient() {
    if (m_Device == CPU) {
        if (m_aGradient) m_aGradient->Reset();
    }

#ifdef __CUDNN__
    else if (m_Device == GPU) {
        if (m_aGradient) m_aGradient->Reset(this->GetCudnnHandle());
    }
#endif  // if __CUDNN__

    else return FALSE;

    return TRUE;
}

// int main(int argc, char const *argv[]) {
// LossFunction<int> *temp1 = new LossFunction<int>("temp1");
// LossFunction<int> *temp2 = new LossFunction<int>(temp1, "temp2");
// LossFunction<int> *temp3 = new LossFunction<int>(temp1, temp2, "temp3");
//
// std::cout << temp3->GetInput()[0]->GetName() << '\n';
// std::cout << temp3->GetInput()[1]->GetName() << '\n';
// std::cout << temp1->GetOutput()[0]->GetName() << '\n';
//
// delete temp1;
// delete temp2;
// delete temp3;
//
// return 0;
// }

#include "LossFunction.h"

template class LossFunction<int>;
template class LossFunction<float>;
template class LossFunction<double>;

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

template<typename DTYPE> LossFunction<DTYPE>::~LossFunction() {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::~LossFunction()" << '\n';
    #endif  // __DEBUG__
    this->Delete();
}

template<typename DTYPE> int LossFunction<DTYPE>::Alloc(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    #endif  // __DEBUG__

    m_pInputOperator = pOperator;
    m_pInputTensor   = m_pInputOperator->GetResult();

    m_pLabel = pLabel;
    return TRUE;
}

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

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::ForwardPropagate(int pTime) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::ForwardPropagate(int pTime)" << '\n';
    std::cout << this->GetName() << '\n';
    #endif  // __DEBUG__
    return NULL;
}

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::BackPropagate(int pTime) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::BackPropagate(int pTime)" << '\n';
    std::cout << this->GetName() << '\n';
    #endif  // __DEBUG__
    return NULL;
}

#ifdef __CUDNN__

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return NULL;
}

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::BackPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return NULL;
}

#endif  // __CUDNN__


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
template<typename DTYPE> int LossFunction<DTYPE>::SetResultOnCPU() {
    if (m_aResult) m_aResult->SetDeviceCPU();

    return TRUE;
}

template<typename DTYPE> int LossFunction<DTYPE>::SetGradientOnCPU() {
    if (m_aGradient) m_aGradient->SetDeviceCPU();

    return TRUE;
}

// template<typename DTYPE> void LossFunction<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
// m_Device = GPU;
// this->SetResultOnGPU(idOfDevice);
// this->SetGradientOnGPU(idOfDevice);
// }

template<typename DTYPE> void LossFunction<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) {
    checkCudaErrors(cudaSetDevice(idOfDevice));
    m_Device       = GPU;
    m_idOfDevice   = idOfDevice;
    m_pCudnnHandle = pCudnnHandle;
    this->SetResultOnGPU(idOfDevice);
    this->SetGradientOnGPU(idOfDevice);
    this->InitializeAttributeForGPU(idOfDevice);
}

template<typename DTYPE> void LossFunction<DTYPE>::InitializeAttributeForGPU(unsigned int idOfDevice) {}

template<typename DTYPE> int LossFunction<DTYPE >::SetResultOnGPU(unsigned int idOfDevice) {
    if (m_aResult) m_aResult->SetDeviceGPU(idOfDevice);

    return TRUE;
}

template<typename DTYPE> int LossFunction<DTYPE>::SetGradientOnGPU(unsigned int idOfDevice) {
    if (m_aGradient) m_aGradient->SetDeviceGPU(idOfDevice);

    return TRUE;
}

template<typename DTYPE> cudnnHandle_t& LossFunction<DTYPE>::GetCudnnHandle() {
    return m_pCudnnHandle;
}

#endif  // __CUDNN__


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

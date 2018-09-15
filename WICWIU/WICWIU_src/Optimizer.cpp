#include "Optimizer.h"

template class Optimizer<int>;
template class Optimizer<float>;
template class Optimizer<double>;

/*!
@brief
@details
@param
@return
@todo Constructor
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> Optimizer<DTYPE>::Optimizer(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    #ifdef __DEBUG__
    std::cout << "Optimizer::Optimizer(Operator<DTYPE> *, float, OptimizeDirection)" << '\n';
    #endif  // __DEBUG__
    m_LearningRate          = 0.f;
    m_OptimizeDirection     = 1;
    m_ppTrainableTensors    = NULL;
    m_TrainableTensorDegree = 0;
    m_weightDecayRate       = 0.f;

    Alloc(pTrainableTensors, pLearningRate, pOptimizeDirection);
}

/*!
@brief
@details
@param
@return
@todo Constructor
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> Optimizer<DTYPE>::Optimizer(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, float pWeightDecayRate, OptimizeDirection pOptimizeDirection) {
    #ifdef __DEBUG__
    std::cout << "Optimizer::Optimizer(Operator<DTYPE> *, float, OptimizeDirection)" << '\n';
    #endif  // __DEBUG__
    m_LearningRate          = 0.f;
    m_OptimizeDirection     = 1;
    m_ppTrainableTensors    = NULL;
    m_TrainableTensorDegree = 0;
    m_weightDecayRate       = 0.f;

    Alloc(pTrainableTensors, pLearningRate, pWeightDecayRate, pOptimizeDirection);
}

/*!
@brief
@details
@param
@return
@todo Constructor
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> Optimizer<DTYPE>::~Optimizer() {
    #ifdef __DEBUG__
    std::cout << "Optimizer::~Optimizer()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

/*!
@brief
@details
@param
@return
@todo Constructor
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Optimizer<DTYPE>::Alloc(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    #ifdef __DEBUG__
    std::cout << "Optimizer::Alloc(Container<Operator<DTYPE> *> *, float , OptimizeDirection )" << '\n';
    #endif  // __DEBUG__
    m_ppTrainableTensors    = pTrainableTensors;
    m_TrainableTensorDegree = pTrainableTensors->GetSize();

    m_LearningRate = pLearningRate;

    if (pOptimizeDirection == MAXIMIZE) m_OptimizeDirection = 1;
    else if (pOptimizeDirection == MINIMIZE) m_OptimizeDirection = -1;

    return TRUE;
}

/*!
@brief
@details
@param
@return
@todo Constructor
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Optimizer<DTYPE>::Alloc(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, float pWeightDecayRate, OptimizeDirection pOptimizeDirection) {
    #ifdef __DEBUG__
    std::cout << "Optimizer::Alloc(Container<Operator<DTYPE> *> *, float , OptimizeDirection )" << '\n';
    #endif  // __DEBUG__
    m_ppTrainableTensors    = pTrainableTensors;
    m_TrainableTensorDegree = pTrainableTensors->GetSize();

    m_LearningRate = pLearningRate;

    if (pOptimizeDirection == MAXIMIZE) m_OptimizeDirection = 1;
    else if (pOptimizeDirection == MINIMIZE) m_OptimizeDirection = -1;

    m_weightDecayRate = pWeightDecayRate;
    // std::cout << "m_weightDecayRate" << m_weightDecayRate << '\n';

    return TRUE;
}

/*!
@brief
@details
@param
@return
@todo Constructor
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Optimizer<DTYPE>::Delete() {
    return TRUE;
}

/*!
@brief
@details
@param
@return
@todo E_Train
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Optimizer<DTYPE>::UpdateParameter() {
    for (int i = 0; i < m_TrainableTensorDegree; i++) {
        UpdateParameter((*m_ppTrainableTensors)[i]);
    }
    return TRUE;
}

#ifdef __CUDNN__

template<typename DTYPE> void Optimizer<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) {
    checkCudaErrors(cudaSetDevice(idOfDevice));
    SetCudnnHandle(pCudnnHandle);
    m_idOfDevice = idOfDevice;
    InitializeAttributeForGPU(idOfDevice);
}

template<typename DTYPE> void Optimizer<DTYPE>::SetCudnnHandle(cudnnHandle_t& pCudnnHandle) {
    m_pCudnnHandle = pCudnnHandle;
}

template<typename DTYPE> int Optimizer<DTYPE>::GetDeviceID() {
    return m_idOfDevice;
}

template<typename DTYPE> cudnnHandle_t& Optimizer<DTYPE>::GetCudnnHandle() {
    return m_pCudnnHandle;
}

template<typename DTYPE> int Optimizer<DTYPE>::UpdateParameterOnGPU() {
    for (int i = 0; i < m_TrainableTensorDegree; i++) {
        UpdateParameterOnGPU((*m_ppTrainableTensors)[i]);
    }
    return TRUE;
}

#endif  // if __CUDNN__


template<typename DTYPE> void Optimizer<DTYPE>::SetLearningRate(float pLearningRate) {
    m_LearningRate = pLearningRate;
}

template<typename DTYPE> void Optimizer<DTYPE>::SetTrainableTensorDegree(int pTrainableTensorDegree) {
    m_TrainableTensorDegree = pTrainableTensorDegree;
}

template<typename DTYPE> void Optimizer<DTYPE>::SetWeightDecayRate(int pWeightDecayRate) {
    m_weightDecayRate = pWeightDecayRate;
}

template<typename DTYPE> float Optimizer<DTYPE>::GetLearningRate()  const {
    return m_LearningRate;
}

template<typename DTYPE> int Optimizer<DTYPE>::GetOptimizeDirection() const {
    return m_OptimizeDirection;
}

template<typename DTYPE> float Optimizer<DTYPE>::GetWeightDecayRate() const {
    return m_weightDecayRate;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Optimizer<DTYPE>::GetTrainableTensor() {
    return m_ppTrainableTensors;
}

template<typename DTYPE> int Optimizer<DTYPE>::GetTrainableTensorDegree() const {
    return m_TrainableTensorDegree;
}

/*!
@brief
@details
@param
@return
@todo E_Train
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Optimizer<DTYPE>::ResetParameterGradient() {
    for (int i = 0; i < m_TrainableTensorDegree; i++) {
        (*m_ppTrainableTensors)[i]->ResetGradient();
    }

    return TRUE;
}

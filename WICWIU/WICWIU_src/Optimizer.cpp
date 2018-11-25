#include "Optimizer.h"

template class Optimizer<int>;
template class Optimizer<float>;
template class Optimizer<double>;

/*!
@brief Optimizer 클래스 생성자
@details 멤버 변수들을 0 또는 NULL로 초기화하고,
@details 전달받은 매개변수를 매개변수로 하여 Optimizer의 Alloc 메소드를 호출한다.
@param pTrainableTensors
@param pLearningRate
@param pOptimizeDirection
@return 없음
@see Optimizer<DTYPE>::Alloc(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection)
@todo 기술 예정
*/
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
@brief Optimizer 클래스 생성자
@details 멤버 변수들을 0 또는 NULL로 초기화하고,
@details 전달받은 매개변수를 매개변수로 하여 Optimizer의 Alloc 메소드를 호출한다.
@param pTrainableTensors
@param pLearningRate
@param pWeightDecayRate
@param pOptimizeDirection
@return 없음
@see Optimizer<DTYPE>::Alloc(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, float pWeightDecayRate, OptimizeDirection pOptimizeDirection)
@todo 기술 예정
*/
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
@brief Optimizer 클래스 소멸자
@details Optimizer<DTYPE>::Delete() 메소드를 호출하고 클래스를 소멸시킨다.
@return 없음
@todo 기술 예정
*/
template<typename DTYPE> Optimizer<DTYPE>::~Optimizer() {
    #ifdef __DEBUG__
    std::cout << "Optimizer::~Optimizer()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

/*!
@brief Optimizer들의 멤버 변수들에 값을 할당하는 메소드
@details 매개변수로 전달 받은 값들을 각각 Trainable Tensor Conatiner, learning rate, Optimize Direction 멤버 변수에 할당한다.
@param pTrainableTensors
@param pLearningRate
@param pOptimizeDirection
@return TRUE
@todo 기술 예정
*/
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
@brief Optimizer들의 멤버 변수들에 값을 할당하는 메소드
@details 매개변수로 전달 받은 값들을 각각 Trainable Tensor Conatiner, learning rate, Optimize Direction, Weight Decay Rate 멤버 변수에 할당한다.
@param pTrainableTensors
@param pLearningRate
@param pWeightDecayRate
@param pOptimizeDirection
@return TRUE
@todo 기술 예정
*/
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

template<typename DTYPE> int Optimizer<DTYPE>::Delete() {
    return TRUE;
}

/*!
@brief
@details
@return TRUE
@todo 기술 예정
*/
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

/*!
@brief
@details
@return TRUE
@todo 기술 예정
*/
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
@return TRUE
@todo 기술 예정
*/
template<typename DTYPE> int Optimizer<DTYPE>::ResetParameterGradient() {
    for (int i = 0; i < m_TrainableTensorDegree; i++) {
        (*m_ppTrainableTensors)[i]->ResetGradient();
    }

    return TRUE;
}

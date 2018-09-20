#include "Module.h"

template class Module<int>;
template class Module<float>;
template class Module<double>;

//////////////////////////////////////////////////////////////////////////////// for private method

/*!
@brief
@details
@param
@return
@todo Constructor
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Module<DTYPE>::Alloc() {
    m_aaExcutableOperator = new Container<Operator<DTYPE> *>();
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
template<typename DTYPE> void Module<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "Module<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__

    if (m_aaExcutableOperator) {
        Operator<DTYPE> **OperatorContainer = m_aaExcutableOperator->GetRawData();

        for (int i = 0; i < m_numOfExcutableOperator; i++) {
            delete OperatorContainer[i];
            OperatorContainer[i] = NULL;
        }
        delete m_aaExcutableOperator;
        m_aaExcutableOperator = NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////// for public method

/*!
@brief
@details
@param
@return
@todo Constructor
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> Module<DTYPE>::Module(std::string pName) : Operator<DTYPE>(pName) {
    #ifdef __DEBUG__
    std::cout << "Module<DTYPE>::Module()" << '\n';
    #endif  // __DEBUG__
    m_aaExcutableOperator    = NULL;
    m_numOfExcutableOperator = 0;
    m_pLastOperator          = NULL;

    Alloc();
}

/*!
@brief
@details
@param
@return
@todo Constructor
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> Module<DTYPE>::~Module() {
    #ifdef __DEBUG__
    std::cout << "Module<DTYPE>::~Module()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

template<typename DTYPE> Operator<DTYPE> *Module<DTYPE>::SetInput(Operator<DTYPE> *pInput) {
    this->AddEdgebetweenOperators(pInput);

    return pInput;
}

template<typename DTYPE> int Module<DTYPE>::SetInput(int pNumOfInput, ...) {
    Operator<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, pNumOfInput);

    for (int i = 0; i < pNumOfInput; i++) {
        temp = va_arg(ap, Operator<DTYPE> *);
        this->SetInput(temp);
    }

    va_end(ap);
    return TRUE;
}

/*!
@brief
@details
@param
@return
@todo E_Graph
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Module<DTYPE>::IsInput(Operator<DTYPE> *pOperator) {
    Container<Operator<DTYPE> *> *m_apInput = this->GetInputContainer();
    int m_InputDegree                       = m_apInput->GetSize();

    for (int i = 0; i < m_InputDegree; i++) {
        if ((*m_apInput)[i] == pOperator) return TRUE;
    }

    return FALSE;
}

/*!
@brief
@details
@param
@return
@todo E_Graph
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Module<DTYPE>::IsValid(Operator<DTYPE> *pOperator) {
    Container<Operator<DTYPE> *> *prevOp = pOperator->GetOutputContainer();
    int numOfOutputEdge                  = prevOp->GetSize();
    int check                            = 0;

    // every Output node is already in Excutable Operator
    for (int i = 0; i < numOfOutputEdge; i++) {
        for (int j = 0; j < m_numOfExcutableOperator; j++) {
            if ((*m_aaExcutableOperator)[j] == (*prevOp)[i]) {
                check++;
                break;
            }
        }

        if (check != (i + 1)) return FALSE;
    }

    return TRUE;
}

/*!
@brief
@details
@param
@return
@todo E_Graph
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> Operator<DTYPE> *Module<DTYPE>::AnalyzeGraph(Operator<DTYPE> *pResultOperator) {
    // BFS
    Container<Operator<DTYPE> *> queue;

    queue.Push(pResultOperator);
    m_pLastOperator = pResultOperator;

    Container<Operator<DTYPE> *> *nextOp = NULL;
    Container<Operator<DTYPE> *> *prevOp = NULL;
    int numOfInputEdge                   = 0;

    Operator<DTYPE> *out = NULL;

    while (queue.GetSize() > 0) {
        out = queue.Pop();

        if (!(this->IsInput(out))) {
            if (this->IsValid(out)) {
                // std::cout << out->GetName() << '\n';

                if (out->GetIsTensorholder()) {
                    this->AddEdgebetweenOperators(out);
                } else {
                    m_aaExcutableOperator->Push(out);
                    m_numOfExcutableOperator++;
                }

                nextOp         = out->GetInputContainer();
                numOfInputEdge = nextOp->GetSize();

                for (int i = 0; i < numOfInputEdge; i++) {
                    prevOp = (*nextOp)[i]->GetOutputContainer();
                    prevOp->Pop(out);

                    queue.Push((*nextOp)[i]);
                }
            } else continue;
        } else continue;
    }
    // std::cout << '\n';

    m_aaExcutableOperator->Reverse();

    // cut output edge of input operator
    // Container<Operator<DTYPE> *> *m_apInput = this->GetInputContainer();
    // int m_InputDegree                       = m_apInput->GetSize();

    // for (int i = 0; i < m_InputDegree; i++) {
    // Operator<DTYPE> *pInput = (*m_apInput)[i];
    // Container<Operator<DTYPE> *> *prevOp = pInput->GetOutputContainer();
    // int numOfOutputEdge                  = prevOp->GetSize();
    //
    // for (int j = 0; j < numOfOutputEdge; j++) {
    // for (int k = 0; k < m_numOfExcutableOperator; k++) {
    // if ((*m_aaExcutableOperator)[j] == (*prevOp)[i]) {
    // (*prevOp)[i] = NULL;
    // }
    // }
    // }
    // }

    // std::cout << "m_aaExcutableOperator : " << '\n';
    //
    // for (int i = 0; i < m_numOfExcutableOperator; i++) {
    // std::cout << (*m_aaExcutableOperator)[i]->GetName() << '\n';
    // }
    // std::cout << '\n';
    //

    return pResultOperator;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Module<DTYPE>::GetExcutableOperatorContainer() {
    return m_aaExcutableOperator;
}

template<typename DTYPE> int Module<DTYPE>::GetNumOfExcutableOperator() {
    return m_numOfExcutableOperator;
}

template<typename DTYPE> Tensor<DTYPE> *Module<DTYPE>::GetResult() const {
    return m_pLastOperator->GetResult();
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Module<DTYPE>::GetResultContainer() {
    return m_pLastOperator->GetResultContainer();
}

template<typename DTYPE> Tensor<DTYPE> *Module<DTYPE>::GetGradient() const {
    return m_pLastOperator->GetGradient();
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Module<DTYPE>::GetGradientContainer() {
    return m_pLastOperator->GetGradientContainer();
}

template<typename DTYPE> Tensor<DTYPE> *Module<DTYPE>::GetDelta() const {
    return m_pLastOperator->GetDelta();
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Module<DTYPE>::GetDeltaContainer() {
    return m_pLastOperator->GetDeltaContainer();
}

template<typename DTYPE> int Module<DTYPE>::SetModeTraining() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->SetModeTraining();
    }
    return TRUE;
}

template<typename DTYPE> int Module<DTYPE>::SetModeAccumulating() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->SetModeAccumulating();
    }
    return TRUE;
}

template<typename DTYPE> int Module<DTYPE>::SetModeInferencing() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->SetModeInferencing();
    }
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
template<typename DTYPE> int Module<DTYPE>::ForwardPropagate(int pTime) {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->ForwardPropagate(pTime);
    }
    return TRUE;
} ////순서대로 실행시킴, 우리가 제공하는 Operator가 아니면 Module을 사용할 수 없음

/*!
@brief
@details
@param
@return
@todo E_Train
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Module<DTYPE>::BackPropagate(int pTime) {
    for (int i = m_numOfExcutableOperator - 1; i >= 0; i--) {
        (*m_aaExcutableOperator)[i]->BackPropagate(pTime);
    }
    return TRUE;
}

/*!
@brief
@details 한 루프(A pair of Forward & backward)마다 초기화 필요, Result가 학습에 이미 사용되었다고 가정, Excutable의 경우만
         Forward -> get Result / Backward -> get Gradient / Update Parameter /
         Parameter Random number / 0 in operators
@param
@return
@todo E_Train
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Module<DTYPE>::ResetResult() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->ResetResult();
    }
    return TRUE;
}

/*!
@brief
@details 한 루프(A pair of Forward & backward)마다 초기화 필요, Gradient가 학습에 이미 사용되었다고 가정, Excutable의 경우만
@param
@return
@todo E_Train
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Module<DTYPE>::ResetGradient() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->ResetGradient();
    }
    return TRUE;
}

/*!
@brief
@details
@param
@return
@todo E_Graph
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> void Module<DTYPE>::PrintInformation() {
    std::cout << this->GetName() << " : ";
    std::cout << this->GetResult()->GetShape() << '\n';

    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        std::cout << "-- ";
        (*m_aaExcutableOperator)[i]->PrintInformation();
    }
}

/*!
@brief
@details
@param
@return
@todo N_Graph
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> void Module<DTYPE>::SetDeviceCPU() {
    this->SetDevice(CPU);

    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->SetDeviceCPU();
    }
}

#ifdef __CUDNN__

// template<typename DTYPE> void Module<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
// this->SetDevice(GPU);
//
// for (int i = 0; i < m_numOfExcutableOperator; i++) {
// (*m_aaExcutableOperator)[i]->SetDeviceGPU(idOfDevice);
// }
// }

/*!
@brief
@details
@param
@return
@todo GPU
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> void Module<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) {
    checkCudaErrors(cudaSetDevice(idOfDevice));
    this->SetDevice(GPU);
    this->SetDeviceID(idOfDevice);
    this->SetCudnnHandle(pCudnnHandle);

    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->SetDeviceGPU(pCudnnHandle, idOfDevice);
    }
}

// template<typename DTYPE> void Module<DTYPE>::InitializeAttributeForGPU(unsigned int idOfDevice) {
// for (int i = 0; i < m_numOfExcutableOperator; i++) {
// (*m_aaExcutableOperator)[i]->InitializeAttributeForGPU(idOfDevice);
// }
// }

/*!
@brief
@details
@param
@return
@todo GPU
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Module<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->ForwardPropagateOnGPU(pTime);
    }
    return TRUE;
}

/*!
@brief
@details
@param
@return
@todo GPU
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int Module<DTYPE>::BackPropagateOnGPU(int pTime) {
    for (int i = m_numOfExcutableOperator - 1; i >= 0; i--) {
        (*m_aaExcutableOperator)[i]->BackPropagateOnGPU(pTime);
    }
    return TRUE;
}

#endif  // if __CUDNN__

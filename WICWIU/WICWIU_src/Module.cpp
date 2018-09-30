#include "Module.h"

template class Module<int>;
template class Module<float>;
template class Module<double>;

//////////////////////////////////////////////////////////////////////////////// for private method

/*!
@brief Module 클래스 안의 Container 멤버 변수를 동적으로 할당해주는 메소드
@details 동적으로 할당 받은 Module 클래스의 Excutable Operator의 메모리를 할당 해제한다.
@return TRUE
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
template<typename DTYPE> int Module<DTYPE>::Alloc() {
    m_aaExcutableOperator = new Container<Operator<DTYPE> *>();
    return TRUE;
}

/*!
@brief 동적으로 할당 받은 Module 클래스의 멤버 변수들을 할당 해제하는 메소드
@details 동적으로 할당 받은 Module 클래스의 Excutable Operator Container의 메모리를 할당 해제한다.
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
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
@brief Module 클래스 생성자
@details 각 멤버 변수들을 초기화하고 Module 클래스를 생성한다.
@details 각 포인터들을 NULL 값으로, 각 정수 타입 변수들은 0으로 초기화하고 Module<DTYPE>::Alloc() 메소드를 호출한다.
@see Module<DTYPE>::Alloc()
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
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
@brief Module 클래스 소멸자
@details 동적으로 할당 받은 Module 클래스의 멤버 변수들을 할당 해제하고 클래스를 소멸시킨다.
@return 없음
@see Module<DTYPE>::Delete()
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
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
@brief 해당 Operator가 Module의 Input인지 확인하는 메소드
@details 매개변수로 받은 Operator가 Module의 Input Container에 포함되어 있는 지 확인한다.
@param pOperator Input 여부를 확인하고자 하는 Operator
@return Input container에 포함되어 있는 경우 TRUE, 포함되어 있지 않는 경우 FALSE를 반환한다.
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
template<typename DTYPE> int Module<DTYPE>::IsInput(Operator<DTYPE> *pOperator) {
    Container<Operator<DTYPE> *> *m_apInput = this->GetInputContainer();
    int m_InputDegree                       = m_apInput->GetSize();

    for (int i = 0; i < m_InputDegree; i++) {
        if ((*m_apInput)[i] == pOperator) return TRUE;
    }

    return FALSE;
}

/*!
@brief 해당 Operator의 Output Operator들이 모듈 그래프에 중복으로 포함되는 지 확인하는 메소드
@details 해당 Operator의 Output container 멤버 변수에 담겨 있는 Operator들이 Module의 Excutable Operator container에 중복되어 포함되어 있는 지 여부를 확인한다.
@param pOperator Output Container 멤버 변수가 Excutable Operator Container에 포함되어 있는 지 확인하고자 하는 Operator
@return 해당 Operator의 Output Container 멤버 변수가 Excutable Operator Container에 중복되어 포함되어 있으면 TRUE를 아니면 FALSE를 반환한다.
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
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
@brief 학습 가능한 형태로 모듈 그래프를 구성해주는 메소드
@details 신경망의 Output에 해당하는 Operator를 매개변수로 받아 너비 우선 탐색으로 신경망 그래프를 구성한다.
@details 매개변수로 받은 신경망의 Output에 해당하는 Operator를 시작으로 신경망의 Input에 해당하는 Output까지 역순으로 NeuralNetwork 클래스의 Container 멤버 변수들에 Operator들을 추가한다.
@details NeuralNetwork 클래스의 Container 멤버 변수들에 Operator들을 모두 추가한 후, 각 Container들의 역순으로 변경한다.
@details Operator 탐색 순서는 너비 우선 탐색을 따르며, 매개변수로 받은 Output Operator부터 해당 Operator의 Input Operator 리스트를 너비 우선 탐색 방식을 이용해 순서대로 진행한다.
@details 신경망의 각 Operator들은 Operator Container에 순서대로 추가되며, 연산에 참여하는 Operator의 경우 Excutable Conatainer에 학습 파라미터에 해당하는 Operator의 경우 Parameter Container에 순서대로 추가된다.
@details 각 Operator들은 NeuralNetwork::IsValid(Operator<DTYPE> *pOperator) 메소드를 이용하여 신경망 그래프 안에서의 중복 여부를 확인하며 중복되는 경우 그래프에 추가하지 않는다.
@param pResultOperator 그래프를 구성하고자 하는 신경망의 Output에 해당하는 Operator
@return 매개변수로 받은 그래프를 구성하고자 하는 신경망의 Output에 해당하는 Operator
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
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
@brief 모듈 그래프의 순전파를 수행하는 메소드
@details Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::ForwardPropagate(int pTime) 메소드를 순서대로 호출한다.
@param pTime 각 ForwardPropagate 메소드에 전달할 Time의 인덱스
@return TRUE
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
template<typename DTYPE> int Module<DTYPE>::ForwardPropagate(int pTime) {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->ForwardPropagate(pTime);
    }
    return TRUE;
} ////순서대로 실행시킴, 우리가 제공하는 Operator가 아니면 Module을 사용할 수 없음

/*!
@brief 모듈 그래프의 역전파를 수행하는 메소드
@details Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::ForwardPropagate(int pTime) 메소드를 역순으로 호출한다.
@param pTime 각 ForwardPropagate 메소드에 전달할 Time의 인덱스
@return TRUE
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
template<typename DTYPE> int Module<DTYPE>::BackPropagate(int pTime) {
    for (int i = m_numOfExcutableOperator - 1; i >= 0; i--) {
        (*m_aaExcutableOperator)[i]->BackPropagate(pTime);
    }
    return TRUE;
}

/*
한 루프(A pair of Forward & backward)마다 초기화 필요, Result가 학습에 이미 사용되었다고 가정, Excutable의 경우만
         Forward -> get Result / Backward -> get Gradient / Update Parameter /
         Parameter Random number / 0 in operators
*/

/*!
@brief 연산에 참여하는 Operator들의 Result Container를 초기화시킨다.
@details Excutable Operator Container에 포함되어 있는 각 Operator들에서 Operator<DTYPE>::ResetResult() 메소드를 호출한다.
@return TRUE
@todo 추가 설명 요
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
template<typename DTYPE> int Module<DTYPE>::ResetResult() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->ResetResult();
    }
    return TRUE;
}

//한 루프(A pair of Forward & backward)마다 초기화 필요, Gradient가 학습에 이미 사용되었다고 가정, Excutable의 경우만
/*!
@brief 연산에 참여하는 Operator들의 Gradient Container를 초기화시킨다.
@details Excutable Operator Container에 포함되어 있는 각 Operator들에서 Operator<DTYPE>::ResetGradient() 메소드를 호출한다.
@return TRUE
@todo 추가 설명 요
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
template<typename DTYPE> int Module<DTYPE>::ResetGradient() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->ResetGradient();
    }
    return TRUE;
}

/*!
@brief 모듈 그래프의 각 구성 요소에 대해 정보를 출력하는 메소드
@return 없음
@see Operator<DTYPE>::PrintInformation()
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
template<typename DTYPE> void Module<DTYPE>::PrintInformation() {
    std::cout << this->GetName() << " : ";
    std::cout << this->GetResult()->GetShape() << '\n';

    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        std::cout << "-- ";
        (*m_aaExcutableOperator)[i]->PrintInformation();
    }
}

/*!
@brief 모듈 그래프 학습에 사용되는 장치를 CPU로 전환하는 메소드
@details Module의 Device 멤버변수를 CPU로 전환하고, Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::SetDeviceCPU() 메소드를 순서대로 호출한다.
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
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

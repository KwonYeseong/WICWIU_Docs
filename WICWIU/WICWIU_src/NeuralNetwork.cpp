#include "NeuralNetwork.h"

template class NeuralNetwork<int>;
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;

//////////////////////////////////////////////////////////////////////////////// for private method

/*!
@brief NeuralNetwork 클래스 안의 Container 멤버 변수들을 동적으로 할당해주는 메소드
@details Operator, Excutable Operator, Input, Parameter Container들 각각에 대해 메모리를 동적으로 할당한다.
@return TRUE
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::Alloc() {
    m_aaOperator          = new Container<Operator<DTYPE> *>();
    m_apExcutableOperator = new Container<Operator<DTYPE> *>();
    m_apInput             = new Container<Operator<DTYPE> *>();
    m_apParameter         = new Container<Operator<DTYPE> *>();
    return TRUE;
}

/*!
@brief 동적으로 할당 받은 NeuralNetwork 클래스의 멤버 변수들을 할당 해제하는 메소드
@details 동적으로 할당 받은 Operator, Excutable Operator, Input, Parameter Container들과 LossFunction, Optimizer의 메모리를 할당 해제한다.
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> void NeuralNetwork<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__
    int size = 0;

    if (m_aaOperator) {
        size = m_aaOperator->GetSize();
        Operator<DTYPE> **OperatorContainer = m_aaOperator->GetRawData();

        for (int i = 0; i < size; i++) {
            if ((*m_aaOperator)[i]) {
                delete OperatorContainer[i];
                OperatorContainer[i] = NULL;
            }
        }
        delete m_aaOperator;
        m_aaOperator = NULL;
    }

    if (m_apExcutableOperator) {
        delete m_apExcutableOperator;
        m_apExcutableOperator = NULL;
    }

    if (m_apInput) {
        delete m_apInput;
        m_apInput = NULL;
    }

    if (m_apParameter) {
        delete m_apParameter;
        m_apParameter = NULL;
    }

    if (m_aLossFunction) {
        delete m_aLossFunction;
        m_aLossFunction = NULL;
    }

    if (m_aOptimizer) {
        delete m_aOptimizer;
        m_aOptimizer = NULL;
    }

#ifdef __CUDNN__
    this->DeleteOnGPU();
#endif  // if __CUDNN__
}

#ifdef __CUDNN__
/*!
@brief
@details
@param
@return
@todo GPU
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int NeuralNetwork<DTYPE>::AllocOnGPU() {
    // checkCudaErrors(cudaSetDevice(2));
    checkCUDNN(cudnnCreate(&m_cudnnHandle));
}

/*!
@brief
@details
@param
@return
@todo GPU
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> void NeuralNetwork<DTYPE>::DeleteOnGPU() {
    // checkCudaErrors(cudaThreadSynchronize());
    // checkCudaErrors(cudaDeviceSynchronize());
    if(m_cudnnHandle) checkCUDNN(cudnnDestroy(m_cudnnHandle));
}


#endif  // if __CUDNN__

//////////////////////////////////////////////////////////////////////////////// for public method


/*!
@brief NeuralNetwork 클래스 생성자
@details 각 멤버 변수들을 초기화하고 NeuralNetwork 클래스를 생성한다.
@details 각 포인터들을 NULL 값으로, 각 정수 타입 변수들은 0으로, Device는 CPU로 초기화한다.
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> NeuralNetwork<DTYPE>::NeuralNetwork() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::NeuralNetwork()" << '\n';
    #endif  // __DEBUG__

    m_aaOperator          = NULL;
    m_apExcutableOperator = NULL;
    m_apInput             = NULL;
    m_apParameter         = NULL;

    m_Operatordegree          = 0;
    m_ExcutableOperatorDegree = 0;
    m_InputDegree             = 0;
    m_ParameterDegree         = 0;

    m_aLossFunction = NULL;
    m_aOptimizer    = NULL;

    m_Device = CPU;

#ifdef __CUDNN__
    m_cudnnHandle = NULL;
#endif  // if __CUDNN__

    Alloc();
}

/*!
@brief NeuralNetwork 클래스 소멸자
@details 동적으로 할당 받은 NeuralNetwork 클래스의 멤버 변수들을 할당 해제하고 클래스를 소멸시킨다.
@return 없음
@see NeuralNetwork::Delete()
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> NeuralNetwork<DTYPE>::~NeuralNetwork() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

/*!
@brief Operator를 신경망의 최초 Input에 추가하는 메소드
@details 매개 변수로 받은 Operator를 NeuralNetwork 클래스의 Operator, Input Container에 추가하고 각 degree를 1만큼 증가시킨다.
@param pInput Input으로 추가하고자 하는 Operator
@return 매개변수로 받은 Operator
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::SetInput(Operator<DTYPE> *pInput) {
    m_aaOperator->Push(pInput);
    m_Operatordegree++;

    m_apInput->Push(pInput);
    m_InputDegree++;
    return pInput;
}

/*!
@brief Operator 리스트를 신경망의 최초 Input에 추가하는 메소드
@details Operator 개수와 Operator 리스트를 매개변수로 받아서, 각각의 Operator에 대해서 NeuralNetwork::SetInput(Operator<DTYPE> *pInput)를 호출한다.
@param pNumOfInput Input에 추가하고자 하는 Operator의 개수
@param ... Input에 추가하고자 하는 Operator의 리스트
@return TRUE
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::SetInput(int pNumOfInput, ...) {
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
@brief 해당 Operator가 Input인지 확인하는 메소드
@details 매개변수로 받은 Opeartor가 Input Container에 포함되어 있는 지 확인한다.
@param pOperator Input 여부를 확인하고자 하는 Operator
@return Input container에 포함되어 있는 경우 TRUE, 포함되어 있지 않는 경우 FALSE를 반환한다.
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::IsInput(Operator<DTYPE> *pOperator) {
    for (int i = 0; i < m_InputDegree; i++) {
        if ((*m_apInput)[i] == pOperator) return TRUE;
    }

    return FALSE;
}

/*!
@brief 해당 Operator가 신경망 그래프에서
@details
@param pOperator
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::IsValid(Operator<DTYPE> *pOperator) {
    Container<Operator<DTYPE> *> *prevOp = pOperator->GetOutputContainer();
    int numOfOutputEdge                  = prevOp->GetSize();
    int check                            = 0;

    // every Output node is already in Excutable Operator
    for (int i = 0; i < numOfOutputEdge; i++) {
        for (int j = 0; j < m_ExcutableOperatorDegree; j++) {
            if ((*m_apExcutableOperator)[j] == (*prevOp)[i]) {
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
@details BFS 로 그래프 분석
@details Result 부터 BFS 순서대로 그래프 탐색
@details Queue, push pop
@details push parameter
@details nextOp 한 층에서 꺼내야 하는 Operator 순서대로 들어감
@details numOfInputEdge 꺼낼 게 몇 개가 존재하는가
@details Input이냐, Valid하냐?(한 번 봤는가 안 봤는가) : excutable, parameter 안에 들어가 있는 지 여부,
@details isParameter? Parameter operator : Excutable operator
@details continue readability
@details reverse 정순으로 세팅하기 위해
@param pResultOperator 신경망의
@return 매개변수로 받은
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AnalyzeGraph(Operator<DTYPE> *pResultOperator) {
    // BFS
    Container<Operator<DTYPE> *> queue;
    queue.Push(pResultOperator);
    Operator<DTYPE> *out                 = NULL;
    Container<Operator<DTYPE> *> *nextOp = NULL;
    int numOfInputEdge                   = 0;

    while (queue.GetSize() > 0) {
        out = queue.Pop();

        if (!(this->IsInput(out))) {
            if (this->IsValid(out)) {
                // std::cout << out->GetName() << '\n';

                m_aaOperator->Push(out);
                m_Operatordegree++;

                if (out->GetIsTensorholder()) {
                    m_apParameter->Push(out);
                    m_ParameterDegree++;
                } else {
                    m_apExcutableOperator->Push(out);
                    m_ExcutableOperatorDegree++;
                }

                nextOp         = out->GetInputContainer();
                numOfInputEdge = nextOp->GetSize();

                // std::cout << numOfInputEdge << '\n';

                for (int i = 0; i < numOfInputEdge; i++) {
                    queue.Push((*nextOp)[i]);
                }
            } else continue;
        } else continue;
    }
    // std::cout << '\n';

    m_aaOperator->Reverse();
    m_apExcutableOperator->Reverse();
    m_apParameter->Reverse();

    // std::cout << "m_aaOperator : " << '\n';
    //
    // for (int i = 0; i < m_Operatordegree; i++) {
    // std::cout << (*m_aaOperator)[i]->GetName() << '\n';
    // }
    // std::cout << '\n';
    //
    // std::cout << "m_apExcutableOperator : " << '\n';
    //
    // for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
    // std::cout << (*m_apExcutableOperator)[i]->GetName() << '\n';
    // }
    // std::cout << '\n';
    //
    // std::cout << "m_apInput : " << '\n';
    //
    // for (int i = 0; i < m_InputDegree; i++) {
    // std::cout << (*m_apInput)[i]->GetName() << '\n';
    // }
    // std::cout << '\n';
    //
    // std::cout << "m_apParameter : " << '\n';
    //
    // for (int i = 0; i < m_ParameterDegree; i++) {
    // std::cout << (*m_apParameter)[i]->GetName() << '\n';
    // }
    // std::cout << '\n';

    return pResultOperator;
}

/*!
@brief 특정 Loss Function을 매개 변수로 받아 이를 신경망의 Loss Function로 지정해주는 메소드
@param pLossFunction 신경망의 Loss Function로 지정하고자 하는 Loss Function
@return 매개변수로 받은 Loss Function
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> LossFunction<DTYPE> *NeuralNetwork<DTYPE>::SetLossFunction(LossFunction<DTYPE> *pLossFunction) {
    m_aLossFunction = pLossFunction;
    return pLossFunction;
}

/*!
@brief 특정 Optimizer를 매개 변수로 받아 이를 신경망의 Optimizer로 지정해주는 메소드
@param pLossFunction 신경망의 Optimizer로 지정하고자 하는 Optimizer
@return 매개변수로 받은 Optimizer
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::SetOptimizer(Optimizer<DTYPE> *pOptimizer) {
    m_aOptimizer = pOptimizer;
    return pOptimizer;
}

/*!
@brief
@details setInput 이후 Feed 순서대로, ... 같은 타입
@param pNumOfInput
@param ...
@return
@todo E_Graph
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::FeedInputTensor(int pNumOfInput, ...) {
    Tensor<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, pNumOfInput);

    for (int i = 0; i < pNumOfInput; i++) {
        temp = va_arg(ap, Tensor<DTYPE> *);
        (*m_apInput)[i]->SetResult(temp);
    }

    va_end(ap);
    return TRUE;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *NeuralNetwork<DTYPE>::GetInputContainer() {
    return m_apInput;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResultOperator() {
    return this->GetResult();
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResult() {
    return m_apExcutableOperator->GetLast();
}

template<typename DTYPE> Container<Operator<DTYPE> *> *NeuralNetwork<DTYPE>::GetExcutableOperatorContainer() {
    return m_apExcutableOperator;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *NeuralNetwork<DTYPE>::GetParameterContainer() {
    return m_apParameter;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *NeuralNetwork<DTYPE>::GetParameter() {
    return m_apParameter;
}

template<typename DTYPE> LossFunction<DTYPE> *NeuralNetwork<DTYPE>::GetLossFunction() {
    return m_aLossFunction;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::GetOptimizer() {
    return m_aOptimizer;
}

/*!
@brief
@details BFS 구조의 Queue, 순서대로 실행
@param pTime
@return
@todo E_Graph
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagate(int pTime) {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ForwardPropagate(pTime);
    }
    m_aLossFunction->ForwardPropagate(pTime);

    return TRUE;
}

/*!
@brief
@details BFS 구조의 Queue, 역순으로 실행
@param pTime
@return
@todo E_Graph
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::BackPropagate(int pTime) {
    m_aLossFunction->BackPropagate(pTime);

    for (int i = m_ExcutableOperatorDegree - 1; i >= 0; i--) {
        (*m_apExcutableOperator)[i]->BackPropagate(pTime);
    }
    return TRUE;
}

/*!
@brief 신경망 내부의 Excutable Operator와
@details
@return
@todo E_Graph
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceCPU() {
    m_Device = CPU;

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetDeviceCPU();
    }
    m_aLossFunction->SetDeviceCPU();
}

/*!
@brief
@details
@return
@todo E_Graph
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeTraining() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeTraining();
    }
}

/*!
@brief
@details
@return
@todo E_Graph
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeAccumulating() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeAccumulating();
    }
}

/*!
@brief
@details
@return
@todo E_Graph
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeInferencing() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeInferencing();
    }
}

/*!
@brief
@details
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::Training() {
    if (m_Device == CPU) {
        this->TrainingOnCPU();
    } else if (m_Device == GPU) {
        this->TrainingOnGPU();
    } else return FALSE;

    return TRUE;
}

/*!
@brief
@details Device CPU를 사용하고 있나, GPU를 쓰고 있나
        각 장치 별 메소드를 사용하겠다
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::Testing() {
    if (m_Device == CPU) {
        this->TestingOnCPU();
    } else if (m_Device == GPU) {
        this->TestingOnGPU();
    } else return FALSE;

    return TRUE;
}

/*!
@brief
@details non-linear Network
@details 한 레이어에 여러 오퍼레이터가 존재하는 경우, ex) Result들의 합을 찾아야 하는 경우, += 사용하는 경우 많음
@details Excutable Operator들의 result를 reset함
@details
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::TrainingOnCPU() {
    this->ResetOperatorResult();
    this->ResetOperatorGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    this->ForwardPropagate();
    this->BackPropagate();

    m_aOptimizer->UpdateParameter();

    return TRUE;
}

/*!
@brief
@details ForwardPropagate만 실행, Gradient reset하지 않음
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::TestingOnCPU() {
    this->ResetOperatorResult();
    this->ResetLossFunctionResult();

    this->ForwardPropagate();
    return TRUE;
}

/*!
@brief
@details
@return
@todo GPU
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int NeuralNetwork<DTYPE>::TrainingOnGPU() {
#ifdef __CUDNN__
    this->ResetOperatorResult();
    this->ResetOperatorGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    this->ForwardPropagateOnGPU();
    this->BackPropagateOnGPU();

    m_aOptimizer->UpdateParameterOnGPU();
#else  // __CUDNN__
    std::cout << "There is no GPU option!" << '\n';
    exit(-1);
#endif  // __CUDNN__

    return TRUE;
}

/*!
@brief
@details
@return
@todo GPU
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int NeuralNetwork<DTYPE>::TestingOnGPU() {
#ifdef __CUDNN__
    this->ResetOperatorResult();
    this->ResetLossFunctionResult();

    this->ForwardPropagateOnGPU();
#else  // __CUDNN__
    std::cout << "There is no GPU option!" << '\n';
    exit(-1);
#endif  // __CUDNN__

    return TRUE;
}

/*!
@brief
@details
@param numOfClass
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> float NeuralNetwork<DTYPE>::GetAccuracy(int numOfClass) {
    Operator<DTYPE> *result = GetResultOperator();
    Operator<DTYPE> *label  = m_aLossFunction->GetLabel();

    int batchsize = label->GetResult()->GetBatchSize();
    int timesize  = label->GetResult()->GetTimeSize();

    Tensor<DTYPE> *pred = result->GetResult();
    Tensor<DTYPE> *ans  = label->GetResult();

    float accuracy = 0.f;

    int pred_index = 0;
    int ans_index  = 0;

    for (int ba = 0; ba < batchsize; ba++) {
        for (int ti = 0; ti < timesize; ti++) {
            pred_index = GetMaxIndex(pred, ba, ti, numOfClass);
            ans_index  = GetMaxIndex(ans, ba, ti, numOfClass);

            if (pred_index == ans_index) {
                accuracy += 1.f;
            }
        }
    }

    return (float)((accuracy / timesize) / batchsize);
}

/*!
@brief
@details
@param data
@param ba
@param ti
@param numOfClass
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::GetMaxIndex(Tensor<DTYPE> *data, int ba, int ti, int numOfClass) {
    Shape *pShape = data->GetShape();
    int    start  = Index5D(pShape, ti, ba, 0, 0, 0);
    int    end    = start + numOfClass;

    // Initial max value is first element
    DTYPE max       = (*data)[start];
    int   max_index = 0;

    for (int dim = start + 1; dim < end; dim++) {
        if ((*data)[dim] > max) {
            max       = (*data)[dim];
            max_index = dim - start;
        }
    }

    return max_index;
}

///////////////////////////////////////////
/*!
@brief
@details
@param numOfClass
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> float NeuralNetwork<DTYPE>::GetTop5Accuracy(int numOfClass) {
    Operator<DTYPE> *result = GetResultOperator();
    Operator<DTYPE> *label  = m_aLossFunction->GetLabel();

    int batchsize = label->GetResult()->GetBatchSize();
    int timesize  = label->GetResult()->GetTimeSize();

    Tensor<DTYPE> *pred = result->GetResult();
    Tensor<DTYPE> *ans  = label->GetResult();

    float top5Accuracy = 0.f;

    int pred_index[5] = { 0, };
    int ans_index     = 0;

    for (int ba = 0; ba < batchsize; ba++) {
        for (int ti = 0; ti < timesize; ti++) {
            pred_index = GetTop5Index(pred, ba, ti, numOfClass);
            ans_index  = GetMaxIndex(ans, ba, ti, numOfClass);

            for(int i = 0; i < 5; i++){
                if (pred_index[i] == ans_index) {
                    top5Accuracy += 1.f;
                    break;
                }
            }
        }
    }

    return (float)((top5Accuracy / timesize) / batchsize);
}

/*!
@brief
@details
@param data
@param ba
@param ti
@param numOfClass
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int* NeuralNetwork<DTYPE>::GetTop5Index(Tensor<DTYPE> *data, int ba, int ti, int numOfClass) {
    Shape *pShape = data->GetShape();
    int    start  = Index5D(pShape, ti, ba, 0, 0, 0);
    int    end    = start + numOfClass;

    // Initial max value is first element
    DTYPE top5Value[5] = { 0, };
    int   top5Index[5] = { 0, };

    // Find 5 top elements
    for (int dim = start; dim < end; dim++) {
        if((*data)[dim]) > Top5Value[0])
        {
            Top5Value[0] = (*data)[dim];
            Top5Index[0] = dim - start;

            for(int i = 0; i < 4; i++){
                if(Top5Value[i] > Top5Value[i+1])
                {
                    std::swap(Top5Value[i], Top5Value[i+1]);
                    std::swap(Top5Index[i], Top5Index[i+1]);
                }
                else
                    break;
            }
        }
    }

    return top5Index;
}
///////////////////////////////////////////

/*!
@brief
@details
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> float NeuralNetwork<DTYPE>::GetLoss() {
    float avg_loss = 0.f;

    int batchsize = m_aLossFunction->GetResult()->GetBatchSize();
    int timesize  = m_aLossFunction->GetResult()->GetTimeSize();

    for (int ba = 0; ba < batchsize; ba++) {
        for (int ti = 0; ti < timesize; ti++) {
            avg_loss += (*m_aLossFunction)[ba] / batchsize / timesize;
        }
    }

    return avg_loss;
}

/*!
@brief
@details
@return 없음
@todo E_Graph
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> void NeuralNetwork<DTYPE>::PrintGraphInformation() {
    std::cout << "Graph Structure: " << "\n\n";

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->PrintInformation();
        std::cout << '\n';
    }

    std::cout << "LossFunction: " << m_aLossFunction->GetName() << '\n';
    // std::cout << "Optimizern: " << m_aOptimizer->GetName() << '\n';
}

/*!
@brief
@details
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetOperatorResult() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ResetResult();
    }
    return TRUE;
}

/*!
@brief
@details
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetOperatorGradient() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ResetGradient();
    }
    return TRUE;
}

/*!
@brief
@details
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetLossFunctionResult() {
    m_aLossFunction->ResetResult();
    return TRUE;
}

/*!
@brief
@details
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetLossFunctionGradient() {
    m_aLossFunction->ResetGradient();
    return TRUE;
}

/*!
@brief
@details
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetParameterGradient() {
    m_aOptimizer->ResetParameterGradient();
    return TRUE;
}

/*!
@brief
@details
@param
@return
@todo N_Graph
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::SerchOperator(std::string pName) {
    std::string name = "NULL";

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        name = (*m_apExcutableOperator)[i]->GetName();

        if (name == pName) return (*m_apExcutableOperator)[i];
    }

    return NULL;
}

/*!
@brief
@details
@param fileForSave
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::Save(FILE *fileForSave) {
    for (int i = 0; i < m_ParameterDegree; i++) {
        // important order
        (*m_apParameter)[i]->Save(fileForSave);
    }
    return TRUE;
}

/*!
@brief
@details
@param fileForLoad
@return
@todo E_Train
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-29
template<typename DTYPE> int NeuralNetwork<DTYPE>::Load(FILE *fileForLoad) {
    for (int i = 0; i < m_ParameterDegree; i++) {
        // important order
        (*m_apParameter)[i]->Load(fileForLoad);
    }
    return TRUE;
}

#ifdef __CUDNN__
/*!
@brief
@details
@param pTime
@return
@todo GPU
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ForwardPropagateOnGPU(pTime);
    }
    m_aLossFunction->ForwardPropagateOnGPU(pTime);

    return TRUE;
}

/*!
@brief
@details
@param pTime
@return
@todo GPU
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int NeuralNetwork<DTYPE>::BackPropagateOnGPU(int pTime) {
    m_aLossFunction->BackPropagateOnGPU(pTime);

    for (int i = m_ExcutableOperatorDegree - 1; i >= 0; i--) {
        (*m_apExcutableOperator)[i]->BackPropagateOnGPU(pTime);
    }
    return TRUE;
}

/*!
@brief
@details
@param idOfDevice
@return
@todo GPU
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
    // std::cout << "NeuralNetwork<DTYPE>::SetModeGPU()" << '\n';
    checkCudaErrors(cudaSetDevice(idOfDevice));

    m_Device = GPU;
    this->AllocOnGPU();

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        // important order
        (*m_apExcutableOperator)[i]->SetDeviceGPU(m_cudnnHandle, idOfDevice);
    }

    for (int i = 0; i < m_ParameterDegree; i++) {
        // important order
        (*m_apParameter)[i]->SetDeviceGPU(m_cudnnHandle, idOfDevice);
    }

    for (int i = 0; i < m_InputDegree; i++) {
        // important order
        (*m_apInput)[i]->SetDeviceGPU(m_cudnnHandle, idOfDevice);
    }

    m_aLossFunction->SetDeviceGPU(m_cudnnHandle, idOfDevice);

    m_aOptimizer->SetDeviceGPU(m_cudnnHandle, idOfDevice);
}

/*!
@brief
@details
@param idOfDevice
@return
@todo GPU
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> int NeuralNetwork<DTYPE>::SetDeviceID(unsigned int idOfDevice) {
    m_idOfDevice = idOfDevice;
    return TRUE;
}

#endif  // __CUDNN__

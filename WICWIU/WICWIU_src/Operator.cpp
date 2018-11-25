
#include "Operator.h"

template class Operator<int>;
template class Operator<float>;
template class Operator<double>;

//////////////////////////////////////////////////////////////////////////////// for private method

/*!
@brief Operator의 맴버 변수들중 포인터들을 메모리에 할당하는 매소드.
@details 맴버변수 m_apOutput, m_apInput, m_aaResult, m_aaGradient들을 메모리에 할당한다
@return 성공시 TRUE.
*/
template<typename DTYPE> int Operator<DTYPE>::Alloc() {
    m_apOutput   = new Container<Operator<DTYPE> *>();
    m_apInput    = new Container<Operator<DTYPE> *>();
    m_aaResult   = new Container<Tensor<DTYPE> *>();
    m_aaGradient = new Container<Tensor<DTYPE> *>();

    return TRUE;
}

/*!
@brief Operator와 다수의 다른 Operator들을 연결시키는 매소드.
@details AddEdgebetweenOperators매소드를 통해 파라미터로 전달받은 Operator의 주소값을 이용해을 연결시킨다.
@param numInput 연결 할 Operator의 갯수.
@param ... Operator와 연결할 input Operator들.
@return 성공 시 TRUE, 실패 시 FALSE.
@ref Operator<DTYPE>::AddEdgebetweenOperators(Operator<DTYPE> *pInput)
*/
template<typename DTYPE> int Operator<DTYPE>::Alloc(int numInput, ...) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    #endif  // __DEBUG__
    Operator<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, numInput);

    int null_count = 0;

    for (int i = 0; i < numInput; i++) {
         temp = va_arg(ap, Operator<DTYPE> *);

         if (!temp) {
           null_count++;
         } else{
           this->AddEdgebetweenOperators(temp);
         }
    }

    va_end(ap);

    if(null_count){
      numInput = numInput - null_count;
      for (int i = 0; i < numInput; i++) {
          delete (*m_apInput)[i];
      }
      delete m_apInput;
      m_apInput = NULL;

      printf("Receive NULL pointer of Operator<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
      return FALSE;
    }

    return TRUE;
}

/*!
@brief Operator를 메모리에 삭제하는 매소드.
@details 메모리에 할당했던 변수들을 삭제하고 포인터들을 NULL로 초기화한다.
*/
template<typename DTYPE> void Operator<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__
    int size = 0;

    if (m_aaResult) {
        size = m_aaResult->GetSize();
        Tensor<DTYPE> **ResultContainer = m_aaResult->GetRawData();

        for (int i = 0; i < size; i++) {
            delete ResultContainer[i];
            ResultContainer[i] = NULL;
        }

        delete m_aaResult;
        m_aaResult = NULL;
    }

    if (m_aaGradient) {
        size = m_aaGradient->GetSize();
        Tensor<DTYPE> **GradientContainer = m_aaGradient->GetRawData();

        for (int i = 0; i < size; i++) {
            if ((*m_aaGradient)[i]) {
                delete GradientContainer[i];
                GradientContainer[i] = NULL;
            }
        }

        delete m_aaGradient;
        m_aaGradient = NULL;
    }

    if (m_apOutput) {
        delete m_apOutput;
        m_apOutput = NULL;
    }

    if (m_apInput) {
        delete m_apInput;
        m_apInput = NULL;
    }
}

/*!
@brief Operator의 m_apInput을 설정하는 함수.
@details 다른 Operator의 주소 값을 받아 Operator의 input값(m_apInput)으로 설정한다.
@param pInput input으로 설정 할 Operator들의 주소 값.
@return 성공 시 TRUE, 실패 시 FALSE
@ref int Push(DTYPE pElement)
*/
template<typename DTYPE> int Operator<DTYPE>::AddInputEdge(Operator<DTYPE> *pInput) {
    try {
        m_apInput->Push(pInput);
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

/*!
@brief Operator의 m_apOutput을 설정하는 함수.
@details 다른 Operator의 주소 값을 받아 Operator의 output값(m_apOutput)으로 설정한다.
@param pOutput Operator의 output을 input으로 사용할 Operator들의 주소 값.
@return 성공 시 TRUE, 실패 시 FALSE
@ref int Push(DTYPE pElement)
*/
template<typename DTYPE> int Operator<DTYPE>::AddOutputEdge(Operator<DTYPE> *pOutput) {
    try {
        m_apOutput->Push(pOutput);
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

//////////////////////////////////////////////////////////////////////////////// for public method

/*!
@brief Operator의 생성자.
@details 파라미터로 전달 받은 pName을 m_name으로 설정 하고 나머지는 변수들은 NULL, CPU, TRAINING으로 초기화 한다.
@param pName 사용자가 설정 할 Operator의 이름.
*/
template<typename DTYPE> Operator<DTYPE>::Operator(std::string pName) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAINING;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    Alloc();
}

/*!
@brief Operator의 생성자.
@details파라미터로 전달 받은 pName을 m_name으로 설정 하고 나머지는 변수들은 NULL, CPU, TRAINING으로 초기화 한 뒤, AddEdgebetweenOperators를 통해 Operator들을 서로 연결한다.
@param pInput Operator와 연결 할 Operator들의 주소 값들.
@param pName 사용자가 설정 할 Operator의 이름.
@ref Operator<DTYPE>::AddEdgebetweenOperators(int numInput, ...)
*/
template<typename DTYPE> Operator<DTYPE>::Operator(Operator<DTYPE> *pInput, std::string pName) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAINING;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    Alloc();
    AddEdgebetweenOperators(1, pInput);
}

/*!
@brief Operator의 생성자.
@details파라미터로 전달 받은 pName을 m_name으로 설정 하고 나머지는 변수들은 NULL, CPU, TRAINING으로 초기화 한 뒤, AddEdgebetweenOperators를 통해 Operator들을 서로 연결한다.
@param pInput0 Operator와 연결 할 Operator들의 주소 값들.
@param pInput1 Operator와 연결 할 Operator들의 주소 값들.
@param pName 사용자가 설정 할 Operator의 이름.
@ref Operator<DTYPE>::AddEdgebetweenOperators(int numInput, ...)
*/
template<typename DTYPE> Operator<DTYPE>::Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAINING;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    Alloc();
    AddEdgebetweenOperators(2, pInput0, pInput1);
}

/*!
@brief Operator의 생성자.
@details파라미터로 전달 받은 pName을 m_name으로 설정 하고 나머지는 변수들은 NULL, CPU, TRAINING으로 초기화 한 뒤, AddEdgebetweenOperators를 통해 Operator들을 서로 연결한다.
@param pInput0 Operator와 연결 할 Operator들의 주소 값들.
@param pInput1 Operator와 연결 할 Operator들의 주소 값들.
@param pInput2 Operator와 연결 할 Operator들의 주소 값둘.
@param pName 사용자가 설정 할 Operator의 이름.
@ref Operator<DTYPE>::AddEdgebetweenOperators(int numInput, ...)
*/
template<typename DTYPE> Operator<DTYPE>::Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, Operator<DTYPE> *pInput2, std::string pName) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAINING;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    Alloc();
    AddEdgebetweenOperators(3, pInput0, pInput1, pInput2);
}

/*!
@brief Operator의 소멸자
@details Delete 매소드를 이용해 삭제한다.
*/
// 문서 작성자 : 권예성, 작성 날짜 : 2018-9-22
template<typename DTYPE> Operator<DTYPE>::~Operator() {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::~Operator()" << '\n';
    #endif  // __DEBUG__
    this->Delete();
}

/*!
@brief Operator와 다른 Operator들을 서로 연결한다.
@details AddInputEdge, AddOutputEdge 매소드를 이옹해 Operator와 다른 Operator 연결한다.
@param pInput 연결 할 다른 Operator의 주소 값.
@ref int Operator<DTYPE>::AddInputEdge(Operator<DTYPE> *pInput), int Operator<DTYPE>::AddOutputEdge(Operator<DTYPE> *pOutput)
@return 성공 시 TRUE.
*/
template<typename DTYPE> int Operator<DTYPE>::AddEdgebetweenOperators(Operator<DTYPE> *pInput) {
    this->AddInputEdge(pInput);
    pInput->AddOutputEdge(this);
    return TRUE;
}

/*!
@brief Operator와 다른 Operator들을 서로 연결한다.
@details AddEdgebetweenOperators 매소드를 이옹해 Operator와 다수의 다른 Operator들을 연결한다.
@param numInput 연결 할 input들의 갯수.
@param ... 연결할 다수의 다른 Operator.
@ref int Operator<DTYPE>::AddEdgebetweenOperators(Operator<DTYPE> *pInput)
@return 성공 시 TRUE, 실패 시 FALSE.
*/
template<typename DTYPE> int Operator<DTYPE>::AddEdgebetweenOperators(int numInput, ...) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    #endif  // __DEBUG__
    Operator<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, numInput);

    int null_count = 0;

    for (int i = 0; i < numInput; i++) {
         temp = va_arg(ap, Operator<DTYPE> *);

         if (!temp) {
           null_count++;
         } else{
           this->AddEdgebetweenOperators(temp);
         }
    }

    va_end(ap);

    if(null_count){
      numInput = numInput - null_count;
      for (int i = 0; i < numInput; i++) {
          delete (*m_apInput)[i];
      }
      delete m_apInput;
      m_apInput = NULL;

      printf("Receive NULL pointer of Operator<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
      return FALSE;
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetResult(Tensor<DTYPE> *pTensor) {
    if (m_aaResult->GetSize()) {
        Tensor<DTYPE> *temp = m_aaResult->Pop();
        delete temp;
        temp = NULL;
    }

    m_aaResult->Push(pTensor);
    return TRUE;
}

/*!
@brief 파라미터로 받은 Tensor를 결과 값으로 설정한다.
@details 파라미터로 받은 pTensor를 m_aaResult애 저장한다.
@param pTensor m_aaResult에 저장 할 Tensor.
@return 성공 시 TRUE.
*/
template<typename DTYPE> int Operator<DTYPE>::AddResult(Tensor<DTYPE> *pTensor) {
    m_aaResult->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetGradient(Tensor<DTYPE> *pTensor) {
    if (m_aaGradient->GetSize()) {
        Tensor<DTYPE> *temp = m_aaGradient->Pop();
        delete temp;
        temp = NULL;
    }

    m_aaGradient->Push(pTensor);
    return TRUE;
}

/*!
@brief 파라미터로 받은 Tensor를 gradient값으로 설정한다.
@details 파라미터로 받은 pTensor를 m_aaGradient에 저장한다.
@param pTensor m_aaGradient에 저장 할 Tensor.
@return 성공 시 TRUE.
*/
// 문서 작성자 : 권예성, 작성 날짜 : 2018-9-22
template<typename DTYPE> int Operator<DTYPE>::AddGradient(Tensor<DTYPE> *pTensor) {
    m_aaGradient->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetDelta(Tensor<DTYPE> *pTensor) {
    if (m_aaGradient->GetSize()) {
        Tensor<DTYPE> *temp = m_aaGradient->Pop();
        delete temp;
        temp = NULL;
    }

    m_aaGradient->Push(pTensor);
    return TRUE;
}

/*!
@brief 파라미터로 받은 Tensor를 Delta값으로 설정한다.
@details 파라미터로 받은 pTensor를 m_aaGradient에 저장한다.
@param pTensor m_aaGradient에 저장 할 Tensor.
@return 성공 시 TRUE.
*/
template<typename DTYPE> int Operator<DTYPE>::AddDelta(Tensor<DTYPE> *pTensor) {
    m_aaGradient->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetDevice(Device pDevice) {
    m_Device = pDevice;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetDeviceID(unsigned int idOfDevice) {
    m_idOfDevice = idOfDevice;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetIsTensorholder(int pIsParameter) {
    m_isParameter = pIsParameter;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetIsTrainable(int pIsTrainable) {
    m_isTrainable = pIsTrainable;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetModeTraining() {
    m_Mode = TRAINING;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetModeAccumulating() {
    m_Mode = ACCUMULATING;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetModeInferencing() {
    m_Mode = INFERENCING;
    return TRUE;
}

template<typename DTYPE> Operator<DTYPE> **Operator<DTYPE>::GetOutput() {
    return m_apOutput->GetRawData();
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Operator<DTYPE>::GetOutputContainer() {
    return m_apOutput;
}

template<typename DTYPE> Operator<DTYPE> **Operator<DTYPE>::GetInput() {
    return m_apInput->GetRawData();
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Operator<DTYPE>::GetInputContainer() {
    return m_apInput;
}

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetResult() const {
    return (*m_aaResult)[0];
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Operator<DTYPE>::GetResultContainer() {
    return m_aaResult;
}

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetGradient() const {
    return (*m_aaGradient)[0];
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Operator<DTYPE>::GetGradientContainer() {
    return m_aaGradient;
}

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetDelta() const {
    return (*m_aaGradient)[0];
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Operator<DTYPE>::GetDeltaContainer() {
    return m_aaGradient;
}

template<typename DTYPE> std::string Operator<DTYPE>::GetName() const {
    return m_name;
}

template<typename DTYPE> Device Operator<DTYPE>::GetDevice() {
    return m_Device;
}

template<typename DTYPE> int Operator<DTYPE>::GetDeviceID() {
    return m_idOfDevice;
}

template<typename DTYPE> int Operator<DTYPE>::GetIsTensorholder() {
    return m_isParameter;
}

template<typename DTYPE> int Operator<DTYPE>::GetIsTrainable() {
    return m_isTrainable;
}

/*!
@brief  ForwardPropagate 매소드. 실제 구현은 파생 클래스에서 정의된다.
@param pTime ForwardPropagate할 데이터의 Time값.
@return 성공 시 TRUE.
*/

template<typename DTYPE> int Operator<DTYPE>::ForwardPropagate(int pTime) {
    #ifdef __DEBUG__
    std::cout << "thread number : " << pThreadNum << '\n';
    std::cout << "number of thread : " << this->GetNumOfThread() << '\n';
    #endif  // __DEBUG__
    return TRUE;
}

/*!
@brief  BackwardPropagate 매소드. 실제 구현은 파생 클래스에서 정의된다.
@param pTime forwardPropagate했던 데이터의 Time값.
@return 성공 시 TRUE.
*/
template<typename DTYPE> int Operator<DTYPE>::BackPropagate(int pTime) {
    #ifdef __DEBUG__
    std::cout << "thread number : " << pThreadNum << '\n';
    std::cout << "number of thread : " << this->GetNumOfThread() << '\n';
    #endif  // __DEBUG__
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::ResetResult() {
    // Tensorholder의 경우는 하면 안된다.
    int size = m_aaResult->GetSize();

    if (m_Device == CPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaResult)[i]->Reset();
        }
    }

#ifdef __CUDNN__
    else if (m_Device == GPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaResult)[i]->Reset(this->GetCudnnHandle());
        }
    }
#endif  // if __CUDNN__

    else return FALSE;

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::ResetGradient() {
    int size = m_aaGradient->GetSize();

    if (m_Device == CPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaGradient)[i]->Reset();
        }
    }

#ifdef __CUDNN__
    else if (m_Device == GPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaGradient)[i]->Reset(this->GetCudnnHandle());
        }
    }
#endif  // if __CUDNN__

    else return FALSE;

    return TRUE;
}

/*!
@brief Operator정보(이름과 Shape)를 출력한다.
*/
template<typename DTYPE> void Operator<DTYPE>::PrintInformation() {
    std::cout << this->GetName() << " : ";
    std::cout << this->GetResult()->GetShape() << '\n';
}

template<typename DTYPE> void Operator<DTYPE>::SetDeviceCPU() {
    this->SetDevice(CPU);

    this->SetResultOnCPU();
    this->SetGradientOnCPU();
}

template<typename DTYPE> int Operator<DTYPE>::SetResultOnCPU() {
    // Tensorholder의 경우는 하면 안된다.
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->SetDeviceCPU();
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetGradientOnCPU() {
    int size = m_aaGradient->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaGradient)[i]->SetDeviceCPU();
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::Save(FILE *fileForSave) {
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->Save(fileForSave);
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::Load(FILE *fileForLoad) {
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->Load(fileForLoad);
    }

    return TRUE;
}

#ifdef __CUDNN__

template<typename DTYPE> int Operator<DTYPE>::SetCudnnHandle(cudnnHandle_t& pCudnnHandle) {
    m_pCudnnHandle = pCudnnHandle;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetResultOnGPU(unsigned int idOfDevice) {
    // Tensorholder의 경우는 하면 안된다.
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->SetDeviceGPU(idOfDevice);
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetGradientOnGPU(unsigned int idOfDevice) {
    int size = m_aaGradient->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaGradient)[i]->SetDeviceGPU(idOfDevice);
    }

    return TRUE;
}

template<typename DTYPE> void Operator<DTYPE>::InitializeAttributeForGPU(unsigned int idOfDevice) {}

/*!
@brief Operator가 GPU에서 연산 될 수 있도록 하는 매소드.
@details Operator의 정보들을 지정된 GPU의 메모리로 복사한다.
@param pCudnnHandle cudnn 라이브러리를 가리키는 포인터.
@param idOfDevice 사용 할 GPU 번호
*/
template<typename DTYPE> void Operator<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) {
    checkCudaErrors(cudaSetDevice(idOfDevice));
    this->SetCudnnHandle(pCudnnHandle);
    this->SetDevice(GPU);
    this->SetDeviceID(idOfDevice);
    this->SetResultOnGPU(idOfDevice);
    this->SetGradientOnGPU(idOfDevice);
    this->InitializeAttributeForGPU(idOfDevice);
}

template<typename DTYPE> cudnnHandle_t& Operator<DTYPE>::GetCudnnHandle() {
    return m_pCudnnHandle;
}

/*!
@brief  ForwardPropagateOnGPU 매소드. 실제 구현은 파생 클래스에서 정의된다.
@param pTime ForwardPropagate할 데이터의 Time값.
@return 성공 시 TRUE.
*/
template<typename DTYPE> int Operator<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << "Operator<DTYPE>::ForwardPropagateOnGPU(int)" << '\n';
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return TRUE;
}

/*!
@brief  BackwardPropagateOnGPU 매소드. 실제 구현은 파생 클래스에서 정의된다.
@param pTime ForwardPropagate할 데이터의 Time값.
@return 성공 시 TRUE.
*/
template<typename DTYPE> int Operator<DTYPE>::BackPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << "Operator<DTYPE>::BackPropagateOnGPU(int)" << '\n';
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return TRUE;
}

#endif  // __CUDNN__


// int main(int argc, char const *argv[]) {
// Operator<int> *temp1 = new Operator<int>("temp1");
// Operator<int> *temp2 = new Operator<int>(temp1, "temp2");
// Operator<int> *temp3 = new Operator<int>(temp1, temp2, "temp3");
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

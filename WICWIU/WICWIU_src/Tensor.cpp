#include "Tensor.h"

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<unsigned char>;

//////////////////////////////////////////////////////////////////////////////// for private method


/*!
@brief Tensor의 Shape 정의 및 하는 메소드.
@details pShape의 형태을 갖는 Tensor를 정의하고 그 Tensor의 m_aLongArray를 메모리에 할당한는 메소드.
@param *pShape alloc하는 Tensor의 Shape형태.
@param pAnswer Time 사용 여부
@return 성공 시 TRUE, 실패 시 FALSE.
*/
template<typename DTYPE> int Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    if (pShape == NULL) {
        printf("Receive NULL pointer of Shape class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape    = pShape;
        m_IsUseTime = pAnswer;

        int rank = pShape->GetRank();

        int pTime            = 1;
        int pCapacityPerTime = 1;

        if (m_IsUseTime == UseTime) {
            pTime = (*pShape)[0];

            for (int i = 1; i < rank; i++) {
                pCapacityPerTime *= (*pShape)[i];
            }
        } else if (m_IsUseTime == NoUseTime) {
            for (int i = 0; i < rank; i++) {
                pCapacityPerTime *= (*pShape)[i];
            }
        } else return FALSE;

        m_aLongArray = new LongArray<DTYPE>(pTime, pCapacityPerTime);
    }

    m_Device = CPU;

    return TRUE;
}

/*!
*@brief Tensor를 deep copy하는 메소드.
*@param *pTensor deep copy할 대상 Tensor.
*@return 성공 시 TRUE, 실패 시 FALSE.
*/
template<typename DTYPE> int Tensor<DTYPE>::Alloc(Tensor<DTYPE> *pTensor) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Alloc(Tensor<DTYPE> *pTensor)" << '\n';
    #endif  // __DEBUG__

    if (pTensor == NULL) {
        printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape     = new Shape(pTensor->GetShape());
        m_aLongArray = new LongArray<DTYPE>(pTensor->GetLongArray());
        m_Device     = pTensor->GetDevice();
        m_IsUseTime  = pTensor->GetIsUseTime();
    }

    return TRUE;
}


/*!
*@brief Tensor를 메모리상에서 삭제하는 메소드.
*@details 메모리에 할당되어 있는 Tensor데이터를 삭제한다.
*@return 없음.
*/
template<typename DTYPE> void Tensor<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__

    if (m_aShape) {
        delete m_aShape;
        m_aShape = NULL;
    }

    if (m_aLongArray) {
        delete m_aLongArray;
        m_aLongArray = NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////// for public method

/*!
*@brief 5D-Tensor 생성자.
*@details 원하는 형태의 Shape정보(Time, Batch, Channel, Row, Column들의 Dimension)를 받아 Tensor를 Alloc하는 생성자.
*@param pSize0 Time의 Dimension
*@param pSize1 Batch의 Dimension
*@param pSize2 Channel의 Dimension
*@param pSize3 Row의 Dimension
*@param pSize4 Column의 Dimension
*@param IsUseTime Time사용 여부.
*@return 없음.
*@see Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer),
*     Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4)
*/
template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    Alloc(new Shape(pSize0, pSize1, pSize2, pSize3, pSize4), pAnswer);
}

/*!
*@brief 4D-Tensor 생성자.
*@details 원하는 형태의 Shape정보(Batch, Channel, Row, Column들의 Dimension)를 받아 Tensor를 Alloc하는 생성자.
*@param pSize0 Batch의 Dimension
*@param pSize1 Channel의 Dimension
*@param pSize2 Row의 Dimension
*@param pSize3 Column의 Dimension
*@param IsUseTime Time사용 여부.
*@return 없음.
*@see Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer),
      Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3)
*/
template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    Alloc(new Shape(pSize0, pSize1, pSize2, pSize3), pAnswer);
}

/*!
*@brief 3D-Tensor 생성자.
*@details 원하는 형태의 Shape정보(Channel, Row, Column들의 Dimension)를 받아 Tensor를 Alloc하는 생성자.
*@param pSize0 Channel의 Dimension
*@param pSize1 Row의 Dimension
*@param pSize2 Column의 Dimension
*@param IsUseTime Time사용 여부
*@return 없음.
*@see Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer),
*     Shape::Shape(int pSize0, int pSize1, int pSize2)
*/
template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    Alloc(new Shape(pSize0, pSize1, pSize2), pAnswer);
}

/*!
*@brief 2D-Tensor 생성자.
*@details 원하는 형태의 Shape정보(Row, Column들의 Dimension)를 받아 Tensor를 Alloc하는 생성자.
*@param pSize0 Row의 Dimension
*@param pSize1 Column의 Dimension
*@param IsUseTime Time사용 여부
*@return 없음.
*@see Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer),
*     Shape::Shape(int pSize0, int pSize1)
*/
template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    Alloc(new Shape(pSize0, pSize1), pAnswer);
}

/*!
*@brief 1D-Tensor 생성자.
*@details 원하는 형태의 Shape정보(Column의 Dimension)를 받아 Tensor를 Alloc하는 생성자.
*@param pSize0 Column의 Dimension
*@param IsUseTime Time사용 여부
*@return 없음.
*@see Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer),
*     Shape::Shape(int pSize0)
*/
template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    Alloc(new Shape(pSize0), pAnswer);
}

/*!
*@brief 입력받은 Shape형태의 Tensor를 Alloc하는 생성자.
*@param pShape 할당할 Tensor의 shape
*@param IsUseTime Time사용 여부
*@return 없음.
*@see Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer)
*/
template<typename DTYPE> Tensor<DTYPE>::Tensor(Shape *pShape, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(Shape *pShape, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    Alloc(pShape, pAnswer);
}

/*!
*@brief Tensor 생성자.
*@details Tensor를 deep copy하는 생성자.
*@param pTensor deep copy할 대상 Tensor
*@return 없음.
*@see Tensor(Tensor<DTYPE> *pTensor)
*/
template<typename DTYPE> Tensor<DTYPE>::Tensor(Tensor *pTensor) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(Tensor *pTensor)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    Alloc(pTensor);
}

/*!
*@brief Tensor 소멸자.
*@details Delete를 사용하여 해당 Tensor를 메모리에서 삭제한다.
*@return 없음.
*@see void Delete()
*/
template<typename DTYPE> Tensor<DTYPE>::~Tensor() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::~Tensor()" << '\n';
    #endif  // __DEBUG__

    Delete();
}

/*!
*@brief Tensor의 Shape을 반환.
*@return m_aShape
*/
template<typename DTYPE> Shape *Tensor<DTYPE>::GetShape() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetShape()" << '\n';
    #endif  // __DEBUG__

    return m_aShape;
}

/*!
*@brief Tensor의 Shape의 Rank를 반환.
*@return m_Rank
*@see Shape::GetRank()
*/
template<typename DTYPE> int Tensor<DTYPE>::GetRank() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetRank()" << '\n';
    #endif  // __DEBUG__

    return m_aShape->GetRank();
}

/*!
*@brief Tensor의 Shape의 Rank의 Dimension를 반환.
*@return Shape의 m_aDim[pRanknum]
*@see Shape::GetDim()
*/
template<typename DTYPE> int Tensor<DTYPE>::GetDim(int pRanknum) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetDim(int pRanknum)" << '\n';
    #endif  // __DEBUG__

    return m_aShape->GetDim(pRanknum);
}

/*!
*@brief Tensor의 m_aLongArray를 반환.
*@return m_aLongArray
*/
template<typename DTYPE> LongArray<DTYPE> *Tensor<DTYPE>::GetLongArray() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetLongArray()" << '\n';
    #endif  // __DEBUG__

    return m_aLongArray;
}

/*!
*@brief Tensor의 m_aLongArray의 Capacity를 반환.
*@return LongArray<DTYPE>의  m_TimeSize * m_CapacityPerTime
*@see LongArray<DTYPE>::GetCapacity()
*/
template<typename DTYPE> int Tensor<DTYPE>::GetCapacity() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetCapacity()" << '\n';
    #endif  // __DEBUG__

    return m_aLongArray->GetCapacity();
}

/*!
*@brief Tensor의 m_aLongArray의 특정 위치의 값를 반환.
*@return LongArray<DTYPE>의 m_aaHostLongArray[index / m_CapacityPerTime][index % m_CapacityPerTime]
*@see LongArray<DTYPE>::GetElement(unsigned int index)
*/
template<typename DTYPE> int Tensor<DTYPE>::GetElement(unsigned int index) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetElement(unsigned int index)" << '\n';
    #endif  // __DEBUG__

    return m_aLongArray->GetElement(index);
}

/*!
*@brief []연산자 오버로딩
*@details m_aLongArray의 특정 위치에 있는 값을 return할 수 있게 한다.
*@details 단, m_Device가 GPU일 시 CPU로 바꿔 준 후 값을 찾아 반환한다.
*/
template<typename DTYPE> DTYPE& Tensor<DTYPE>::operator[](unsigned int index) {
    #ifdef __CUDNN__
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::operator[](unsigned int index)" << '\n';

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    return (*m_aLongArray)[index];
}

/*!
*@brief Tensor가 어느 Device를 사용하는지 나타낸다.
*@return m_Device
*@see enum Device
*/
template<typename DTYPE> Device Tensor<DTYPE>::GetDevice() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetDevice()" << '\n';
    #endif  // __DEBUG__

    return m_Device;
}

/*!
*@brief Tensor의 Time사용 여부를 반환한다.
*@return m_IsUseTime
*/
template<typename DTYPE> IsUseTime Tensor<DTYPE>::GetIsUseTime() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetIsUseTime()" << '\n';
    #endif  // __DEBUG__

    return m_IsUseTime;
}

/*!
*@brief Tensor의 LongArray<DTYPE> 데이터를 반환한다.
*@details m_Device가 GPU일 경우 CPU로 바꾸고 LongArray<DTYPE>의 m_aaHostLongArray[pTime]를 반환한다
*@return m_aaHostLongArray[pTime]
*@see *LongArray<DTYPE>::GetCPULongArray(unsigned int pTime)
*/
template<typename DTYPE> DTYPE *Tensor<DTYPE>::GetCPULongArray(unsigned int pTime) {
    #ifdef __CUDNN__
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetCPULongArray(unsigned int pTime)" << '\n';

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    return m_aLongArray->GetCPULongArray(pTime);
}

/*!
*@brief Tensor의 time의 rank를 반환한다.
*@details Tensor가 time 축을 사용하는 경우, Tensor의 time의 rank 값을 반환한다 @ref Shape::GetRank()
*@return 값이 존재할 시 Tensor의 time의 rank, 존재하지 않을 시 0
*/
template<typename DTYPE> int Tensor<DTYPE>::GetTimeSize() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetTimeSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[0];
    else return 0;
}

/*!
*@brief 삭제 예정
*/
template<typename DTYPE> int Tensor<DTYPE>::GetBatchSize() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetBatchSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[1];
    else return 0;
}

/*!
@brief 삭제 예정
*/
template<typename DTYPE> int Tensor<DTYPE>::GetChannelSize() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetChannelSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[2];
    else return 0;
}

/*!
@brief 삭제 예정
*/
template<typename DTYPE> int Tensor<DTYPE>::GetRowSize() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetRowSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[3];
    else return 0;
}

/*!
@brief 삭제 예정
*/
template<typename DTYPE> int Tensor<DTYPE>::GetColSize() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetColSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[4];
    else return 0;
}

/*!
*@brief Tensor의 Shape을 변경하는 메소드
*@details 현재 Tensor의 Shape을 다른 형태의 Shape으로 재정의한다.
*         단, Time, Batch, Channel, Row, Column의 곱은 같게 유지되어야 한다.
*@param pSize0 Time의 Dimension
*@param pSize1 Batch의 Dimension
*@param pSize2 Channel의 Dimension
*@param pSize3 Row의 Dimension
*@param pSize4 Column의 Dimension
*@return 성공 시 TRUE, 실패 시 FALSE.
*@see Shape::ReShape(int pRank, ...)
*/
template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1 * pSize2 * pSize3 * pSize4;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(5, pSize0, pSize1, pSize2, pSize3, pSize4);
    }

    return TRUE;
}

/*!
*@brief Tensor의 Shape을 변경하는 메소드
*@details 현재 Tensor의 Shape을 다른 형태의 Shape으로 재정의한다.
*         단, Batch, Channel, Row, Column의 곱은 같게 유지되어야 한다.
*@param pSize0 Batch의 Dimension
*@param pSize1 Channel의 Dimension
*@param pSize2 Row의 Dimension
*@param pSize3 Column의 Dimension
*@return 성공 시 TRUE, 실패 시 FALSE.
*@see Shape::ReShape(int pRank, ...)
*/
template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1 * pSize2 * pSize3;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(4, pSize0, pSize1, pSize2, pSize3);
    }

    return TRUE;
}

/*!
*@brief Tensor의 Shape을 변경하는 메소드
*@details 현재 Tensor의 Shape을 다른 형태의 Shape으로 재정의한다.
*         단, Channel, Row, Column의 곱은 같게 유지되어야 한다.
*@param pSize0 Channel의 Dimension
*@param pSize1 Row의 Dimension
*@param pSize2 Column의 Dimension
*@return 성공 시 TRUE, 실패 시 FALSE.
*@see Shape::ReShape(int pRank, ...)
*/
template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1 * pSize2;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(3, pSize0, pSize1, pSize2);
    }

    return TRUE;
}

/*!
*@brief Tensor의 Shape을 변경하는 메소드
*@details 현재 Tensor의 Shape을 다른 형태의 Shape으로 재정의한다.
*         단, Row, Column의 곱은 같게 유지되어야 한다.
*@param pSize0 Row의 Dimension
*@param pSize1 Column의 Dimension
*@return 성공 시 TRUE, 실패 시 FALSE.
*@see Shape::ReShape(int pRank, ...)
*/
template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(2, pSize0, pSize1);
    }

    return TRUE;
}

/*!
*@brief Tensor의 Shape을 변경하는 메소드
*@details 현재 Tensor의 Shape을 다른 형태의 Shape으로 재정의한다.
*         단, Column값은 같게 유지되어야 한다.
*@param pSize0 Column의 Dimension
*@return 성공 시 TRUE, 실패 시 FALSE.
*@see Shape::ReShape(int pRank, ...)
*/
template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(1, pSize0);
    }

    return TRUE;
}

/*!
*@brief Tensor의 m_aLongArray값들을 초기화하는 메소드
*@details Tensor의 m_Device를 CPU지정한 뒤, m_aLongArray값을 모두 0으로 초기화한다.
*@return 없음.
*@see Tensor<DTYPE>::SetDeviceCPU()
*/
template<typename DTYPE> void Tensor<DTYPE>::Reset() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Reset()" << '\n';
    #endif  // __DEBUG__

    int capacity = GetCapacity();

    #ifdef __CUDNN__
    # if __DEBUG__

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    for (int i = 0; i < capacity; i++) {
        (*m_aLongArray)[i] = 0;
    }
}

/*!
*@brief Tensor의 m_Device를 CPU로 지정하는 메소드.
*@details Tensor의 m_Device 및 m_aLongArray와 m_aShape의 m_Device를 CPU로 지정한다.
*@return 없음.
*@see Shape::SetDeviceCPU(), LongArray<DTYPE>::SetDeviceCPU()
*/
template<typename DTYPE> void Tensor<DTYPE>::SetDeviceCPU() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::SetDeviceCPU()" << '\n';
    #endif  // __DEBUG__

    m_Device = CPU;
    m_aLongArray->SetDeviceCPU();
    m_aShape->SetDeviceCPU();
}

/*!
*@brief Tensor의 데이터를 파일에 저장
*@details 장치를 CPU로 전환하고 m_aLongArray의 Save(fileForsave) 메소드 호출 @ref LongArray<DTYPE>::Save(FILE *fileForSave)
*@param fileForSave 불러오기 할 FILE에 대한 포인터
*@return 성공 시 TRUE
*@see LongArray<DTYPE>::Save(FILE *fileForSave), Tensor<DTYPE>::SetDeviceCPU()
*/
template<typename DTYPE> int Tensor<DTYPE>::Save(FILE *fileForSave) {
    #ifdef __CUDNN__
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::Save(FILE *fileForSave)" << '\n';

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    m_aLongArray->Save(fileForSave);


    return TRUE;
}

/*!
*@brief 파일에서 Tensor의 데이터를 불러옴
*@details 장치를 CPU로 전환하고 m_aLongArray의 Load(fileForsave) 메소드 호출 @ref LongArray<DTYPE>::Load(FILE *fileForSave)
*@param fileForload 불러오기 할 FILE에 대한 포인터
*@return 성공 시 TRUE
*@see LongArray<DTYPE>::Load(FILE *fileForSave), Tensor<DTYPE>::SetDeviceCPU()
*/
template<typename DTYPE> int Tensor<DTYPE>::Load(FILE *fileForLoad) {
    #ifdef __CUDNN__
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::Load(FILE *fileForSave)" << '\n';

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    m_aLongArray->Load(fileForLoad);

    return TRUE;
}

#ifdef __CUDNN__

/*!
@*brief 작성 예정
*/
template<typename DTYPE> void Tensor<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::SetDeviceGPU()" << '\n';
    # endif // __DEBUG__
    checkCudaErrors(cudaSetDevice(idOfDevice));

    m_Device     = GPU;
    m_idOfDevice = idOfDevice;
    m_aLongArray->SetDeviceGPU(idOfDevice);
    m_aShape->SetDeviceGPU(idOfDevice);
}

/*!
*@brief 작성 예정
*/
template<typename DTYPE> DTYPE *Tensor<DTYPE>::GetGPUData(unsigned int pTime) {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetGPUData(unsigned int pTime)" << '\n';

    if (m_Device == CPU) {
        printf("Warning! Tensor is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");

        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetGPUData" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # else // if __DEBUG__

    if (m_Device == CPU) {
        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetGPUData" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # endif // __DEBUG__

    return m_aLongArray->GetGPUData(pTime);
}

/*!
*@brief 작성 예정
*/
template<typename DTYPE> cudnnTensorDescriptor_t& Tensor<DTYPE>::GetDescriptor() {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetDescriptor()" << '\n';

    if (m_Device == CPU) {
        printf("Warning! Tensor is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");

        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetDescriptor" << '\n';
            exit(-1);
        }
        this->SetDeviceGPU(m_idOfDevice);
    }

    # else // if __DEBUG__

    if (m_Device == CPU) {
        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetDescriptor" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # endif // __DEBUG__

    return m_aShape->GetDescriptor();
}

/*!
*@brief 작성 예정
*/
template<typename DTYPE> void Tensor<DTYPE>::Reset(cudnnHandle_t& pCudnnHandle) {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::Reset(cudnnHandle_t& pCudnnHandle)" << '\n';

    if (m_Device == CPU) {
        printf("Warning! Tensor is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");

        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : Reset" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # else // if __DEBUG__

    if (m_Device == CPU) {
        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : Reset" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # endif // __DEBUG__

    int pTime                     = this->GetTimeSize();
    cudnnTensorDescriptor_t pDesc = this->GetDescriptor();
    DTYPE *pDevLongArray          = NULL;
    float  zero                   = 0.f;

    for (int i = 0; i < pTime; i++) {
        pDevLongArray = this->GetGPUData(i);
        checkCUDNN(cudnnAddTensor(pCudnnHandle,
                                  &zero, pDesc, pDevLongArray,
                                  &zero, pDesc, pDevLongArray));
    }
}

#endif  // if __CUDNN__

////////////////////////////////////////////////////////////////////////////////static method

/*!
*@brief 정규분포를 따르는 임의의 값을 갖는 Tensor를 생성
*@details pSize 0~4를 매개변수로 받는 Shape를 생성하고, mean을 평균으로 stddev를 표준편차로 갖는 정규분포에서 임의로 얻어진 값으로 초기화된 LongArray<DTYPE>를 갖는 텐서를 생성한다.
*@param pSize0 생성하려는 Tensor의 Shape를 구성하는 Time의 Dimension
*@param pSize1 생성하려는 Tensor의 Shape를 구성하는 Batch의 Dimension
*@param pSize2 생성하려는 Tensor의 Shape를 구성하는 Channel의 Dimension
*@param pSize3 생성하려는 Tensor의 Shape를 구성하는 Row의 Dimension
*@param pSize4 생성하려는 Tensor의 Shape를 구성하는 Column의 Dimension
*@param mean 임의로 생성되는 값이 따르는 정규분포의 평균
*@param stddev 임의로 생성되는 값이 따르는 정규분포의 표준 편차
*@param pAnswer 생성하려는 Tensor의 time 축 사용 유무, @ref IsUseTime
*@return 정규분포를 따르는 임의의 값을 갖는 Tensor
*@see Tensor<DTYPE>::Random_normal(Shape * pShape, float mean, float stddev, IsUseTime pAnswer = UseTime)
*/
template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Random_normal(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, float mean, float stddev, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Random_normal()" << '\n';
    #endif  // __DEBUG__

    return Tensor<DTYPE>::Random_normal(new Shape(pSize0, pSize1, pSize2, pSize3, pSize4), mean, stddev, pAnswer);
}

/*!
*@brief 정규분포를 따르는 임의의 값을 갖는 Tensor를 생성
*@details pShape를 Shape로 갖고, mean을 평균으로 stddev를 표준편차로 갖는 정규분포에서 임의로 얻어진 값으로 초기화된 LongArray<DTYPE>를 갖는 텐서를 생성한다.
*@param pShape 생성하려는 Tensor의 Shape
*@param mean 임의로 생성되는 값이 따르는 정규분포의 평균
*@param stddev 임의로 생성되는 값이 따르는 정규분포의 표준 편차
*@param pAnswer 생성하려는 Tensor의 time 축 사용 유무, @ref IsUseTime
*@return 정규분포를 따르는 임의의 값을 갖는 Tensor
*@see Tensor<DTYPE>::Tensor(Shape *pShape, IsUseTime pAnswer)
*/
template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Random_normal(Shape *pShape, float mean, float stddev, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Random_normal()" << '\n';
    #endif  // __DEBUG__
    srand((unsigned)time(NULL));

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pShape, pAnswer);

    int   capacity = temp->GetCapacity();
    DTYPE v1 = 0.f, v2 = 0.f, mid_result = 0.f;

    // Random number generator on normal distribution
    for (int i = 0; i < capacity; i++) {
        do {
            v1         = 2 * ((float)rand() / RAND_MAX) - 1; // -1.0 ~ 1.0 까지의 값
            v2         = 2 * ((float)rand() / RAND_MAX) - 1; // -1.0 ~ 1.0 까지의 값
            mid_result = v1 * v1 + v2 * v2;
        } while (mid_result >= 1 || mid_result == 0);

        mid_result = sqrt((-2 * log(mid_result)) / mid_result);
        mid_result = v1 * mid_result;
        (*temp)[i] = (stddev * mid_result) + mean;
    }

    return temp;
}

/*!
*@brief 0으로 초기화된 Tensor를 생성
*@details pSize 0~4를 매개변수로 받는 Shape를 생성하고, 그 생성된 Shape를 매개변수로 갖고 0으로 초기화된 LongArray<DTYPE>를 갖는 텐서를 생성한다.
*@param pSize0 생성하려는 Tensor의 Shape를 구성하는 Time의 Dimension
*@param pSize1 생성하려는 Tensor의 Shape를 구성하는 Batch의 Dimension
*@param pSize2 생성하려는 Tensor의 Shape를 구성하는 Channel의 Dimension
*param pSize3 생성하려는 Tensor의 Shape를 구성하는 Row의 Dimension
*@param pSize4 생성하려는 Tensor의 Shape를 구성하는 Column의 Dimension
*@param pAnswer 생성하려는 Tensor의 time 축 사용 유무, @ref IsUseTime
*@return 0으로 초기화된 Tensor
*@see Tensor<DT1YPE>::Zeros(Shape *pShape, IsUseTime pAnswer)
*/
template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Zeros(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';
    #endif  // __DEBUG__

    return Tensor<DTYPE>::Zeros(new Shape(pSize0, pSize1, pSize2, pSize3, pSize4), pAnswer);
}

/*!
@brief 0으로 초기화된 Tensor를 생성
@details pShape를 Shape로 갖고, 0으로 초기화된 LongArray<DTYPE>를 갖는 텐서를 생성한다.
@param pShape 생성하려는 Tensor의 Shape
@param pAnswer 생성하려는 Tensor의 Time 축 사용 유무, @ref IsUseTime
@return 0으로 초기화된 Tensor
@see Tensor<DTYPE>::Tensor(Shape *pShape, IsUseTime pAnswer)
*/
template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Zeros(Shape *pShape, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';
    #endif  // __DEBUG__

    return new Tensor<DTYPE>(pShape, pAnswer);
}

/*!
*@brief 상수로 초기화된 Tensor를 생성
*@details pSize 0~4를 매개변수로 받는 Shape를 생성하고, 그 생성된 Shape를 매개변수로 갖고 constant로 초기화된 LongArray<DTYPE>를 갖는 텐서를 생성한다.
*@param pSize0 생성하려는 Tensor의 Shape를 구성하는 Time의 Dimension
*@param pSize1 생성하려는 Tensor의 Shape를 구성하는 Batch의 Dimension
*@param pSize2 생성하려는 Tensor의 Shape를 구성하는 Channel의 Dimension
*@param pSize3 생성하려는 Tensor의 Shape를 구성하는 Row의 Dimension
*@param pSize4 생성하려는 Tensor의 Shape를 구성하는 Col의 Dimension
*@param constant 생성하려는 Tensor를 구성하는 LongArray<DTYPE>의 element가 갖게되는 상수
*@param pAnswer 생성하려는 Tensor의 time 축 사용 유무, @ref IsUseTime
*@return 상수로 초기화된 텐서
*@see Tensor<DTYPE>::Tensor(Shape *pShape, DTYPE constant, IsUseTime pAnswer)
*/
template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Constants(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, DTYPE constant, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';
    #endif  // __DEBUG__

    return Tensor<DTYPE>::Constants(new Shape(pSize0, pSize1, pSize2, pSize3, pSize4), constant, pAnswer);
}

/*!
*@brief 상수로 초기화된 Tensor를 생성
*@details pShape를 Shape로 갖고, constant로 초기화된 LongArray<DTYPE>를 갖는 텐서를 생성한다.
*@param pShape 생성하려는 Tensor의 Shape
*@param constant 생성하려는 Tensor를 구성하는 LongArray<DTYPE>의 element가 갖게되는 상수
*@param pAnswer 생성하려는 Tensor의 time 축 사용 유무, @ref IsUseTime
*@return 상수로 초기화된 텐서
*@see Tensor<DTYPE>::Tensor(Shape *pShape, IsUseTime pAnswer)
*/
template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Constants(Shape *pShape, DTYPE constant, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';
    #endif  // __DEBUG__

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pShape, pAnswer);

    int capacity = temp->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = constant;
    }

    return temp;
}

// example code
// int main(int argc, char const *argv[]) {
// Tensor<float> *left  = Tensor<float>::Constants(1, 2, 3, 3, 3, 2);
// Tensor<float> *right = Tensor<float>::Random_normal(1, 1, 3, 1, 1, 0.0, 0.1);
// Tensor<float> *dst   = Tensor<float>::Zeros(1, 2, 3, 3, 3);
//
// std::cout << left << '\n';
// std::cout << right << '\n';
// std::cout << dst << '\n';
//
// Tensor<float>::BroadcastAdd(left, right);
//
// std::cout << dst << '\n';
//
// return 0;
// }

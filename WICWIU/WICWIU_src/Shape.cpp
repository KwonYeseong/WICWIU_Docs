<<<<<<< current
#include "Shape.h"


//////////////////////////////////////////////////////////////////////////////// for private method

/*!
@brief
@details
@param pRank
@param ...
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
int Shape::Alloc(int pRank, ...) {
    #ifdef __DEBUG__
    std::cout << "Shape::Alloc(int pRank, ...)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRank > 0) m_Rank = pRank;
        else throw pRank;
    } catch (int e) {
        printf("Receive invalid rank value %d in %s (%s %d)\n", e, __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    try {
        m_aDim = new int[m_Rank];
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    va_list ap;
    va_start(ap, pRank);

    // need to check compare between pRank value and number of another parameter
    for (int i = 0; i < pRank; i++) {
        // need to check whether int or not
        m_aDim[i] = va_arg(ap, int);
    }
    va_end(ap);

    m_Device = CPU;

    return TRUE;
}

/*!
@brief
@details
@param pShape
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
int Shape::Alloc(Shape *pShape) {
    #ifdef __DEBUG__
    std::cout << "Shape::Alloc(Shape *pShape)" << '\n';
    #endif  // __DEBUG__

    try {
        m_Rank = pShape->GetRank();
        m_aDim = new int[m_Rank];

        for (int i = 0; i < m_Rank; i++) {
            m_aDim[i] = (*pShape)[i];
        }
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    m_Device = pShape->GetDevice();

#ifdef __CUDNN__
    m_idOfDevice = pShape->GetDeviceID();

    if (m_Device == GPU) SetDeviceGPU(m_idOfDevice);
#endif  // if __CUDNN__

    return TRUE;
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
void Shape::Delete() {
    #ifdef __DEBUG__
    std::cout << "Shape::Delete()" << '\n';
    #endif  // __DEBUG__

    if (m_aDim) {
        delete[] m_aDim;
        m_aDim = NULL;
    }

#ifdef __CUDNN__
    DeleteOnGPU();
#endif  // if __CUDNN__
}

/*!
@brief
@details
@param idOfDevice
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
#ifdef __CUDNN__
int Shape::AllocOnGPU(unsigned int idOfDevice) {
    # if __DEBUG__
    std::cout << "Shape::AllocOnGPU()" << '\n';
    # endif // __DEBUG__

    m_idOfDevice = idOfDevice;
    checkCudaErrors(cudaSetDevice(idOfDevice));

    if (m_desc == NULL) {
        if (m_Rank == 5) {
            checkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
            checkCUDNN(cudnnSetTensor4dDescriptor(m_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                  m_aDim[1], m_aDim[2], m_aDim[3], m_aDim[4]));
        } else if (m_Rank == 4) {
            checkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
            checkCUDNN(cudnnSetTensor4dDescriptor(m_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                  m_aDim[0], m_aDim[1], m_aDim[2], m_aDim[3]));
        } else ;
    } else ;

    return TRUE;
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
void Shape::DeleteOnGPU() {
    # if __DEBUG__
    std::cout << "Shape::DeleteOnGPU()" << '\n';
    # endif // __DEBUG__

    if (m_desc) {
        checkCUDNN(cudnnDestroyTensorDescriptor(m_desc));
        m_desc = NULL;
    }
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
int Shape::ReShapeOnGPU() {
    # if __DEBUG__
    std::cout << "Shape::ReShapeOnGPU()" << '\n';
    # endif // __DEBUG__

    DeleteOnGPU();

    if (m_idOfDevice == -1) {
        std::cout << "you need to set device GPU first before : ReShapeOnGPU" << '\n';
        exit(-1);
    }
    AllocOnGPU(m_idOfDevice);

    return TRUE;
}

#endif  // if __CUDNN__

//////////////////////////////////////////////////////////////////////////////// for public method
/*!
@brief
@details
@param pSize0
@param pSize1
@param pSize2
@param pSize3
@param pSize4
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(5, pSize0, pSize1, pSize2, pSize3, pSize4);
}

/*!
@brief
@details
@param pSize0
@param pSize1
@param pSize2
@param pSize3
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(4, pSize0, pSize1, pSize2, pSize3);
}

/*!
@brief
@details
@param pSize0
@param pSize1
@param pSize2
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
Shape::Shape(int pSize0, int pSize1, int pSize2) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(3, pSize0, pSize1, pSize2);
}

/*!
@brief
@details
@param pSize0
@param pSize1
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
Shape::Shape(int pSize0, int pSize1) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(2, pSize0, pSize1);
}

/*!
@brief
@details
@param pSize0
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
Shape::Shape(int pSize0) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(1, pSize0);
}

/*!
@brief
@details
@param pShape
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
Shape::Shape(Shape *pShape) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(Shape *pShape)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(pShape);
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
Shape::~Shape() {
    #ifdef __DEBUG__
    std::cout << "Shape::~Shape()" << '\n';
    #endif  // __DEBUG__

    Delete();
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
int Shape::GetRank() {
    #ifdef __DEBUG__
    std::cout << "Shape::GetRank()" << '\n';
    #endif  // __DEBUG__

    return m_Rank;
}

/*!
@brief
@details
@param pRanknum
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
int Shape::GetDim(int pRanknum) {
    #ifdef __DEBUG__
    std::cout << "Shape::GetDim(int pRanknum)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRanknum >= 0) return m_aDim[pRanknum];
        else throw;
    }
    catch (...) {
        printf("Receive invalid pRanknum value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        exit(0);
        // return FALSE;
    }
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
Device Shape::GetDevice() {
    #ifdef __DEBUG__
    std::cout << "Shape::GetDevice()" << '\n';
    #endif  // __DEBUG__

    return m_Device;
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
int Shape::GetDeviceID() {
    return m_idOfDevice;
}

/*!
@brief
@details
@param pRanknum
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
int& Shape::operator[](int pRanknum) {
    #ifdef __DEBUG__
    std::cout << "Shape::operator[](int pRanknum)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRanknum >= 0) return m_aDim[pRanknum];
        else throw;
    }
    catch (...) {
        printf("Receive invalid pRanknum value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        exit(0);
        // return FALSE;
    }
}

/*!
@brief
@details
@param pSize0
@param pSize1
@param pSize2
@param pSize3
@param pSize4
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
int Shape::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    #ifdef __DEBUG__
    std::cout << "Shape::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4)" << '\n';
    #endif  // __DEBUG__

    return ReShape(5, pSize0, pSize1, pSize2, pSize3, pSize4);
}

/*!
@brief
@details
@param pRank
@param ...
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
int Shape::ReShape(int pRank, ...) {
    #ifdef __DEBUG__
    std::cout << "Shape::ReShape(int pRank, ...)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRank > 0) m_Rank = pRank;
        else throw pRank;
    } catch (int e) {
        printf("Receive invalid rank value %d in %s (%s %d)\n", e, __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    try {
        if (m_aDim) {
            delete[] m_aDim;
            m_aDim = NULL;
        }
        m_aDim = new int[m_Rank];
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    va_list ap;
    va_start(ap, pRank);

    for (int i = 0; i < pRank; i++) {
        m_aDim[i] = va_arg(ap, int);
    }

    va_end(ap);

#ifdef __CUDNN__

    if (m_Device == GPU) ReShapeOnGPU();
#endif  // if __CUDNN__

    return TRUE;
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
int Shape::SetDeviceCPU() {
    #ifdef __DEBUG__
    std::cout << "Shape::SetDeviceCPU()" << '\n';
    #endif  // __DEBUG__

    m_Device = CPU;

    return TRUE;
}

#ifdef __CUDNN__

/*!
@brief
@details
@param idOfDevice
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
int Shape::SetDeviceGPU(unsigned int idOfDevice) {
    # if __DEBUG__
    std::cout << "Shape::SetDeviceGPU()" << '\n';
    # endif // __DEBUG__

    m_Device = GPU;

    if (m_desc == NULL) AllocOnGPU(idOfDevice);

    return TRUE;
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
cudnnTensorDescriptor_t& Shape::GetDescriptor() {
    # if __DEBUG__
    std::cout << "Shape::GetDescriptor()" << '\n';

    if (m_Device == CPU) {
        printf("Warning! Tensor is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");

        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetDescriptor" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # else // if __DEBUG__

    if (m_Device == CPU) {
        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetDescriptor" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # endif // __DEBUG__

    return m_desc;
}

#endif  // __CUDNN__

/*!
@brief
@details
@param pOS
@param pShape
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-06
std::ostream& operator<<(std::ostream& pOS, Shape *pShape) {
    #ifdef __DEBUG__
    std::cout << "std::ostream& operator<<(std::ostream& pOS, Shape *pShape)" << '\n';
    #endif  // __DEBUG__

    int rank = pShape->GetRank();

    pOS << "Rank is " << rank << ", Dimension is [";

    for (int i = 0; i < rank; i++) pOS << (*pShape)[i] << ", ";
    pOS << "]";
    return pOS;
}
=======
#include "Shape.h"


//////////////////////////////////////////////////////////////////////////////// for private method

/*!
@brief Shape 클래스를 동적으로 할당하는 메소드
@details Shape의 Rank와 각 축의 dimension을 매개변수로 받아 Shape 클래스의
@param pRank
@param ...
@return 성공 시 TRUE, 실패 시 FALSE
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int Shape::Alloc(int pRank, ...) {
    #ifdef __DEBUG__
    std::cout << "Shape::Alloc(int pRank, ...)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRank > 0) m_Rank = pRank;
        else throw pRank;
    } catch (int e) {
        printf("Receive invalid rank value %d in %s (%s %d)\n", e, __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    try {
        m_aDim = new int[m_Rank];
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    va_list ap;
    va_start(ap, pRank);

    // need to check compare between pRank value and number of another parameter
    for (int i = 0; i < pRank; i++) {
        // need to check whether int or not
        m_aDim[i] = va_arg(ap, int);
    }
    va_end(ap);

    m_Device = CPU;

    return TRUE;
}

/*!
@brief Shape 클래스를 깊은 복사하는 메소드
@param pShape 깊은 복사의 대상이 되는 클래스에 대한 포인터
@return 할당 성공 시 TRUE, 실패 시 FALSE
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int Shape::Alloc(Shape *pShape) {
    #ifdef __DEBUG__
    std::cout << "Shape::Alloc(Shape *pShape)" << '\n';
    #endif  // __DEBUG__

    try {
        m_Rank = pShape->GetRank();
        m_aDim = new int[m_Rank];

        for (int i = 0; i < m_Rank; i++) {
            m_aDim[i] = (*pShape)[i];
        }
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    m_Device = pShape->GetDevice();

#ifdef __CUDNN__
    m_idOfDevice = pShape->GetDeviceID();

    if (m_Device == GPU) SetDeviceGPU(m_idOfDevice);
#endif  // if __CUDNN__

    return TRUE;
}

/*!
@brief Shape 클래스에 할당된 메모리를 반환하는 메소드
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
void Shape::Delete() {
    #ifdef __DEBUG__
    std::cout << "Shape::Delete()" << '\n';
    #endif  // __DEBUG__

    if (m_aDim) {
        delete[] m_aDim;
        m_aDim = NULL;
    }

#ifdef __CUDNN__
    DeleteOnGPU();
#endif  // if __CUDNN__
}

#ifdef __CUDNN__
/*!
@brief
@details
@param idOfDevice
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int Shape::AllocOnGPU(unsigned int idOfDevice) {
    # if __DEBUG__
    std::cout << "Shape::AllocOnGPU()" << '\n';
    # endif // __DEBUG__

    m_idOfDevice = idOfDevice;
    checkCudaErrors(cudaSetDevice(idOfDevice));

    if (m_desc == NULL) {
        if (m_Rank == 5) {
            checkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
            checkCUDNN(cudnnSetTensor4dDescriptor(m_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                  m_aDim[1], m_aDim[2], m_aDim[3], m_aDim[4]));
        } else if (m_Rank == 4) {
            checkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
            checkCUDNN(cudnnSetTensor4dDescriptor(m_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                  m_aDim[0], m_aDim[1], m_aDim[2], m_aDim[3]));
        } else ;
    } else ;

    return TRUE;
}

/*!
@brief
@details
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
void Shape::DeleteOnGPU() {
    # if __DEBUG__
    std::cout << "Shape::DeleteOnGPU()" << '\n';
    # endif // __DEBUG__

    if (m_desc) {
        checkCUDNN(cudnnDestroyTensorDescriptor(m_desc));
        m_desc = NULL;
    }
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int Shape::ReShapeOnGPU() {
    # if __DEBUG__
    std::cout << "Shape::ReShapeOnGPU()" << '\n';
    # endif // __DEBUG__

    DeleteOnGPU();

    if (m_idOfDevice == -1) {
        std::cout << "you need to set device GPU first before : ReShapeOnGPU" << '\n';
        exit(-1);
    }
    AllocOnGPU(m_idOfDevice);

    return TRUE;
}

#endif  // if __CUDNN__

//////////////////////////////////////////////////////////////////////////////// for public method

/*!
@brief 5D-Shape 생성자
@details 5개 축의 Dimension을 매개변수로 받아 Shape를 할당하는 생성자
@param pSize0 첫 번째 축의 Dimension
@param pSize1 두 번째 축의 Dimension
@param pSize2 세 번째 축의 Dimension
@param pSize3 네 번째 축의 Dimension
@param pSize4 다섯 번째 축의 Dimension
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(5, pSize0, pSize1, pSize2, pSize3, pSize4);
}

/*!
@brief 4D-Shape 생성자
@details 4개 축의 Dimension을 매개변수로 받아 Shape를 할당하는 생성자
@param pSize0 첫 번째 축의 Dimension
@param pSize1 두 번째 축의 Dimension
@param pSize2 세 번째 축의 Dimension
@param pSize3 네 번째 축의 Dimension
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(4, pSize0, pSize1, pSize2, pSize3);
}

/*!
@brief 3D-Shape 생성자
@details 3개 축의 Dimension을 매개변수로 받아 Shape를 할당하는 생성자
@param pSize0 첫 번째 축의 Dimension
@param pSize1 두 번째 축의 Dimension
@param pSize2 세 번째 축의 Dimension
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
Shape::Shape(int pSize0, int pSize1, int pSize2) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(3, pSize0, pSize1, pSize2);
}

/*!
@brief 2D-Shape 생성자
@details 2개 축의 Dimension을 매개변수로 받아 Shape를 할당하는 생성자
@param pSize0 첫 번째 축의 Dimension
@param pSize1 두 번째 축의 Dimension
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
Shape::Shape(int pSize0, int pSize1) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(2, pSize0, pSize1);
}

/*!
@brief 1D-Shape 생성자
@details 1개 축의 Dimension을 매개변수로 받아 Shape를 할당하는 생성자
@param pSize0 첫 번째 축의 Dimension
@return 없음
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
Shape::Shape(int pSize0) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(1, pSize0);
}

/*!
@brief
@details
@param pShape
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
Shape::Shape(Shape *pShape) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(Shape *pShape)" << '\n';
    #endif  // __DEBUG__

    m_Rank = 0;
    m_aDim = NULL;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(pShape);
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
Shape::~Shape() {
    #ifdef __DEBUG__
    std::cout << "Shape::~Shape()" << '\n';
    #endif  // __DEBUG__

    Delete();
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int Shape::GetRank() {
    #ifdef __DEBUG__
    std::cout << "Shape::GetRank()" << '\n';
    #endif  // __DEBUG__

    return m_Rank;
}

/*!
@brief
@details
@param pRanknum
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int Shape::GetDim(int pRanknum) {
    #ifdef __DEBUG__
    std::cout << "Shape::GetDim(int pRanknum)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRanknum >= 0) return m_aDim[pRanknum];
        else throw;
    }
    catch (...) {
        printf("Receive invalid pRanknum value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        exit(0);
        // return FALSE;
    }
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
Device Shape::GetDevice() {
    #ifdef __DEBUG__
    std::cout << "Shape::GetDevice()" << '\n';
    #endif  // __DEBUG__

    return m_Device;
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int Shape::GetDeviceID() {
    return m_idOfDevice;
}

/*!
@brief
@details
@param pRnaknum
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int& Shape::operator[](int pRanknum) {
    #ifdef __DEBUG__
    std::cout << "Shape::operator[](int pRanknum)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRanknum >= 0) return m_aDim[pRanknum];
        else throw;
    }
    catch (...) {
        printf("Receive invalid pRanknum value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        exit(0);
        // return FALSE;
    }
}

/*!
@brief
@details
@param pSize0 Time의 Dimension
@param pSize1 Batch의 Dimension
@param pSize2 Channel의 Dimension
@param pSize3 Row의 Dimension
@param pSize4 Column의 Dimension
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int Shape::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    #ifdef __DEBUG__
    std::cout << "Shape::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4)" << '\n';
    #endif  // __DEBUG__

    return ReShape(5, pSize0, pSize1, pSize2, pSize3, pSize4);
}

/*!
@brief
@details
@param pRnak
@param ...
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int Shape::ReShape(int pRank, ...) {
    #ifdef __DEBUG__
    std::cout << "Shape::ReShape(int pRank, ...)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRank > 0) m_Rank = pRank;
        else throw pRank;
    } catch (int e) {
        printf("Receive invalid rank value %d in %s (%s %d)\n", e, __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    try {
        if (m_aDim) {
            delete[] m_aDim;
            m_aDim = NULL;
        }
        m_aDim = new int[m_Rank];
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    va_list ap;
    va_start(ap, pRank);

    for (int i = 0; i < pRank; i++) {
        m_aDim[i] = va_arg(ap, int);
    }

    va_end(ap);

#ifdef __CUDNN__

    if (m_Device == GPU) ReShapeOnGPU();
#endif  // if __CUDNN__

    return TRUE;
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int Shape::SetDeviceCPU() {
    #ifdef __DEBUG__
    std::cout << "Shape::SetDeviceCPU()" << '\n';
    #endif  // __DEBUG__

    m_Device = CPU;

    return TRUE;
}

#ifdef __CUDNN__

/*!
@brief
@details
@param idOfDevice
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
int Shape::SetDeviceGPU(unsigned int idOfDevice) {
    # if __DEBUG__
    std::cout << "Shape::SetDeviceGPU()" << '\n';
    # endif // __DEBUG__

    m_Device = GPU;

    if (m_desc == NULL) AllocOnGPU(idOfDevice);

    return TRUE;
}

/*!
@brief
@details
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
cudnnTensorDescriptor_t& Shape::GetDescriptor() {
    # if __DEBUG__
    std::cout << "Shape::GetDescriptor()" << '\n';

    if (m_Device == CPU) {
        printf("Warning! Tensor is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");

        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetDescriptor" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # else // if __DEBUG__

    if (m_Device == CPU) {
        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetDescriptor" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # endif // __DEBUG__

    return m_desc;
}

#endif  // __CUDNN__

/*!
@brief
@details
@param pOS
@param pShape
@return
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
std::ostream& operator<<(std::ostream& pOS, Shape *pShape) {
    #ifdef __DEBUG__
    std::cout << "std::ostream& operator<<(std::ostream& pOS, Shape *pShape)" << '\n';
    #endif  // __DEBUG__

    int rank = pShape->GetRank();

    pOS << "Rank is " << rank << ", Dimension is [";

    for (int i = 0; i < rank; i++) pOS << (*pShape)[i] << ", ";
    pOS << "]";
    return pOS;
}
>>>>>>> before discard

#ifndef __SHAPE__
#define __SHAPE__    value

#include "Common.h"

// #ifdef __CUDNN__
// typedef cudnnTensorDescriptor_t ShapeOnGPU;
// #endif  // if __CUDNN__

/*!
@class Shape Tensor의 차원 값을 저장하고 관리하는 클래스
@details 추가 예정
*/
// 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
class Shape {
private:
    int m_Rank;///< Shape 클래스를 구성하는 Rank 멤버변수, 텐서를 구성하는 축의 개수
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
    int *m_aDim;///< Shape 클래스를 구성하는 Dimension 멤버변수, 각 축의 차원
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
    Device m_Device;///< 장치 사용 구분자, CPU or GPU, Device 참고
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09
    int m_idOfDevice = -1;///< GPU 사용 시, 사용하려는 GPU의 번호
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-09-09

#ifdef __CUDNN__
    cudnnTensorDescriptor_t m_desc;
#endif  // if __CUDNN__

private:
    int  Alloc(int pRank, ...);
    int  Alloc(Shape *pShape);
    void Delete();

#ifdef __CUDNN__
    int  AllocOnGPU(unsigned int idOfDevice);
    void DeleteOnGPU();
    int  ReShapeOnGPU();
#endif  // if __CUDNN__

public:
    Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4);
    Shape(int pSize0, int pSize1, int pSize2, int pSize3);
    Shape(int pSize0, int pSize1, int pSize2);
    Shape(int pSize0, int pSize1);
    Shape(int pSize0);
    Shape(Shape *pShape);  // Copy Constructor
    virtual ~Shape();

    int                      GetRank();
    int                      GetDim(int pRanknum);
    int                    & operator[](int pRanknum); // operator[] overload
    Device                   GetDevice();
    int                      GetDeviceID();

    int                      ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4);
    int                      ReShape(int pRank, ...);


    int                      SetDeviceCPU();
#ifdef __CUDNN__
    int                      SetDeviceGPU(unsigned int idOfDevice);
    cudnnTensorDescriptor_t& GetDescriptor();
#endif  // __CUDNN__
};

std::ostream& operator<<(std::ostream& pOS, Shape *pShape);

#endif  // __SHAPE__

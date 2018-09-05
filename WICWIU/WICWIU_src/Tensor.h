#ifndef TENSOR_H_
#define TENSOR_H_

#include "Shape.h"
#include "LongArray.h"

/*!
@brief 시간 사용 유무
@details 시간 사용 유무를 이뉴머레이션으로 정의
         시간 사용 시 UseTime, 사용하지 않을 시 NoUseTime
*/
//author 윤동휘
//date 2018-09-03
enum IsUseTime {
    UseTime,
    NoUseTime
};

/*!
@class Tensor 다차원의 tensor데이터를 저장하고 관리하는 클래스
@brief 학습에 사용될 Tensor를 정의하기 위한 클래스
@details Tensor클래스는 Shape와 LongArray를 이용하여 Tensor의 모양과 데이터를 저장한다.
         Operator클래스에서   m_aaResult(ForwardPropagate한 값)와 m_aaGradient(BackPropagate한 값)을 저장한다.
*/
//author 윤동휘
//date 2018-09-03
template<typename DTYPE> class Tensor {
private:
    Shape *m_aShape; ///< Tensor를 구성하는 Shape 클래스, 텐서의 차원을 정의
    LongArray<DTYPE> *m_aLongArray; ///< Tensor를 구성하는 LongArray 클래스, 텐서의 원소들의 값을 저장
    Device m_Device; ///< 장치 사용 구분자, CPU or GPU, Device 참고
    int m_idOfDevice = -1; ///< GPU 사용 시, 사용하려는 GPU의 번호
    IsUseTime m_IsUseTime; ///< time 축 사용 유무, IsUseTime 참고

private:
    int  Alloc(Shape *pShape, IsUseTime pAnswer);
    int  Alloc(Tensor *pTensor);
    void Delete();

public:
    Tensor(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer = UseTime);  // For 5D-Tensor
    Tensor(int pSize0, int pSize1, int pSize2, int pSize3, IsUseTime pAnswer = UseTime);  // For 4D-Tensor
    Tensor(int pSize0, int pSize1, int pSize2, IsUseTime pAnswer = UseTime);  // For 3D-Tensor
    Tensor(int pSize0, int pSize1, IsUseTime pAnswer = UseTime);  // For 2D-Tensor
    Tensor(int pSize0, IsUseTime pAnswer = UseTime);  // For 1D-Tensor
    Tensor(Shape *pShape, IsUseTime pAnswer = UseTime);
    Tensor(Tensor<DTYPE> *pTensor);  // Copy Constructor

    virtual ~Tensor();

    Shape                  * GetShape();
    int                      GetRank();
    int                      GetDim(int pRanknum);
    LongArray<DTYPE>       * GetLongArray();
    int                      GetCapacity();
    int                      GetElement(unsigned int index);
    DTYPE                  & operator[](unsigned int index);
    Device                   GetDevice();
    IsUseTime                GetIsUseTime();
    DTYPE                  * GetCPULongArray(unsigned int pTime = 0);

    int                      GetTimeSize(); // 추후 LongArray의 Timesize 반환
    int                      GetBatchSize(); // 삭제 예정
    int                      GetChannelSize(); // 삭제 예정
    int                      GetRowSize(); // 삭제 예정
    int                      GetColSize(); // 삭제 예정


    int                      ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4);
    int                      ReShape(int pSize0, int pSize1, int pSize2, int pSize3);
    int                      ReShape(int pSize0, int pSize1, int pSize2);
    int                      ReShape(int pSize0, int pSize1);
    int                      ReShape(int pSize0);

    void                     Reset();


    void                     SetDeviceCPU();

    int                      Save(FILE *fileForSave);
    int                      Load(FILE *fileForLoad);
#ifdef __CUDNN__
    void                     SetDeviceGPU(unsigned int idOfDevice);

    DTYPE                  * GetGPUData(unsigned int pTime = 0);
    cudnnTensorDescriptor_t& GetDescriptor();

    void                     Reset(cudnnHandle_t& pCudnnHandle);


#endif  // if __CUDNN__


    static Tensor<DTYPE>* Random_normal(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, float mean, float stddev, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Random_normal(Shape *pShape, float mean, float stddev, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Zeros(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Zeros(Shape *pShape, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Constants(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, DTYPE constant, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Constants(Shape *pShape, DTYPE constant, IsUseTime pAnswer = UseTime);
};

/*!
@brief 5차원으로 정의된 Tensor의 LongArray에 접근하기 위한 인덱스를 계산
@details 5차원으로 정의된 Tensor에 대해, pShape와 각 축의 인덱스를 매개변수로 받아 이에 대응되는 LongArray의 원소에 접근하기 위한 인덱스 번호를 계산하여 반환한다.
@param pShape 인덱싱하려는 Tensor의 Shape
@param ti 인덱하려는 Tensor의 shape의 time 축의 인덱스 번호
@param ba 접근하려는 Tensor의 shape의 batch 축의 인덱스 번호
@param ch 접근하려는 Tensor의 shape의 channel 축의 인덱스 번호
@param ro 접근하려는 Tensor의 shape의 row 축의 인덱스 번호
@param co 접근하려는 Tensor의 shape의 column 축의 인덱스 번호
@return LongArray 원소에 접근하기 위한 인덱스 번호
*/
//author 윤동휘
//date 2018-09-03

inline unsigned int Index5D(Shape *pShape, int ti, int ba, int ch, int ro, int co) {
    return (((ti * (*pShape)[1] + ba) * (*pShape)[2] + ch) * (*pShape)[3] + ro) * (*pShape)[4] + co;
}

/*!
@brief 4차원으로 정의된 Tensor의 LongArray에 접근하기 위한 인덱스를 계산
@details 4차원으로 정의된 Tensor에 대해, pShape와 각 축의 인덱스를 매개변수로 받아 이에 대응되는 LongArray의 원소에 접근하기 위한 인덱스 번호를 계산하여 반환한다.
@param pShape 인덱싱하려는 Tensor의 Shape
@param ba 접근하려는 Tensor의 shape의 batch 축의 인덱스 번호
@param ch 접근하려는 Tensor의 shape의 channel 축의 인덱스 번호
@param ro 접근하려는 Tensor의 shape의 row 축의 인덱스 번호
@param co 접근하려는 Tensor의 shape의 column 축의 인덱스 번호
@return LongArray 원소에 접근하기 위한 인덱스 번호
*/
//author 윤동휘
//date 2018-09-03

inline unsigned int Index4D(Shape *pShape, int ba, int ch, int ro, int co) {
    return ((ba * (*pShape)[1] + ch) * (*pShape)[2] + ro) * (*pShape)[3] + co;
}

/*!
@brief 3차원으로 정의된 Tensor의 LongArray에 접근하기 위한 인덱스를 계산
@details 3차원으로 정의된 Tensor에 대해, pShape와 각 축의 인덱스를 매개변수로 받아 이에 대응되는 LongArray의 원소에 접근하기 위한 인덱스 번호를 계산하여 반환한다.
@param pShape 인덱싱하려는 Tensor의 Shape
@param ch 접근하려는 Tensor의 shape의 channel 축의 인덱스 번호
@param ro 접근하려는 Tensor의 shape의 row 축의 인덱스 번호
@param co 접근하려는 Tensor의 shape의 column 축의 인덱스 번호
@return LongArray 원소에 접근하기 위한 인덱스 번호
*/
//author 윤동휘
//date 2018-09-03

inline unsigned int Index3D(Shape *pShape, int ch, int ro, int co) {
    return (ch * (*pShape)[1] + ro) * (*pShape)[2] + co;
}

/*!
@brief 2차원으로 정의된 Tensor의 LongArray에 접근하기 위한 인덱스를 계산
@details 2차원으로 정의된 Tensor에 대해, pShape와 각 축의 인덱스를 매개변수로 받아 이에 대응되는 LongArray의 원소에 접근하기 위한 인덱스 번호를 계산하여 반환한다.
@param pShape 인덱싱하려는 Tensor의 Shape
@param ro 접근하려는 Tensor의 shape의 row 축의 인덱스 번호
@param co 접근하려는 Tensor의 shape의 column 축의 인덱스 번호
@return LongArray 원소에 접근하기 위한 인덱스 번호
*/
//author 윤동휘
//date 2018-09-03

inline unsigned int Index2D(Shape *pShape, int ro, int co) {
    return ro * (*pShape)[1] + co;
}

#endif  // TENSOR_H_

#ifndef __DATA__
#define __DATA__    value

#include "Common.h"

/*!
@class LongArray 다차원 Tensor의 데이터를 저장하고 관리하는 클래스.
@brief 학습에 사용 될 Tensor의 맴버변수 중 LongArray를 정의하기 위한 클래스.
@details 실질적으로 Tensor클래스의 데이터를 저장하고 관리하기위한 클래스.
@details 데이터를 초기화하고 CPU와 GPU간 데이터의 이동을 가능하게 한다.
*/

template<typename DTYPE> class LongArray {
private:
    DTYPE **m_aaHostLongArray; ///< 메모리에 올라가 있는 데이터의 주소 값.

    int m_CapacityOfLongArray; ///< LongArray의 총 용량.
    int m_TimeSize; ///< Tensor의 TimeSize
    int m_CapacityPerTime; ///< Time으로 나누어진 data의 Capacity

    Device m_Device; ///< 장치 사용 구분자 (CPU or GPU)
    int m_idOfDevice = -1; ///< GPU사용 시, 사용하려는 GPU의 번호.

#ifdef __CUDNN__
    DTYPE **m_aaDevLongArray; ///< GPU메모리에 올라가있는 데이터의 주소 값. m_aaHostLongArray와 비슷한 역할을 한다.
#endif  // __CUDNN

private:
    int  Alloc(unsigned int pTimeSize, unsigned int pCapacityPerTime);
    int  Alloc(LongArray *pLongArray);
    void Delete();

#ifdef __CUDNN__
    int  AllocOnGPU(unsigned int idOfDevice);
    void DeleteOnGPU();
    int  MemcpyCPU2GPU();
    int  MemcpyGPU2CPU();
#endif  // __CUDNN

public:
    LongArray(unsigned int pCapacity);
    LongArray(unsigned int pTimeSize, unsigned int pCapacityPerTime);
    LongArray(LongArray *pLongArray);  // Copy Constructor
    virtual ~LongArray();

    int    GetCapacity();
    int    GetTimeSize();
    int    GetCapacityPerTime();
    DTYPE  GetElement(unsigned int index);
    DTYPE& operator[](unsigned int index);
    Device GetDevice();
    int    GetDeviceID();
    DTYPE* GetCPULongArray(unsigned int pTime = 0);

    int    SetDeviceCPU();

    int    Save(FILE *fileForSave);
    int    Load(FILE *fileForLoad);
#ifdef __CUDNN__
    int    SetDeviceGPU(unsigned int idOfDevice);

    DTYPE* GetGPUData(unsigned int pTime = 0);

#endif  // if __CUDNN__
};


#endif  // __DATA__

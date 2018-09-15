#ifndef RESHAPE_H_
#define RESHAPE_H_    value

#include "../Operator.h"

/*!
@class
@details
@todo EXTRA
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE>
class ReShape : public Operator<DTYPE>{
private:
public:
    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    ReShape(Operator<DTYPE> *pInput, int pRowSize, int pColSize, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "ReShape::ReShape(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, 0, 0, 0, pRowSize, pColSize);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    ReShape(Operator<DTYPE> *pInput, int pChannelSize, int pRowSize, int pColSize, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "ReShape::ReShape(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, 0, 0, pChannelSize, pRowSize, pColSize);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    ReShape(Operator<DTYPE> *pInput, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "ReShape::ReShape(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, 0, pBatchSize, pChannelSize, pRowSize, pColSize);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    ReShape(Operator<DTYPE> *pInput, int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "ReShape::ReShape(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    ~ReShape() {
        #ifdef __DEBUG__
        std::cout << "ReShape::~ReShape()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int Alloc(Operator<DTYPE> *pInput, int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
        #ifdef __DEBUG__
        std::cout << "ReShape::Alloc(Operator *, Operator *)" << '\n';
        #endif  // __DEBUG__

        Shape *pInputShape = pInput->GetResult()->GetShape();

        if (pTimeSize == 0) pTimeSize = (*pInputShape)[0];

        if (pBatchSize == 0) pBatchSize = (*pInputShape)[1];

        if (pChannelSize == 0) pChannelSize = (*pInputShape)[2];


        Tensor<DTYPE> *result = new Tensor<DTYPE>(pInput->GetResult());
        result->ReShape(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

        this->SetResult(result);  // copy data

        this->SetDelta(new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize));

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
    void Delete() {}

    /*!
    @brief
    @details
    @param
    @return
    @todo E_Train
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int  ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                            = (*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)];
                    }
                }
            }
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
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();

        Shape *deltaTenShape = this_delta->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        (*input_delta)[Index5D(deltaTenShape, ti, ba, ch, ro, co)]
                            += (*this_delta)[Index5D(deltaTenShape, ti, ba, ch, ro, co)];
                    }
                }
            }
        }

        return TRUE;
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
    int ForwardPropagateOnGPU(int pTime) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        DTYPE *pDevInput  = input->GetGPUData();
        DTYPE *pDevResult = result->GetGPUData();

        cudnnTensorDescriptor_t pDesc = input->GetDescriptor();

        float alpha = 1.f;
        float beta  = 0.f;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &alpha, pDesc, pDevInput,
                                  &alpha, pDesc, pDevResult));

        // this->ForwardPropagate(pTime);
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
    int BackPropagateOnGPU(int pTime) {
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        DTYPE *pDevDelta      = this_delta->GetGPUData();
        DTYPE *pDevInputDelta = input_delta->GetGPUData();

        cudnnTensorDescriptor_t pDesc = this_delta->GetDescriptor();

        float alpha = 1.f;
        float beta  = 0.f;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &alpha, pDesc, pDevDelta,
                                  &alpha, pDesc, pDevInputDelta));

        // this->BackPropagate(pTime);

        return TRUE;
    }

#endif  // __CUDNN__
};

#endif  // RESHAPE_H_

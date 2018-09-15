#ifndef MSE_H_
#define MSE_H_    value

#include "../LossFunction.h"

/*!
@class
@details
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE>
class MSE : public LossFunction<DTYPE>{
public:
    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    MSE(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    virtual ~MSE() {
        #ifdef __DEBUG__
        std::cout << "MSE::~MSE()" << '\n';
        #endif  // __DEBUG__
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    virtual int Alloc(Operator<DTYPE> *pOperator) {
        #ifdef __DEBUG__
        std::cout << "MSE::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(
            new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1)
            );

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
    Tensor<DTYPE>* ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetTensor();
        Tensor<DTYPE> *label  = this->GetLabel()->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = ti * batchsize + ba;

            for (int j = 0, index = 0; j < capacity; j++) {
                index         = i * capacity + j;
                (*result)[i] += Error((*input)[index], (*label)[index]);
            }
        }

        return result;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo E_Train
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE>* BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input       = this->GetTensor();
        Tensor<DTYPE> *label       = this->GetLabel()->GetResult();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchsize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = ti * batchsize + ba;

            for (int j = 0, index = 0; j < capacity; j++) {
                index                  = i * capacity + j;
                (*input_delta)[index] += ((*input)[index] - (*label)[index]) / batchsize;
            }
        }

        return NULL;
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
    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0) {
        this->ForwardPropagate();
        return NULL;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo GPU
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0) {
        this->BackPropagate();
        return NULL;
    }

#endif  // __CUDNN__


    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    inline DTYPE Error(DTYPE pred, DTYPE ans) {
        return (pred - ans) * (pred - ans) / 2;
    }
};

#endif  // MSE_H_

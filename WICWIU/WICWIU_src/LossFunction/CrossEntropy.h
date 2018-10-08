#ifndef CROSSENTROPY_H_
#define CROSSENTROPY_H_    value

#include "../LossFunction.h"

/*!
@class
@details
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE>
class CrossEntropy : public LossFunction<DTYPE>{
private:
    DTYPE m_epsilon = 0.0;  // for backprop ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

public:
    /*!
    @brief CrossEntropy LossFunction 클래스 생성자
    @details LossFunction 클래스의 생성자를 호출하고, Operator와 epsilon을 매개변수로 전달하여 CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드를 호출한다.
    @param pOperator CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드의 매개변수로 전달할 Operator
    @param pLabel LossFunction의 입력 레이블에 해당하는 Operator
    @param epsilon 값을 지정하지 않을 시 1e-6f로 초기화
    @return 없음
    @see CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon)
    */
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-08
    CrossEntropy(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, int epsilon = 1e-6f) : LossFunction<DTYPE>(pOperator, pLabel) {
        #ifdef __DEBUG__
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator, epsilon);
    }

    /*!
    @brief CrossEntropy LossFunction 클래스 생성자
    @details LossFunction 클래스의 생성자를 호출하고, Operator와 1e-6f에 해당하는 epsilon 값을 매개변수로 전달하여 CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드를 호출한다.
    @param pOperator CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드의 매개변수로 전달할 Operator
    @param pLabel LossFunction의 입력 레이블에 해당하는 Operator
    @param pName LossFunction의 이름
    @return 없음
    @see CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon)
    */
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-08
    CrossEntropy(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator, 1e-6f);
    }

    /*!
    @brief CrossEntropy LossFunction 클래스 생성자
    @details LossFunction 클래스의 생성자를 호출하고, Operator와 epsilon을 매개변수로 전달하여 CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드를 호출한다.
    @param pOperator CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드의 매개변수로 전달할 Operator
    @param pLabel LossFunction의 입력 레이블에 해당하는 Operator
    @param epsilon
    @param pName LossFunction의 이름
    @return 없음
    @see CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon)
    */
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-08
    CrossEntropy(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, int epsilon, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator, epsilon);
    }

    /*!
    @brief CrossEntropy LossFunction 클래스 소멸자
    @return 없음
    */
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-08
    ~CrossEntropy() {
        #ifdef __DEBUG__
        std::cout << "CrossEntropy::~CrossEntropy()" << '\n';
        #endif  // __DEBUG__
    }

    /*!
    @brief CrossEntropy Lossfunction의 멤버 변수들을 동적 할당하는 메소드
    @details 매개변수로 전달받은 Operator를 Input Operator에 할당하고 초기화 된 Result 텐서를 동적으로 할당 및 생성한다.
    @param pOperator CrossEntropy LossFunction의 입력에 해당하는 Operator
    @param epsilon 더미 변수
    @return TRUE
    */
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-08
    virtual int Alloc(Operator<DTYPE> *pOperator, int epsilon) {
        #ifdef __DEBUG__
        std::cout << "CrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        return TRUE;
    }

    /*!
    @brief
    @details
    @param pTime
    @return
    */
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-08
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
            i = (ti * batchsize + ba);

            for (int j = 0, index = 0; j < capacity; j++) {
                index         = i * capacity + j;
                (*result)[i] += -(*label)[index] * log((*input)[index] + m_epsilon);
            }
        }

        return result;
    }

    /*!
    @brief
    @details
    @param pTime
    @return
    */
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-08
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
                (*input_delta)[index] += -(*label)[index] / (*input)[index] / batchsize;
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
};

#endif  // CROSSENTROPY_H_

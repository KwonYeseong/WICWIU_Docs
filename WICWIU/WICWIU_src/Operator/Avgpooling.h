#ifndef __AVGPOOLING__
#define __AVGPOOLING__    value

#include "../Operator.h"

/*!
@class GlobalAvaragePooling2D GlobalAvaragePooling2D class
@details Row * Colunm 공간을 GlobalAvaragePooling하는 클래스.
*/
// 문서 작성자 : 권예성, 작성 날짜 : 2018-9-23
template<typename DTYPE> class GlobalAvaragePooling2D : public Operator<DTYPE>{
private:
    int m_timesize; ///< timetime
    int m_batchsize; ///< batchbatch
    int m_channelsize; ///< channelchannel
    int m_rowsize; ///< rowrow
    int m_colsize; ///< colcol

    int m_divisor; ///< Average를 결정 짓는 값. ex) row_size * col_size
    // 문서 작성자 : 권예성, 작성 날짜 : 2018-9-23

public:
    /*!
    @brief GlobalAvaragePooling2D의 생성자.
    @details 파라미터로 받은 pInput으로 Alloc한다.
    @param pInput GlobalAvaragePooling2D할 대상 Operator.
    @param pName 사용자가 부여한 Operator이름.
    @ref int Alloc(Operator<DTYPE> *pInput).
    */
    // 문서 작성자 : 권예성, 작성 날짜 : 2018-9-23
    GlobalAvaragePooling2D(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "GlobalAvaragePooling2D::GlobalAvaragePooling2D(Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        Alloc(pInput);
    }

    /*!
    @brief GlobalAvaragePooling2D의 소멸자.
    */
    virtual ~GlobalAvaragePooling2D() {}

    /*!
    @brief 파라미터로 받은 pInput으로부터 맴버 변수들을 초기화 한다.
    @details Result와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
    @param pInput 생성 할 Tensor의 Shape정보를 가진 Operator
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE> *pInput) {
        Shape *pInputTenShape = pInput->GetResult()->GetShape();

        m_timesize    = (*pInputTenShape)[0];
        m_batchsize   = (*pInputTenShape)[1];
        m_channelsize = (*pInputTenShape)[2];
        m_rowsize     = (*pInputTenShape)[3];
        m_colsize     = (*pInputTenShape)[4];

        m_divisor = m_rowsize * m_colsize;

        this->AddResult(new Tensor<DTYPE>(new Shape(m_timesize, m_batchsize, m_channelsize, 1, 1)));
        this->AddGradient(new Tensor<DTYPE>(new Shape(m_timesize, m_batchsize, m_channelsize, 1, 1)));

        return TRUE;
    }

    /*!
    @brief GlobalAvaragePooling2D의 ForwardPropagate 매소드
    @details input의 row, col상의 값들들 모두 더하고 m_divisor로 나눈 값을 result Tensor에 저장한다.
    @param pTime pInput의 m_timesize값, default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < m_batchsize; ba++) {
            for (int ch = 0; ch < m_channelsize; ch++) {
                for (int ro = 0; ro < m_rowsize; ro++) {
                    for (int co = 0; co < m_colsize; co++) {
                        (*result)[Index5D(resultTenShape, ti, ba, ch, 0, 0)]
                            += (*input)[Index5D(inputTenShape, ti, ba, ch, ro, co)];
                    }
                }
                (*result)[Index5D(resultTenShape, ti, ba, ch, 0, 0)] /= m_divisor;
            }
        }

        return TRUE;
    }

    /*!
    @brief GlobalAvaragePooling2D의 BackPropagate 매소드
    @details Input_grad에 계산한 Gradient / m_divisor 한 값을 더한다.
    @param pTime pInput의 m_timesize값, default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner         = this->GetInputContainer();
        Container<Tensor<DTYPE> *>   *input_gradient_container = (*input_contatiner)[0]->GetGradientContainer();
        Container<Tensor<DTYPE> *>   *this_gradient_container  = this->GetGradientContainer();

        Tensor<DTYPE> *this_grad  = (*this_gradient_container)[0];
        Tensor<DTYPE> *input_grad = (*input_gradient_container)[0];

        Shape *resultTenShape = this_grad->GetShape();
        Shape *inputTenShape  = input_grad->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < m_batchsize; ba++) {
            for (int ch = 0; ch < m_channelsize; ch++) {
                for (int ro = 0; ro < m_rowsize; ro++) {
                    for (int co = 0; co < m_colsize; co++) {
                        (*input_grad)[Index5D(inputTenShape, ti, ba, ch, ro, co)]
                            += (*this_grad)[Index5D(resultTenShape, ti, ba, ch, 0, 0)] / m_divisor;
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
        this->ForwardPropagate(pTime);
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
        this->BackPropagate(pTime);

        return TRUE;
    }

#endif  // __CUDNN__
};
//
#endif  // __AVGPOOLING__

#ifndef LRELU_H_
#define LRELU_H_    value

#include "../Operator.h"

/*!
@class LRelu 클래스
*/
template<typename DTYPE> class LRelu : public Operator<DTYPE>{
private:
      float m_negativeSlope; ///< 입력값이 음수일 경우 연산에 사용하는 변수

public:
    /*!
    @brief LRelu의 생성자.
    @details 파라미터로 받은 pInput, negativeSlope으로 Alloc한다.
    @param pInput Alloc할 대상 Operator
    @param negativeSlope 입력값이 음수일 경우 사용하는 기울기
    @ref int Alloc(Operator<DTYPE> *pInput, float negativeSlope)
    */
    LRelu(Operator<DTYPE> *pInput, float negativeSlope) : Operator<DTYPE>(pInput) {
        #ifdef __DEBUG__
        std::cout << "LRelu::LRelu(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, negativeSlope);
    }

    /*!
    @brief LRelu의 생성자.
    @details 파라미터로 받은 pInput, negativeSlope으로 Alloc한다.
    @param pInput Alloc할 대상 Operator
    @param negativeSlope 입력값이 음수일 경우 사용하는 기울기
    @param pName Operator에 사용자가 부여한 이름.
    @ref int Alloc(Operator<DTYPE> *pInput, float negativeSlope)
    */
    LRelu(Operator<DTYPE> *pInput, float negativeSlope, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "LRelu::LRelu(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, negativeSlope);
    }

    /*!
    @brief LRelu의 소멸자.
    @see void Delete()
    */
    ~LRelu() {
        #ifdef __DEBUG__
        std::cout << "LRelu::~LRelu()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    /*!
    @brief 파라미터로 받은 pinput으로부터 맴버 변수들을 초기화 한다.
    @details Result와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
    @details  negativeSlope은 m_negativeSlope에 저장한다.
    @param pInput 생성할 Tensor의 Shape정보를 가진 Operator
    @param negativeSlope 입력값이 음수일 경우 사용하는 기울기
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE> *pInput, float negativeSlope) {
        #ifdef __DEBUG__
        std::cout << "LRelu::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_negativeSlope = negativeSlope;
        //std::cout << m_negativeSlope << '\n';

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }
//
// #ifdef __CUDNN__
//     void InitializeAttributeForGPU(unsigned int idOfDevice) {
//
//     }
//
// #endif  // if __CUDNN__

    /*
    @brief 메모리를 헤제하는 Delete 메소드
    */
    void Delete() {

    }

    /*!
    @brief LRelu의 ForwardPropagate 매소드.
    @details input의 Tensor값들 중 0.f이상의 값은 그대로 result에 저장하고,
    @details 0.f미만의 값은 m_negativeSlope을 곱한 후 저장한다.
    @param pTime pInput의 m_timesize값, default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        int ti = pTime;


		    /* 	improved version 1
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        int idx = Index5D(resultTenShape, ti, ba, ch, ro, co);
                        if((*input)[idx] >= 0.f)
            							(*result)[idx] = (*input)[idx];
            						else
            							(*result)[idx] = m_negativeSlope * (*input)[idx];
                    }
                }
            }
        }
		    */

		    /*  improved version 2
		    int idx = Index5D(resultTenShape, 0, 0, 0, 0, 0);
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++, idx++) {
            						if((*input)[idx] >= 0.f)
            							(*result)[idx] = (*input)[idx];
            						else
            							(*result)[idx] = m_negativeSlope * (*input)[idx];
                    }
                }
            }
        }
		    */

		    //  improved version 3
		    int index = 0;
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        index = Index5D(resultTenShape, ti, ba, ch, ro, co);
            						if((*input)[index] >= 0.f)
            							(*result)[index] = (*input)[index];
            						else
            							(*result)[index] = m_negativeSlope * (*input)[index];
                    }
                }
            }
        }


		    /*  improved version 4
    		int idx = Index5D(resultTenShape, 0, 0, 0, 0, 0);
    		DTYPE* ip = &(*input)[idx];
    		DTYPE* op = &(*result)[idx];
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++, ip++, op++) {
            						if(*ip >= 0.f)
            							*op = *ip;
            						else
            							*op = m_negativeSlope * *ip;
                    }
                }
            }
        }
		    */

    		/* improved version 5
    		int index = Index5D(resultTenShape, 0, 0, 0, 0, 0); //initial index
    		DTYPE *input_ptr = &(*input)[index];
    		DTYPE *result_ptr = &(*result)[index];
    		DTYPE *input_limit = input_ptr + batchsize * channelsize * rowsize * colsize;
    		for(; input_ptr < input_limit; input_ptr++, result_ptr++){

      			if(*input_ptr >= 0.f)
      				*result_ptr = *input_ptr;
      			else
      				*result_ptr = m_negativeSlope * (*input_ptr);
    		}*/

        return TRUE;
    }

    /*!
    @brief LRelu의 BackPropagate 매소드.
    @details result값이 0보다 클 경우 input_delta에 더하고,
    @details 0보다 작을 경우 input_delta에 m_negativeSlope을 곱한 후 더한다.
    @param pTime pInput의 m_timesize값, default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        int ti = pTime;
        int index = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {

                        index = Index5D(resultTenShape, ti, ba, ch, ro, co);

                        if ((*result)[index] > 0.f)
                            (*input_delta)[index] += (*this_delta)[index];
                        else
                            (*input_delta)[index] += m_negativeSlope * (*this_delta)[index];
                    }
                }
            }
        }

        /*improved version
        int index = Index5D(resultTenShape, 0, 0, 0, 0, 0); //initial index
        DTYPE *input_delta_ptr = &(*input_delta)[index];
        DTYPE *this_delta_ptr = &(*this_delta)[index];
        DTYPE *result_ptr = &(*result)[index];
        DTYPE *delta_limit = input_delta_ptr + batchsize * channelsize * rowsize * colsize;
        for(; input_delta_ptr < delta_limit; input_delta_ptr++, this_delta_ptr++, result_ptr++){

            if(*result_ptr > 0.f)
              *input_delta_ptr += *this_delta_ptr;
            else
              *input_delta_ptr += m_negativeSlope * (*this_delta_ptr);
        }
        */

        return TRUE;
    }


#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0);

    int BackPropagateOnGPU(int pTime = 0);

#endif  // if __CUDNN__
};

#endif  // LRELU_H_

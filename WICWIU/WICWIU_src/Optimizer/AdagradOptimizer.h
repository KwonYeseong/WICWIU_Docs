#ifndef ADAGRADOPTIMIZER_H_
#define ADAGRADOPTIMIZER_H_   value

#include "../Optimizer.h"

/*!
@class Adagrad Optimizer
@details
*/
// 문서 작성자 : 윤성결 , 작성 날짜 : 2018-11-19

template<typename DTYPE> class AdagradOptimizer : public Optimizer<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_ppParameter; ///< 값을 업데이트 할 Tensor들을 가리키는 포인터
    Container<Tensor<DTYPE> *> *m_aaGradientSquared; ///<  @todo Variable

    int m_numOfParameter; ///<  업데이트 할 Tensor의 degree
    float m_epsilon;  ///< 분모 값이 0이 되는 것을 방지 하는 값


public:
    /*!
    @brief AdagradOptimizer 생성자.
    @details 맴버변수들을 초기화하고 Alloc 매소드를 호출한다.
    @param *pParameterContainer 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pLearningRate Optimizer의 learning rate
    @param epsilon 분모 값이 0이 되는 것을 방지 하는 값
    @param pOptimizeDirection Optimizing의 방향(MAXIMIZE or MINIMIZE)
    @return 없음
    @see int Alloc(epsilon)
    */

    AdagradOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float epsilon, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
    #ifdef __DEBUG__
    std::cout << "AdagradOptimizer::AdagradOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
    #endif  // __DEBUG__
    m_ppParameter            = NULL;
    m_aaGradientSquared      = NULL;

    m_numOfParameter         = 0;
    m_epsilon                = 0.f;

      Alloc(epsilon);
  }

    /*!
    @brief AdagradOptimizer 생성자.
    @details 맴버변수들을 초기화하고 Alloc 매소드를 호출한다.
    @param *pParameterContainer 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pLearningRate Optimizer의 learning rate
    @param epsilon 분모 값이 0이 되는 것을 방지 하는 값
    @param weightDecayRate 가중치 매개변수가 클 때 패널티를 부과하는 값
    @param pOptimizeDirection Optimizing의 방향(MAXIMIZE or MINIMIZE)
    @return 없음
    @see int Alloc(epsilon)
    */

    AdagradOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float epsilon, float weightDecayRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, weightDecayRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "AdagradOptimizer::AdagradOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter          = NULL;
        m_aaGradientSquared    = NULL;

        m_numOfParameter       = 0;
        m_epsilon              = 0.f;

        Alloc(epsilon);
    }

    /*!
    @brief AdagradOptimizer 소멸자
    @return 없음
    */

    ~AdagradOptimizer() {
        #ifdef __DEBUG__
        std::cout << "AdagradOptimizer::~AdagradOptimizer()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    /*!
    @brief Optimizer의 Delete 매소드
    @details 맴버 변수 m_aaGradientSquared 메모리 할당을 해제한다.
    @return 없음
    */

    virtual void Delete(){
      if (m_aaGradientSquared) {
          delete m_aaGradientSquared;
          m_aaGradientSquared = NULL;
      }
    }

    /*!
    @brief Optimizer의 Alloc 매소드
    @details 맴버 변수 m_ppParameter, m_numOfParameter, m_ppParameter, m_aaGradientSquared 초기화한다.
    @details m_aaGradientSquared m_ppParameter와 같은 Shape의 Tensor를 생성하여 넣는다.
    @param epsilon 분모 값이 0이 되는 것을 방지 하는 값
    @return 성공 시 TRUE
    @see Container<Operator<DTYPE> *>* GetTrainableTensor()
    @see int GetTrainableTensorDegree()
    */

    int Alloc(float epsilon){
      m_ppParameter    = this->GetTrainableTensor();
      m_numOfParameter = this->GetTrainableTensorDegree();
      m_aaGradientSquared = new Container<Tensor<DTYPE> *>();

      Shape *pParameterGradShape = NULL;

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();
            m_aaGradientSquared->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            pParameterGradShape = NULL;
        }

        m_epsilon = epsilon;

        return TRUE;
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 메소드
    @details m_epsilon 유무에 따라 UpdateParameter 호출과 에러 메세지 호출
    @return 성공 시 TRUE
    @see int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pGradientSquared)
    */

    virtual int UpdateParameter() {
      if(m_epsilon != 0.f){
          for (int i = 0; i < m_numOfParameter; i++) {
              UpdateParameter((*m_ppParameter)[i], (*m_aaGradientSquared)[i]);
          }
      }else{std::cout << "Don't execute UpdateParameter"<<std::endl;}
      return TRUE;
    }

    /*!
    @brief UpdateParameter default 함수
    @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @return 성공 시 TRUE
    */

    int UpdateParameter(Operator<DTYPE> *pParameter){
      return TRUE;
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 메소드
    @details gradient 제곱 값으로 pGradientSquared 업데이트
    @details signed_learning_rate와 gradient의 곱을 업데이트 된 pGradientSquared값에 root를 적용 한 값으로 나누어 파라미터를 업데이트 한다.
    @param pGradientSquared 업데이트 할 pGradientSquared

    @return 성공 시 TURE
    */


    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pGradientSquared) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*pGradientSquared)[i] = ((*gradient)[i] * (*gradient)[i]);
            (*trainable_data)[i]    += (signed_learning_rate * (*gradient)[i]) / std::sqrt((*pGradientSquared)[i] + m_epsilon);
        }

        return TRUE;
    }


#ifdef __CUDNN__

    /*!
    @brief m_aaGradientSquared Tensor의 device를 idOfDevice번째 GPU로 바꾼다. 실패시 에러 메세지 호출
    @param idOfDevice 사용 하는 GPU번호
    */
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        if(m_epsilon != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                (*m_aaGradientSquared)[i]->SetDeviceGPU(idOfDevice);
            }
        }else{std::cout << "Don't execute SetDeviceGPU"<< std::endl;}
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 메소드(GPU)
    @details m_epsilon 유무에 따라 UpdateParameterOnGPU 호출과 에러 메세지 호출
    @return 성공 시 TRUE
    @see int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pGradientSquared)
    */

    virtual int UpdateParameterOnGPU() {
        if(m_epsilon != 0.f){
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaGradientSquared)[i]);
            }
        }else{std::cout << "Don't execute UpdateParameterOnGPU"<<std::endl;}
        return TRUE;
    }

    /*!
    @brief UpdateParameterOnGPU default 함수
    @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @return 성공 시 TRUE
    */

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter){
      return TRUE;
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 GPU 메서드.
    @details 실행시 해당 cu 파일의 커널함수 실행을 위한 생성자 호출
    @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pGradientSquared 업데이트 할 pGradientSquared
    @see int AdagradOptimizer<DTYPE>::UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pGradientSquared)
    */

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pGradientSquared);


 #endif  // if __CUDNN__
};

#endif  // ADAGRADOPTIMIZER_H_

#ifndef SEQUENTIAL_H_
#define SEQUENTIAL_H_    value

#include "../Operator.h"

/*!
@class Sequential class
*/
// 문서 작성자 : 권예성, 작성 날짜 : 2018-9-24
template<typename DTYPE>
class Sequential : public Operator<DTYPE>{
    Operator<DTYPE> **m_listOfOperator; ///<   @todo Variable
    int m_numOfOperator; ///<   @todo Variable

public:
    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    Sequential(int numOfOperator, ...) {
        std::cout << "Sequential::Sequential(Operator<DTYPE> *)" << '\n';

        m_listOfOperator = NULL;
        m_numOfOperator  = 0;

        va_list ap;
        va_start(ap, numOfOperator);

        Alloc(numOfOperator, &ap);

        va_end(ap);
    }

    /*!
    @brief Sequential의 소멸자.
    */
    ~Sequential() {
        std::cout << "Sequential::~Sequential()" << '\n';
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int Alloc(int numOfOperator, va_list *ap) {
        std::cout << "Sequential::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        m_listOfOperator = new Operator<DTYPE> *[numOfOperator];
        m_numOfOperator  = numOfOperator;

        for (int i = 0; i < numOfOperator; i++) {
            m_listOfOperator[i] = va_arg(*ap, Operator<DTYPE> *);
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
    int ForwardPropagate() {
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
    int BackPropagate() {
        return TRUE;
    }
};

#endif  // SEQUENTIAL_H_

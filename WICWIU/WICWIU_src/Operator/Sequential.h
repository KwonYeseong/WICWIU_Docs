#ifndef SEQUENTIAL_H_
#define SEQUENTIAL_H_    value

#include "../Operator.h"

/*!
@class
@details
@todo EXTRA
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE>
class Sequential : public Operator<DTYPE>{
    Operator<DTYPE> **m_listOfOperator; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_numOfOperator; ///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-

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
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
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

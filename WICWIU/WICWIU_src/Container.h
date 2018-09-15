#include "Common.h"

template<typename DTYPE> class Tensor;
template<typename DTYPE> class Operator;
template<typename DTYPE> class Tensorholder;

/*!
@class
@details
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> class Container {
private:
    DTYPE *m_aElement;///<   @todo Variable
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_size;///<   @todo Variable
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
    Container() {
        #ifdef __DEBUG__
        std::cout << "Container<DTYPE>::Container()" << '\n';
        #endif  // __DEBUG__
        m_aElement = NULL;
        m_size     = 0;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo Constructor
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    virtual ~Container() {
        if (m_aElement) {
            delete[] m_aElement;
            m_aElement = NULL;
        }
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선 순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int Push(DTYPE pElement) {
        try {
            DTYPE *temp = new DTYPE[m_size + 1];

            for (int i = 0; i < m_size; i++) temp[i] = m_aElement[i];
            temp[m_size] = pElement;

            if (m_aElement) {
                delete[] m_aElement;
                m_aElement = NULL;
            }

            m_aElement = temp;
        } catch (...) {
            printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        m_size++;

        return TRUE;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선 순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    DTYPE Pop() {
        DTYPE  element = m_aElement[0];
        DTYPE *temp    = new DTYPE[m_size - 1];

        for (int i = 1; i < m_size; i++) temp[i - 1] = m_aElement[i];

        if (m_aElement) {
            delete[] m_aElement;
            m_aElement = NULL;
        }

        m_aElement = temp;

        m_size--;

        return element;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선 순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    DTYPE Pop(DTYPE pElement) {
        int index = -1;

        for (int i = 0; i < m_size; i++) {
            if (m_aElement[i] == pElement) index = i;
        }

        if (index == -1) {
            std::cout << "There is no element!" << '\n';
            return NULL;
        }

        // DTYPE  element = m_aElement[index];
        DTYPE *temp = new DTYPE[m_size - 1];

        for (int i = 0, j = 0; i < m_size - 1;) {
            if (index != j) {
                temp[i] = m_aElement[j];
                i++;
            }

            j++;
        }

        if (m_aElement) {
            delete[] m_aElement;
            m_aElement = NULL;
        }

        m_aElement = temp;

        m_size--;

        return pElement;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선 순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int Reverse() {
        DTYPE *temp = new DTYPE[m_size];

        for (int i = 0; i < m_size; i++) temp[m_size - i - 1] = m_aElement[i];

        if (m_aElement) {
            delete[] m_aElement;
            m_aElement = NULL;
        }

        m_aElement = temp;

        return TRUE;
    }

    int SetElement(DTYPE pElement, unsigned int index) {
        m_aElement[index] = pElement;
        return TRUE;
    }

    int GetSize() {
        // std::cout << "Container<DTYPE>::GetSize()" << '\n';
        return m_size;
    }

    DTYPE GetLast() {
        // std::cout << "Container<DTYPE>::GetLast()" << '\n';
        return m_aElement[m_size - 1];
    }

    DTYPE* GetRawData() const {
        return m_aElement;
    }

    DTYPE GetElement(unsigned int index) {
        return m_aElement[index];
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선 순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    DTYPE& operator[](unsigned int index) {
        return m_aElement[index];
    }
};

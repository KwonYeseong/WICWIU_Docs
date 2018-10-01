#ifndef __LINEAR_LAYER__
#define __LINEAR_LAYER__    value

#include "../Module.h"

/*!
@class
@details
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> class Linear : public Module<DTYPE>{
private:
public:
    /*!
    @brief Linear 클래스 생성자
    @details Linear 클래스의 Alloc 메소드를 호출한다.
    @see linear<DTYPE>::Alloc(Operator<DTYPE> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias, std::string pName)
    */
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
    Linear(Operator<DTYPE> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias = FALSE, std::string pName = NULL) : Module<DTYPE>(pName) {
        Alloc(pInput, pNumInputCol, pNumOutputCol, use_bias, pName);
    }

    /*!
    @brief Linear 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.
    */
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
    virtual ~Linear() {}

    /*!
    @brief Linear(Fully Connected) Layer 그래프를 동적으로 할당 및 구성하는 메소드
    @details Input Operator의 Element에 대해 Weight를 이용해 행렬 곱(Matrix Multiplication)을 수행하고 Bias가 존재할 시 Bias를 합(Column Wise Addition)해 Output Operator로 내보내는 layer를 구성한다.
    @param pInput 해당 Layer의 Input에 해당하는 Operator
    @param pNumInputCol 해당 Layer의 Input Operator의 Column의 갯수, Input Column에 대한 Dimension
    @param pNumOutputCol 해당 Layer의 Output Operator의 Column의 갯수, Output Column에 대한 Dimension
    @param use_bias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
    @param pName Module의 이름
    @return TRUE
    @see MatMul<DTYPE>::MatMul(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput, std::string pName) AddColWise<DTYPE>::AddColWise(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias, std::string pName) Module<DTYPE>::AnalyzeGraph(Operator<DTYPE> *pResultOperator)
    */
    // 문서 작성자 : 윤동휘, 작성 날짜 : 2018-10-01
    int Alloc(Operator<DTYPE> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        Tensorholder<DTYPE> *pWeight = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, pNumOutputCol, pNumInputCol, 0.0, 0.1), "Layer_Weight_" + pName);
        out = new MatMul<DTYPE>(pWeight, out, "Layer_MatMul_" + pName);

        if (use_bias) {
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumOutputCol, 0.f), "Add_Bias_" + pName);
            out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif  // __LINEAR_LAYER__

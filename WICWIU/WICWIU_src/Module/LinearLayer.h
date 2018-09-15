#ifndef __LINEAR_LAYER__
#define __LINEAR_LAYER__    value

#include "../Module.h"

/*!
@class
@details
@todo 우선순위
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> class Linear : public Module<DTYPE>{
private:
public:
    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    Linear(Operator<DTYPE> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias = FALSE, std::string pName = NULL) : Module<DTYPE>(pName) {
        Alloc(pInput, pNumInputCol, pNumOutputCol, use_bias, pName);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    virtual ~Linear() {}

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
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

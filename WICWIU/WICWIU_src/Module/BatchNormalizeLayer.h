#ifndef __BATCH_NORMALIZE_LAYER__
#define __BATCH_NORMALIZE_LAYER__    value

#include "../Module.h"

/*!
@class
@details
@todo 우선순위
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE> class BatchNormalizeLayer : public Module<DTYPE>{
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
    BatchNormalizeLayer(Operator<DTYPE> *pInput, int pIsChannelwise = FALSE, std::string pName = "NO NAME") {
        Alloc(pInput, pIsChannelwise, pName);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    virtual ~BatchNormalizeLayer() {}

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int Alloc(Operator<DTYPE> *pInput, int pIsChannelwise, std::string pName) {
        this->SetInput(pInput);
        Operator<DTYPE> *out = pInput;
        Shape *pInputShape   = out->GetResult()->GetShape();

        Tensorholder<DTYPE> *pGamma = NULL;
        Tensorholder<DTYPE> *pBeta  = NULL;

        if (pIsChannelwise) {
            int pNumInputChannel = (*pInputShape)[2];
            pGamma = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumInputChannel, 1, 1, 1.0), "BatchNormalize_Gamma_" + pName);
            pBeta  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, pNumInputChannel, 1, 1), "BatchNormalize_Beta_" + pName);
        } else {
            int pNumInputCol = (*pInputShape)[4];
            pGamma = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumInputCol, 1.0), "BatchNormalize_Gamma_" + pName);
            pBeta  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, 1, 1, pNumInputCol), "BatchNormalize_Beta_" + pName);
        }
        // std::cout << pGamma->GetResult()->GetShape() << '\n';
        // std::cout << pBeta->GetResult()->GetShape() << '\n';

        out = new BatchNormalize<DTYPE>(out, pGamma, pBeta, pIsChannelwise, "BatchNormalize_BatchNormalize_" + pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __BATCH_NORMALIZE_LAYER__

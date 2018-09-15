#ifndef __CUDNN_BATCH_NORMALIZE__
#define __CUDNN_BATCH_NORMALIZE__    value

#include "../Operator.h"

#include <cmath>

/*!
@class
@details
@todo 우선순위
*/
// 문서 작성자 : , 작성 날짜 : 2018-
template<typename DTYPE>
class BatchNormalize : public Operator<DTYPE>{
private:
    Tensor<DTYPE> *m_pTenInput; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE> *m_pTenScale; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE> *m_pTenBias; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE> *m_pTenResult; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-

    Tensor<DTYPE> *m_pTenDerInput; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE> *m_pTenDerScale; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE> *m_pTenDerBias; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE> *m_pTenDerResult; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-

    Tensor<DTYPE> *m_aTenTotalMean; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE> *m_aTenTotalVariance; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-

    Tensor<DTYPE> *m_aTenCachedMean; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    Tensor<DTYPE> *m_aTenCachedInvVariance; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-

    int m_inputTimeSize; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_inputBatchSize; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_numChannel; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_numInputRow; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_numInputColumn; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-

    int m_isChannelwise; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    Mode m_mode; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-

    int m_inputCapacity; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    int m_batchSummaryCapacity; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-

    float m_epsilon; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    float m_momentum; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    double m_exponentialAverageFactor; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-

#ifdef __CUDNN__
    cudnnHandle_t m_CUDNNHandle; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    cudnnBatchNormMode_t m_CUDNNMode; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    cudnnTensorDescriptor_t m_CUDNNXDesc; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    cudnnTensorDescriptor_t m_CUDNNYDesc; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    cudnnTensorDescriptor_t m_CUDNNDxDesc; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    cudnnTensorDescriptor_t m_CUDNNDyDesc; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    cudnnTensorDescriptor_t m_CUDNNBatchSummaryDesc; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-

    float m_CUDNNAlpha; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    float m_CUDNNBeta; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    double m_CUDNNEpsilon; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
    double m_CUDNNExponentialAverageFactor; ///<   @todo 우선순위
    // 문서 작성자 : , 작성 날짜 : 2018-
#endif  // _CUDNN__

public:
    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    BatchNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise = TRUE, std::string pName = NULL) : Operator<DTYPE>(pInput, pScale, pBias, pName) {
#if __DEBUG__
        std::cout << "BatchNormalize:: BatchNormalize( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, std:: string)" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, pScale, pBias, pIsChannelwise);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    BatchNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise = TRUE, float pMomentum = 0.1, std::string pName = NULL) : Operator<DTYPE>(pInput, pScale, pBias, pName) {
#if __DEBUG__
        std::cout << "BatchNormalize:: BatchNormalize( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, std:: string)" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, pScale, pBias, pIsChannelwise, pMomentum);
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    ~BatchNormalize() {
#if __DEBUG__
        std::cout << "BatchNormalize:: ~ BatchNormalize()" << '\n';
#endif  // __DEBUG__

        Delete();
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise, float pMomentum = 0.1, double pEpsilon = 0.01) {
#if __DEBUG__
        std::cout << "BatchNormalize:: Alloc( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, double)" << '\n';
#endif  // __DEBUG__

        m_pTenInput = pInput->GetResult();
        m_pTenScale = pScale->GetResult();
        m_pTenBias  = pBias->GetResult();

        m_pTenDerInput = pInput->GetGradient();
        m_pTenDerScale = pScale->GetGradient();
        m_pTenDerBias  = pBias->GetGradient();

        m_inputCapacity = m_pTenInput->GetCapacity();

        Shape *pInputShape = m_pTenInput->GetShape();

        m_inputTimeSize  = m_pTenInput->GetTimeSize();
        m_inputBatchSize = m_pTenInput->GetBatchSize();
        m_numChannel     = m_pTenInput->GetChannelSize();
        m_numInputRow    = m_pTenInput->GetRowSize();
        m_numInputColumn = m_pTenInput->GetColSize();

        m_isChannelwise = pIsChannelwise;
        m_momentum      = pMomentum;

        if (m_isChannelwise) {
            m_batchSummaryCapacity  = m_numChannel;
            m_aTenTotalMean         = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
            m_aTenTotalVariance     = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
            m_aTenCachedMean        = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
            m_aTenCachedInvVariance = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
        } else {
            m_batchSummaryCapacity  = m_numChannel * m_numInputRow * m_numInputColumn;
            m_aTenTotalMean         = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, m_numInputRow, m_numInputColumn);
            m_aTenTotalVariance     = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, m_numInputRow, m_numInputColumn);
            m_aTenCachedMean        = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, m_numInputRow, m_numInputColumn);
            m_aTenCachedInvVariance = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, m_numInputRow, m_numInputColumn);
        }

        this->SetResult(new Tensor<DTYPE>(m_inputTimeSize, m_inputBatchSize, m_numChannel, m_numInputRow, m_numInputColumn));
        this->SetGradient(new Tensor<DTYPE>(m_inputTimeSize, m_inputBatchSize, m_numChannel, m_numInputRow, m_numInputColumn));

        m_pTenResult    = this->GetResult();
        m_pTenDerResult = this->GetGradient();

        m_mode = TRAINING;

        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        checkCudaErrors(cudaSetDevice(idOfDevice));

        if (m_isChannelwise) {
            m_CUDNNMode = CUDNN_BATCHNORM_SPATIAL;
        } else {
            m_CUDNNMode = CUDNN_BATCHNORM_PER_ACTIVATION;
        }

        m_CUDNNHandle = this->GetCudnnHandle();
        m_CUDNNXDesc  = m_pTenInput->GetDescriptor();
        m_CUDNNYDesc  = m_pTenResult->GetDescriptor();
        m_CUDNNDxDesc = m_pTenDerInput->GetDescriptor();
        m_CUDNNDyDesc = m_pTenDerResult->GetDescriptor();
        checkCUDNN(cudnnCreateTensorDescriptor(&m_CUDNNBatchSummaryDesc));
        checkCUDNN(cudnnDeriveBNTensorDescriptor(m_CUDNNBatchSummaryDesc, m_CUDNNXDesc, m_CUDNNMode));

        m_aTenTotalMean->SetDeviceGPU(idOfDevice);
        m_aTenTotalVariance->SetDeviceGPU(idOfDevice);
        m_aTenCachedMean->SetDeviceGPU(idOfDevice);
        m_aTenCachedInvVariance->SetDeviceGPU(idOfDevice);

        m_CUDNNAlpha   = 1.f;
        m_CUDNNBeta    = 0.f;
        m_CUDNNEpsilon = CUDNN_BN_MIN_EPSILON;

        if (m_momentum != 0) m_CUDNNExponentialAverageFactor = m_momentum;
        else m_CUDNNExponentialAverageFactor = 1.0;
    }

#endif  // if __CUDNN__

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    void Delete() {
#ifdef __CUDNN__
        checkCUDNN(cudnnDestroyTensorDescriptor(m_CUDNNBatchSummaryDesc));
        m_CUDNNBatchSummaryDesc = NULL;

        delete m_aTenTotalMean;
        delete m_aTenTotalVariance;
        delete m_aTenCachedMean;
        delete m_aTenCachedInvVariance;
#endif  // if __CUDNN__
    }

#ifdef __CUDNN__
    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int ForwardPropagateOnGPU(int pTime = 0) {
        DTYPE *CUDNNX = m_pTenInput->GetGPUData(pTime);

        DTYPE *CUDNNBnScale = m_pTenScale->GetGPUData(0);
        DTYPE *CUDNNBnBias  = m_pTenBias->GetGPUData(0);

        DTYPE *CUDNNY = m_pTenResult->GetGPUData(pTime);

        DTYPE *CUDNNTotalMean     = m_aTenTotalMean->GetGPUData(0);
        DTYPE *CUDNNTotalVariance = m_aTenTotalVariance->GetGPUData(0);

        DTYPE *CUDNNCachedMean        = NULL;
        DTYPE *CUDNNCachedInvVariance = NULL;

        float temp = 0.f;

        switch (m_mode) {
            case TRAINING:
                m_aTenCachedMean->Reset(m_CUDNNHandle);
                m_aTenCachedInvVariance->Reset(m_CUDNNHandle);
                CUDNNCachedMean        = m_aTenCachedMean->GetGPUData(0);
                CUDNNCachedInvVariance = m_aTenCachedInvVariance->GetGPUData(0);
                checkCUDNN(cudnnBatchNormalizationForwardTraining(
                               m_CUDNNHandle, m_CUDNNMode, &m_CUDNNAlpha, &m_CUDNNBeta,
                               m_CUDNNXDesc, CUDNNX, m_CUDNNYDesc, CUDNNY,
                               m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnBias,
                               m_CUDNNExponentialAverageFactor, CUDNNTotalMean, CUDNNTotalVariance,
                               m_CUDNNEpsilon, CUDNNCachedMean, CUDNNCachedInvVariance));

                if (m_momentum == 0) m_CUDNNExponentialAverageFactor = (m_CUDNNExponentialAverageFactor / (m_CUDNNExponentialAverageFactor + 1));  // for exponential
                break;
            case ACCUMULATING:
                checkCUDNN(cudnnBatchNormalizationForwardTraining(
                               m_CUDNNHandle, m_CUDNNMode, &m_CUDNNAlpha, &m_CUDNNBeta,
                               m_CUDNNXDesc, CUDNNX, m_CUDNNYDesc, CUDNNY,
                               m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnBias,
                               m_CUDNNExponentialAverageFactor, CUDNNTotalMean, CUDNNTotalVariance,
                               m_CUDNNEpsilon, NULL, NULL)
                           );
                m_CUDNNExponentialAverageFactor = (m_CUDNNExponentialAverageFactor / (m_CUDNNExponentialAverageFactor + 1));
                break;
            case INFERENCING:
                checkCUDNN(cudnnBatchNormalizationForwardInference(
                               m_CUDNNHandle, m_CUDNNMode, &m_CUDNNAlpha, &m_CUDNNBeta,
                               m_CUDNNXDesc, CUDNNX, m_CUDNNYDesc, CUDNNY,
                               m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnBias,
                               CUDNNTotalMean, CUDNNTotalVariance, m_CUDNNEpsilon));
                break;
            default:
                break;
        }

        checkCudaErrors(cudaDeviceSynchronize());


        return TRUE;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int BackPropagateOnGPU(int pTime = 0) {
        DTYPE *CUDNNX       = m_pTenInput->GetGPUData(pTime);
        DTYPE *CUDNNBnScale = m_pTenScale->GetGPUData(0);
        DTYPE *CUDNNBnBias  = m_pTenBias->GetGPUData(0);
        DTYPE *CUDNNDx      = m_pTenDerInput->GetGPUData(pTime);
        DTYPE *CUDNNDy      = m_pTenDerResult->GetGPUData(pTime);

        DTYPE *CUDNNCachedMean        = m_aTenCachedMean->GetGPUData(0);
        DTYPE *CUDNNCachedInvVariance = m_aTenCachedInvVariance->GetGPUData(0);

        DTYPE *CUDNNBnScaleDiff = m_pTenDerScale->GetGPUData(0);
        DTYPE *CUDNNBnBiasDiff  = m_pTenDerBias->GetGPUData(0);

        checkCUDNN(cudnnBatchNormalizationBackward(
                       this->GetCudnnHandle(), m_CUDNNMode,
                       &m_CUDNNAlpha, &m_CUDNNAlpha, &m_CUDNNAlpha, &m_CUDNNAlpha,
                       m_CUDNNXDesc, CUDNNX, m_CUDNNDyDesc, CUDNNDy, m_CUDNNDxDesc, CUDNNDx,
                       m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnScaleDiff, CUDNNBnBiasDiff,
                       m_CUDNNEpsilon, CUDNNCachedMean, CUDNNCachedInvVariance  /* CUDNNCachedMean, CUDNNCachedInvVariance*/));

        checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

#endif  // if __CUDNN__

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int SetModeTraining() {
        if (m_mode == ACCUMULATING) {
#ifdef __CUDNN__

            if (m_momentum == 0) m_CUDNNExponentialAverageFactor = 1.0;
            m_aTenTotalMean->Reset(m_CUDNNHandle);
            m_aTenTotalVariance->Reset(m_CUDNNHandle);
#endif  // ifdef __CUDNN__
        } else if (m_mode == INFERENCING) {
#ifdef __CUDNN__

            if (m_momentum == 0) m_CUDNNExponentialAverageFactor = 1.0;
            m_aTenTotalMean->Reset(m_CUDNNHandle);
            m_aTenTotalVariance->Reset(m_CUDNNHandle);
#endif  // ifdef __CUDNN__
        } else {
            return TRUE;
        }
        m_mode = TRAINING;

        return TRUE;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int SetModeAccumulating() {
        // std::cout << m_aTenTotalMean << '\n';
        // std::cout << m_aTenTotalVariance << '\n';

        if (m_mode == TRAINING) {
#ifdef __CUDNN__
            m_CUDNNExponentialAverageFactor = 1.0;
            m_aTenTotalMean->Reset(m_CUDNNHandle);
            m_aTenTotalVariance->Reset(m_CUDNNHandle);
#endif  // ifdef __CUDNN__
        } else if (m_mode == INFERENCING) {
#ifdef __CUDNN__
            m_CUDNNExponentialAverageFactor = 1.0;
            m_aTenTotalMean->Reset(m_CUDNNHandle);
            m_aTenTotalVariance->Reset(m_CUDNNHandle);
#endif  // ifdef __CUDNN__
        } else {
            return TRUE;
        }
        // std::cout << m_aTenTotalMean << '\n';
        // std::cout << m_aTenTotalVariance << '\n';

        m_mode = ACCUMULATING;
        return TRUE;
    }

    /*!
    @brief
    @details
    @param
    @return
    @todo 우선순위
    */
    // 문서 작성자 : , 작성 날짜 : 2018-
    int SetModeInferencing() {
        // std::cout << m_aTenTotalMean << '\n';
        // std::cout << m_aTenTotalVariance << '\n';
        if (m_mode == TRAINING) {
            ;
        } else if (m_mode == ACCUMULATING) {
            ;
        } else {
            return TRUE;
        }
        m_mode = INFERENCING;
        return TRUE;
    }
};

#endif  // __CUDNN_BATCH_NORMALIZE__

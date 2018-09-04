template<typename DTYPE> float NeuralNetwork<DTYPE>::GetTop5Accuracy(int numOfClass) {
    Operator<DTYPE> *result = GetResultOperator();
    Operator<DTYPE> *label  = m_aLossFunction->GetLabel();

    int batchsize = label->GetResult()->GetBatchSize();
    int timesize  = label->GetResult()->GetTimeSize();

    Tensor<DTYPE> *pred = result->GetResult();
    Tensor<DTYPE> *ans  = label->GetResult();

    float top5Accuracy = 0.f;

    int pred_index[5] = { 0, };
    int ans_index     = 0;

    for (int ba = 0; ba < batchsize; ba++) {
        for (int ti = 0; ti < timesize; ti++) {
            GetTop5Index(pred, pred_index, ba, ti, numOfClass);
            ans_index  = GetMaxIndex(ans, ba, ti, numOfClass);

            // pred_index[5] (top5Index) 중 하나라도 레이블과 같은 경우, 1을 더하고 break
            for(int i = 0; i < 5; i++){
                // printf("pred_index[%d] = %d, ans_Index = %d\n", i, pred_index[i], ans_index);
                if (pred_index[i] == ans_index) {
                    top5Accuracy += 1.f;
                    break;
                }
            }
        }
    }

    return (float)((top5Accuracy / timesize) / batchsize);
}

/*
상위 5개 노드의 값과 인덱스를 위한 5칸짜리 어레이 두 개를 생성
Value, Index Array 각각 0으로 초기화

어레이의 4부터 0까지 순서대로 큰 값들을 저장,
4인 경우 가장 큰 값과 인덱스, 0인 경우 5번째로 큰 값과 인덱스

Index 어레이는 Accuracy 메소드에서 생성한 후, 포인터를 통해 전달
텐서의 아웃풋 노드들과 하나씩 비교 및 스왑하면서 어레이를 채워감

swap method의 경우 std::swap 이용
각각의 아웃풋 노드들에 대해 먼저 0번째 값과 비교한 후,
노드의 값이 더 큰 경우 0번째 값과 인덱스의 해당 노드의 값과 인덱스을 대입
그 뒤 어레이의 원소들을 차례대로 비교하고 스왑이 필요한 경우 스왑, 필요 없는 경우 break (Sorting)
*/

template<typename DTYPE> void NeuralNetwork<DTYPE>::GetTop5Index(Tensor<DTYPE> *data, int* top5Index, int ba, int ti, int numOfClass) {
    Shape *pShape = data->GetShape();
    int    start  = Index5D(pShape, ti, ba, 0, 0, 0);
    int    end    = start + numOfClass;

    // Initialize array with 0
    DTYPE top5Value[5] = { 0, };

    // Find 5 top elements
    for (int dim = start; dim < end; dim++) {
        //printf("(*data)(%d) = %f, top5Value[0] = %f\n", dim, (float)(*data)[dim], (float)top5Value[0]);

        if((*data)[dim] > top5Value[0])
        {
            //printf("if((*data)[dim] > top5Value[0]) clear\n");
            top5Value[0] = (*data)[dim];
            top5Index[0] = dim - start;
            //printf("top5Value[0] = %f, top5Index[0] = %d\n", (float)top5Value[0], (float)top5index[0]);
            for(int i = 0; i < 4; i++){
                //printf("for(int i = 0; i < 4; i++) clear\n");
                //printf("top5Value[0] = %f, top5Index[0] = %d\n", (float)top5Value[0], (float)top5index[0]);
                if(top5Value[i] > top5Value[i+1])
                {
                    //printf("if(top5Value[i] > top5Value[i+1]) clear\n");
                    //printf("top5Value[%d] = %f, top5Index[%d] = %d\n", i, (float)top5Value[i], i, (float)top5index[i]);
                    std::swap(top5Value[i], top5Value[i+1]);
                    std::swap(top5Index[i], top5Index[i+1]);
                    //printf("swap clear\n");
                    //printf("top5Value[%d] = %f, top5Index[%d] = %d\n", i, (float)top5Value[i], i, (float)top5index[i]);
                }
                else
                    break;
            }
        }
    }
}

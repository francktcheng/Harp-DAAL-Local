/* file: mf_sgd_dense_default_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Declaration of template function that calculate mf_sgds.
//--
*/

#ifndef __MF_SGD_FPK_H__
#define __MF_SGD_FPK_H__

#include "mf_sgd_batch.h"
#include "kernel.h"
#include "numeric_table.h"

// head files for TBB
#include "task_scheduler_init.h"
#include "blocked_range.h"
#include "parallel_for.h"
#include "queuing_mutex.h"

using namespace tbb;
using namespace daal::data_management;

typedef queuing_mutex currentMutex_t;

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
namespace internal
{

//MF-SGD-BATCH kernel
template<typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
class MF_SGDBatchKernel : public Kernel
{
public:

    void compute(NumericTable** TrainSet,NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par);

    void compute_thr(NumericTable** TrainSet,NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par);

};

//MF-SGD-DISTRI kernel
template<typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
class MF_SGDDistriKernel : public Kernel
{
public:

    void compute(NumericTable** TrainWPos, NumericTable** TrainHPos, NumericTable** TrainVal, NumericTable *r[], const daal::algorithms::Parameter *par);

    void compute_thr(NumericTable** TrainWPos, NumericTable** TrainHPos, NumericTable** TrainVal, NumericTable *r[], const daal::algorithms::Parameter *par);

};

/* TBB kernel for MF-SGD */
template<typename interm, CpuType cpu>
struct MFSGDTBB
{
    interm* _mtWDataTable;
    interm* _mtHDataTable;

    int* _workWPos;
    int* _workHPos;
    interm* _workV;

    long _Dim;
    interm _learningRate;
    interm _lambda;

    int _Avx512_explicit;

    int _step;
    int _dim_train;
    int _itr;

    currentMutex_t* _mutex_w;
    currentMutex_t* _mutex_h;

    
    MFSGDTBB(
            interm* mtWDataTable,
            interm* mtHDataTable,
            int* workWPos,
            int* workHPos,
            interm *workV,
            const long Dim,
            const interm learningRate,
            const interm lambda,
            currentMutex_t* mutex_w,
            currentMutex_t* mutex_h,
            const int Avx512_explicit,
            const int step,
            const int dim_train
    );

    void operator()( const blocked_range<int>& range ) const; 

    void setItr(int itr)
    {
        _itr = itr;
    }


};

/* MF_SGD_Test kernel implemented by TBB */
// calculating the RMSE value
template<typename interm, CpuType cpu>
struct MFSGDTBB_TEST
{
    interm* _mtWDataTable;
    interm* _mtHDataTable;

    int* _testWPos;
    int* _testHPos;
    interm* _testV;

    int _Dim;
    interm* _testRMSE;
    int _Avx512_explicit;

    currentMutex_t* _mutex_w;
    currentMutex_t* _mutex_h;

    MFSGDTBB_TEST(

            interm* mtWDataTable,
            interm* mtHDataTable,
            int* testWPos,
            int* testHPos,
            interm *testV,
            const long Dim,
            interm* testRMSE,
            currentMutex_t* mutex_w,
            currentMutex_t* mutex_h,
            const int Avx512_explicit
    );

    void operator()( const blocked_range<int>& range ) const; 


};
} // namespace daal::internal
}
}
} // namespace daal

#endif

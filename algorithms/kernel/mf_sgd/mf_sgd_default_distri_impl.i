/* file: mf_sgd_default_distri_impl.i */
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
//  Implementation of distributed mode mf_sgd method
//--
*/

#ifndef __MF_SGD_KERNEL_DISTRI_IMPL_I__
#define __MF_SGD_KERNEL_DISTRI_IMPL_I__

#include <time.h>
#include <math.h>       
#include <algorithm>
#include <cstdlib> 
#include <iostream>

#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "threading.h"
#include "task_scheduler_init.h"
#include "blocked_range.h"
#include "parallel_for.h"
#include "queuing_mutex.h"

#include "mf_sgd_default_impl.i"

using namespace tbb;
using namespace daal::internal;
using namespace daal::services::internal;

typedef queuing_mutex currentMutex_t;

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
namespace internal
{
    
template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute(const NumericTable** TrainWPos, 
                                                      const NumericTable** TrainHPos, 
                                                      const NumericTable** TrainVal, 
                                                      NumericTable *r[], const daal::algorithms::Parameter *par)
{
    MF_SGDDistriKernel<interm, method, cpu>::compute_thr(TrainWPos,TrainHPos,TrainVal, r, par);
}

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute_thr(const NumericTable** TrainWPos, 
                                                          const NumericTable** TrainHPos, 
                                                          const NumericTable** TrainVal, 
                                                          NumericTable *r[], 
                                                          const daal::algorithms::Parameter *par)
{/*{{{*/

    /* retrieve members of parameter */
    const Parameter *parameter = static_cast<const Parameter *>(par);
    const long dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    const double learningRate = parameter->_learningRate;
    const double lambda = parameter->_lambda;
    const int iteration = parameter->_iteration;
    const int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx512_explicit = parameter->_Avx512_explicit;

    const double ratio = parameter->_ratio;
    const int itr = parameter->_itr;

    const int dim_train = TrainVal[0]->getNumberOfRows();

    /* ------------- Retrieve Training Data Set -------------*/

    FeatureMicroTable<int, readOnly, cpu> workflowW_ptr(TrainWPos[0]);
    FeatureMicroTable<int, readOnly, cpu> workflowH_ptr(TrainHPos[0]);
    FeatureMicroTable<interm, readOnly, cpu> workflow_ptr(TrainVal[0]);

    int *workWPos = 0;
    workflowW_ptr.getBlockOfColumnValues(0, 0, dim_train, &workWPos);

    int *workHPos = 0;
    workflowH_ptr.getBlockOfColumnValues(0, 0, dim_train, &workHPos);

    interm *workV;
    workflow_ptr.getBlockOfColumnValues(0, 0, dim_train, &workV);

    /* ---------------- Retrieve Model W ---------------- */
    BlockMicroTable<interm, readWrite, cpu> mtWDataTable(r[0]);

    interm* mtWDataPtr = 0;
    mtWDataTable.getBlockOfRows(0, dim_w, &mtWDataPtr);

    /* ---------------- Retrieve Model H ---------------- */
    BlockMicroTable<interm, readWrite, cpu> mtHDataTable(r[1]);

    interm* mtHDataPtr = 0;
    mtHDataTable.getBlockOfRows(0, dim_h, &mtHDataPtr);

    /* create the mutex for WData and HData */
    services::SharedPtr<currentMutex_t> mutex_w(new currentMutex_t[dim_w]);
    services::SharedPtr<currentMutex_t> mutex_h(new currentMutex_t[dim_h]);

    /* ------------------- Starting TBB based Training  -------------------*/
    task_scheduler_init init(task_scheduler_init::deferred);

    if (thread_num != 0)
    {
        /* use explicitly specified num of threads  */
        init.initialize(thread_num);
    }
    else
    {
        /* use automatically generated threads by TBB  */
        init.initialize();
    }

    /* if ratio != 1, dim_ratio is the ratio of computed tasks */
    int dim_ratio = (int)(ratio*dim_train);

    /* step is the stride of choosing tasks in a rotated way */
    const int step = dim_train - dim_ratio;

    MFSGDTBB<interm, cpu> mfsgd(mtWDataPtr, mtHDataPtr, workWPos, workHPos, workV, dim_r, learningRate, lambda, mutex_w.get(), mutex_h.get(), Avx512_explicit, step, dim_train);
    mfsgd.setItr(itr);

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double train_time = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    /* training MF-SGD */
    if (tbb_grainsize != 0)
        parallel_for(blocked_range<int>(0, dim_ratio, tbb_grainsize), mfsgd);
    else
        parallel_for(blocked_range<int>(0, dim_ratio), mfsgd, auto_partitioner());

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    /* get the training time for each iteration */
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    train_time += (double)(diff)/1000000L;

    init.terminate();
    std::printf("Training time this iteration: %f\n", train_time);
    std::fflush(stdout);

    return;

}/*}}}*/


} // namespace daal::internal
}
}
} // namespace daal

#endif

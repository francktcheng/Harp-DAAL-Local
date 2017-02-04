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
#include <cstdio> 
#include <algorithm>
#include <ctime>        
#include <omp.h>
#include <immintrin.h>

#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "threading.h"
#include "tbb/tick_count.h"
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
void MF_SGDDistriKernel<interm, method, cpu>::compute(const NumericTable** WPos, 
                                                      const NumericTable** HPos, 
                                                      const NumericTable** Val, 
                                                      NumericTable** WPosTest,
                                                      NumericTable** HPosTest, 
                                                      NumericTable** ValTest, 
                                                      NumericTable *r[], const daal::algorithms::Parameter *par)
{/*{{{*/
    /* retrieve members of parameter */
    const Parameter *parameter = static_cast<const Parameter *>(par);
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    const int isTrain = parameter->_isTrain;
    const int dim_set = Val[0]->getNumberOfRows();
    const int isSGD2 = parameter->_sgd2;

    if (isSGD2 == 0)
    {
        /* ------------- Retrieve Data Set -------------*/
        FeatureMicroTable<int, readOnly, cpu> workflowW_ptr(WPos[0]);
        FeatureMicroTable<int, readOnly, cpu> workflowH_ptr(HPos[0]);
        FeatureMicroTable<interm, readOnly, cpu> workflow_ptr(Val[0]);

        int *workWPos = 0;
        workflowW_ptr.getBlockOfColumnValues(0, 0, dim_set, &workWPos);

        int *workHPos = 0;
        workflowH_ptr.getBlockOfColumnValues(0, 0, dim_set, &workHPos);

        interm *workV;
        workflow_ptr.getBlockOfColumnValues(0, 0, dim_set, &workV);

        /* ---------------- Retrieve Model W ---------------- */
        BlockMicroTable<interm, readWrite, cpu> mtWDataTable(r[0]);

        interm* mtWDataPtr = 0;
        mtWDataTable.getBlockOfRows(0, dim_w, &mtWDataPtr);

        /* ---------------- Retrieve Model H ---------------- */
        BlockMicroTable<interm, readWrite, cpu> mtHDataTable(r[1]);

        interm* mtHDataPtr = 0;
        mtHDataTable.getBlockOfRows(0, dim_h, &mtHDataPtr);

        if (isTrain == 1)
        {
            /* MF_SGDDistriKernel<interm, method, cpu>::compute_train(workWPos,workHPos,workV, dim_set, mtWDataPtr, mtHDataPtr, parameter);  */
            MF_SGDDistriKernel<interm, method, cpu>::compute_train_omp(workWPos,workHPos,workV, dim_set, mtWDataPtr, mtHDataPtr, parameter); 
        }
        else
        {
            /* ---------------- Retrieve RMSE for Test dataset --------- */
            interm* mtRMSEPtr = 0;
            BlockMicroTable<interm, readWrite, cpu> mtRMSETable(r[2]);
            mtRMSETable.getBlockOfRows(0, 1, &mtRMSEPtr);
            /* MF_SGDDistriKernel<interm, method, cpu>::compute_test(workWPos,workHPos,workV, dim_set, mtWDataPtr,mtHDataPtr, mtRMSEPtr, parameter); */
            MF_SGDDistriKernel<interm, method, cpu>::compute_test_omp(workWPos,workHPos,workV, dim_set, mtWDataPtr,mtHDataPtr, mtRMSEPtr, parameter);
        }

    }
    else
    {


    }

}/*}}}*/

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute_train(int* workWPos, 
                                                            int* workHPos, 
                                                            interm* workV, 
                                                            const int dim_set,
                                                            interm* mtWDataPtr, 
                                                            interm* mtHDataPtr, 
                                                            const Parameter *parameter)
{/*{{{*/

    /* retrieve members of parameter */
    const long dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    const double learningRate = parameter->_learningRate;
    const double lambda = parameter->_lambda;
    const int iteration = parameter->_iteration;
    const int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;

    const double ratio = parameter->_ratio;
    const int itr = parameter->_itr;
    double timeout = parameter->_timeout;

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
    int dim_ratio = (int)(ratio*dim_set);

    /* step is the stride of choosing tasks in a rotated way */
    const int step = dim_set - dim_ratio;

    /* randomize the execution order */
    /* services::SharedPtr<int> random_index(new int[dim_set]);  */
    /* int* random_index_ptr = random_index.get(); */
    /* for(int j=0;j<dim_set;j++) */
    /*     random_index_ptr[j] = j; */

    /* std::srand ( unsigned (std::time(0))); */
    /* std::random_shuffle(random_index_ptr,random_index_ptr + dim_set - 1); */

    MFSGDTBB<interm, cpu> mfsgd(mtWDataPtr, mtHDataPtr, workWPos, workHPos, workV, dim_r, learningRate, lambda, mutex_w.get(), mutex_h.get(), Avx_explicit, step, dim_set);
    mfsgd.setItr(itr);
    mfsgd.setTimeStart(tbb::tick_count::now());
    mfsgd.setTimeOut(timeout);

    /* test randomize order */
    /* mfsgd.setOrder(random_index_ptr); */

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

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute_train_omp(int* workWPos, 
                                                                int* workHPos, 
                                                                interm* workV, 
                                                                const int dim_set,
                                                                interm* mtWDataPtr, 
                                                                interm* mtHDataPtr, 
                                                                const Parameter *parameter)
{/*{{{*/

#ifdef _OPENMP

    /* retrieve members of parameter */
    const long dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    const double learningRate = parameter->_learningRate;
    const double lambda = parameter->_lambda;
    const int iteration = parameter->_iteration;
    int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;

    const double ratio = parameter->_ratio;
    const int itr = parameter->_itr;
    double timeout = parameter->_timeout;

    /* create the mutex for WData and HData */
    /* services::SharedPtr<currentMutex_t> mutex_w(new currentMutex_t[dim_w]); */
    /* services::SharedPtr<currentMutex_t> mutex_h(new currentMutex_t[dim_h]); */

    /* omp_lock_t* mutex_w_ptr = mutex_w.get(); */
    /* for(int j=0; j<dim_w;j++) */
         /* omp_init_lock(&(mutex_w_ptr[j])); */

    /* omp_lock_t* mutex_h_ptr = mutex_h.get(); */
    /* for(int j=0; j<dim_h;j++) */
         /* omp_init_lock(&(mutex_h_ptr[j])); */

    /* ------------------- Starting OpenMP based Training  -------------------*/
    int num_thds_max = omp_get_max_threads();
    std::printf("Max threads number: %d\n", num_thds_max);
    std::fflush(stdout);

    if (thread_num == 0)
        thread_num = num_thds_max;

    /* if ratio != 1, dim_ratio is the ratio of computed tasks */
    int dim_ratio = (int)(ratio*dim_set);

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double train_time = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    /* training MF-SGD */
    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(int k=0;k<dim_ratio;k++)
    {

        interm *WMat = 0;
        interm *HMat = 0;

        interm Mult = 0;
        interm Err = 0;
        interm WMatVal = 0;
        interm HMatVal = 0;
        int p = 0;

        interm* mtWDataLocal = mtWDataPtr;
        interm* mtHDataLocal = mtHDataPtr;

        int* workWPosLocal = workWPos;
        int* workHPosLocal = workHPos;
        interm* workVLocal = workV;

        long Dim = dim_r;
        interm learningRateLocal = learningRate;
        interm lambdaLocal = lambda;

        WMat = mtWDataLocal + workWPosLocal[k]*Dim;
        HMat = mtHDataLocal + workHPosLocal[k]*Dim;


        for(p = 0; p<Dim; p++)
            Mult += (WMat[p]*HMat[p]);


        Err = workVLocal[k] - Mult;

        for(p = 0;p<Dim;p++)
        {

            WMatVal = WMat[p];
            HMatVal = HMat[p];

            WMat[p] = WMatVal + learningRateLocal*(Err*HMatVal - lambdaLocal*WMatVal);
            HMat[p] = HMatVal + learningRateLocal*(Err*WMatVal - lambdaLocal*HMatVal);

        }


    }

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    /* get the training time for each iteration */
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    train_time += (double)(diff)/1000000L;

    /* init.terminate(); */
    std::printf("Training time this iteration: %f\n", train_time);
    std::fflush(stdout);

#else

    std::printf("Error: OpenMP module is not enabled\n");
    std::fflush(stdout);

#endif

    return;

}/*}}}*/

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute_test(int* workWPos, 
                                                            int* workHPos, 
                                                            interm* workV, 
                                                            const int dim_set,
                                                            interm* mtWDataPtr, 
                                                            interm* mtHDataPtr, 
                                                            interm* mtRMSEPtr,
                                                            const Parameter *parameter)
{/*{{{*/

    /* retrieve members of parameter */
    const long dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    const int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;

    /* create the mutex for WData and HData */
    services::SharedPtr<currentMutex_t> mutex_w(new currentMutex_t[dim_w]);
    services::SharedPtr<currentMutex_t> mutex_h(new currentMutex_t[dim_h]);

    /* RMSE value for test dataset */
    services::SharedPtr<interm> testRMSE(new interm[dim_set]);

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

    MFSGDTBB_TEST<interm, cpu> mfsgd_test(mtWDataPtr, mtHDataPtr, workWPos, workHPos, workV, dim_r, testRMSE.get(), mutex_w.get(), mutex_h.get(), Avx_explicit);

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double test_time = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    /* test dataset compute RMSE for MF-SGD */
    if (tbb_grainsize != 0)
        parallel_for(blocked_range<int>(0, dim_set, tbb_grainsize), mfsgd_test);
    else
        parallel_for(blocked_range<int>(0, dim_set), mfsgd_test, auto_partitioner());

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    test_time += (double)(diff)/1000000L;

    init.terminate();

    interm totalRMSE = 0;
    interm* testRMSE_ptr = testRMSE.get();

    for(int k=0;k<dim_set;k++)
        totalRMSE += testRMSE_ptr[k];

    mtRMSEPtr[0] = totalRMSE;

    return;

}/*}}}*/

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute_test_omp(int* workWPos, 
                                                               int* workHPos, 
                                                               interm* workV, 
                                                               const int dim_set,
                                                               interm* mtWDataPtr, 
                                                               interm* mtHDataPtr, 
                                                               interm* mtRMSEPtr,
                                                               const Parameter *parameter)
{/*{{{*/

#ifdef _OPENMP

    /* retrieve members of parameter */
    const long dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;

    /* create the mutex for WData and HData */
    services::SharedPtr<omp_lock_t> mutex_w(new omp_lock_t[dim_w]);
    services::SharedPtr<omp_lock_t> mutex_h(new omp_lock_t[dim_h]);

    /* RMSE value for test dataset */
    services::SharedPtr<interm> testRMSE(new interm[dim_set]);

    omp_lock_t* mutex_w_ptr = mutex_w.get();
    for(int j=0; j<dim_w;j++)
         omp_init_lock(&(mutex_w_ptr[j]));

    omp_lock_t* mutex_h_ptr = mutex_h.get();
    for(int j=0; j<dim_h;j++)
         omp_init_lock(&(mutex_h_ptr[j]));

    /* ------------------- Starting OpenMP based testing  -------------------*/
    int num_thds_max = omp_get_max_threads();
    std::printf("Max threads number: %d\n", num_thds_max);
    std::fflush(stdout);

    if (thread_num == 0)
        thread_num = num_thds_max;

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double test_time = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(int k = 0; k<dim_set; k++)
    {

        interm *WMat = 0;
        interm *HMat = 0;

        interm* mtWDataLocal = mtWDataPtr;
        interm* mtHDataLocal = mtHDataPtr;

        int* testWPosLocal = workWPos;
        int* testHPosLocal = workHPos;

        interm* testVLocal = workV;

        long Dim = dim_r;
        interm* testRMSELocal = testRMSE.get();

        int p = 0;
        interm Mult = 0;
        interm Err = 0;

        if (testWPosLocal[k] != -1 && testHPosLocal[k] != -1)
        {

            omp_set_lock(&(mutex_w_ptr[testWPosLocal[k]]));
            omp_set_lock(&(mutex_h_ptr[testHPosLocal[k]]));

            WMat = mtWDataLocal + testWPosLocal[k]*Dim;
            HMat = mtHDataLocal + testHPosLocal[k]*Dim;

            for(p = 0; p<Dim; p++)
                Mult += (WMat[p]*HMat[p]);

            Err = testVLocal[k] - Mult;

            testRMSELocal[k] = Err*Err;

            omp_unset_lock(&(mutex_w_ptr[testWPosLocal[k]]));
            omp_unset_lock(&(mutex_h_ptr[testHPosLocal[k]]));

        }
        else
            testRMSELocal[k] = 0;

    }

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    test_time += (double)(diff)/1000000L;

    interm totalRMSE = 0;
    interm* testRMSE_ptr = testRMSE.get();

    for(int k=0;k<dim_set;k++)
        totalRMSE += testRMSE_ptr[k];

    mtRMSEPtr[0] = totalRMSE;

#else

    std::printf("Error: OpenMP module is not enabled\n");
    std::fflush(stdout);

#endif

    return;

}/*}}}*/

} // namespace daal::internal
}
}
} // namespace daal

#endif

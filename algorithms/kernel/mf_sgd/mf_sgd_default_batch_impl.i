/* file: mf_sgd_dense_default_batch_impl.i */
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
//  Implementation of mf_sgds batch mode
//--
*/

#ifndef __MF_SGD_KERNEL_BATCH_IMPL_I__
#define __MF_SGD_KERNEL_BATCH_IMPL_I__

#include <time.h>
#include <math.h>       
#include <algorithm>
#include <cstdlib> 
#include <cstdio> 
#include <iostream>
#include <vector>

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
void MF_SGDBatchKernel<interm, method, cpu>::compute(const NumericTable** TrainSet, const NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par)
{/*{{{*/

    /* retrieve members of parameter */
    const Parameter *parameter = static_cast<const Parameter *>(par);
    const int64_t dim_w = parameter->_Dim_w;
    const int64_t dim_h = parameter->_Dim_h;

    const int dim_train = TrainSet[0]->getNumberOfRows();
    const int dim_test = TestSet[0]->getNumberOfRows();

    /*  Retrieve Training Data Set */
    FeatureMicroTable<int, readOnly, cpu> workflowW_ptr(TrainSet[0]);
    FeatureMicroTable<int, readOnly, cpu> workflowH_ptr(TrainSet[0]);
    FeatureMicroTable<interm, readOnly, cpu> workflow_ptr(TrainSet[0]);

    int *workWPos = 0;
    workflowW_ptr.getBlockOfColumnValues(0, 0, dim_train, &workWPos);

    int *workHPos = 0;
    workflowH_ptr.getBlockOfColumnValues(1, 0, dim_train, &workHPos);

    interm *workV;
    workflow_ptr.getBlockOfColumnValues(2,0,dim_train,&workV);

    /*  Retrieve Test Data Set */
    FeatureMicroTable<int, readOnly, cpu> testW_ptr(TestSet[0]);
    FeatureMicroTable<int, readOnly, cpu> testH_ptr(TestSet[0]);
    FeatureMicroTable<interm, readOnly, cpu> test_ptr(TestSet[0]);

    int *testWPos = 0;
    testW_ptr.getBlockOfColumnValues(0, 0, dim_test, &testWPos);

    int *testHPos = 0;
    testH_ptr.getBlockOfColumnValues(1, 0, dim_test, &testHPos);

    interm *testV;
    test_ptr.getBlockOfColumnValues(2, 0, dim_test, &testV);

    /*  Retrieve Model W  */
    BlockMicroTable<interm, readWrite, cpu> mtWDataTable(r[0]);

    /* screen print out the size of model data */
    std::printf("model W row: %zu\n",r[0]->getNumberOfRows());
    std::fflush(stdout);
    std::printf("model W col: %zu\n",r[0]->getNumberOfColumns());
    std::fflush(stdout);

    interm* mtWDataPtr = 0;
    mtWDataTable.getBlockOfRows(0, dim_w, &mtWDataPtr);

    /*  Retrieve Model H  */
    BlockMicroTable<interm, readWrite, cpu> mtHDataTable(r[1]);

    interm* mtHDataPtr = 0;
    mtHDataTable.getBlockOfRows(0, dim_h, &mtHDataPtr);

    MF_SGDBatchKernel<interm, method, cpu>::reorder(workWPos, workHPos, workV, dim_train, parameter);
    MF_SGDBatchKernel<interm, method, cpu>::compute_thr(workWPos, workHPos, workV, dim_train, testWPos, testHPos, testV, dim_test, mtWDataPtr, mtHDataPtr, parameter);

}/*}}}*/

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDBatchKernel<interm, method, cpu>::reorder(int* trainWPos, int* trainHPos, interm* trainV, const int train_num, const Parameter *parameter)
{
    const size_t length = 200;
    const int64_t dim_w = parameter->_Dim_w;
 
    services::SharedPtr<std::vector<int>* > trainHPool(new std::vector<int>*[dim_w]);
    services::SharedPtr<std::vector<interm>* > trainVPool(new std::vector<interm>*[dim_w]);

    for(int i=0;i<dim_w;i++)
    {
        (trainHPool.get())[i] = new std::vector<int>();
        (trainVPool.get())[i] = new std::vector<interm>();
    }

    services::SharedPtr<std::vector<int> > trainWQueue(new std::vector<int>());
    services::SharedPtr<std::vector<int*> > trainHQueue(new std::vector<int*>());
    services::SharedPtr<std::vector<interm*> > trainVQueue(new std::vector<interm*>());

    for(int i=0;i<train_num;i++)
    {
        int w_idx = trainWPos[i];
        int h_idx = trainHPos[i];
        interm v_val = trainV[i];

        if (trainHPool.get()[w_idx]->size() > length)
        {
            /* move vector to trainWQueue */
            int* trainHVec = &(*(trainHPool.get()[w_idx]))[0];
            interm* trainVVec = &(*(trainVPool.get()[w_idx]))[0];

            trainWQueue->push_back(w_idx);
            trainHQueue->push_back(trainHVec);
            trainVQueue->push_back(trainVVec);

            trainHPool.get()[w_idx] = new std::vector<int>();
            trainVPool.get()[w_idx] = new std::vector<interm>();
            
        }
        else
        {
            trainHPool.get()[w_idx]->push_back(h_idx);
            trainVPool.get()[w_idx]->push_back(v_val);
        }

    }

    /* move the rest data points at pools to queue */
    for(int i=0;i<dim_w;i++)
    {
        int* trainHVec = &(*(trainHPool.get()[i]))[0];
        interm* trainVVec = &(*(trainVPool.get()[i]))[0];

        trainWQueue->push_back(i);
        trainHQueue->push_back(trainHVec);
        trainVQueue->push_back(trainVVec);

    }

    std::printf("trainWQueue size: %zu\n", trainWQueue->size());
    std::fflush(stdout);

    std::printf("trainHQueue size: %zu\n", trainHQueue->size());
    std::fflush(stdout);

    std::printf("trainVQueue size: %zu\n", trainVQueue->size());
    std::fflush(stdout);

    _trainWQueue = trainWQueue;
    _trainHQueue = trainHQueue;
    _trainVQueue = trainVQueue;

}

/* A multi-threading version by using TBB */
template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDBatchKernel<interm, method, cpu>::compute_thr_reordered(int* testWPos, int* testHPos, interm* testV, const int dim_test,
                                                                   interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter)
{/*{{{*/

    /* retrieve members of parameter */
    const int64_t dim_r = parameter->_Dim_r;
    const int64_t dim_w = parameter->_Dim_w;
    const int64_t dim_h = parameter->_Dim_h;
    const double learningRate = parameter->_learningRate;
    const double lambda = parameter->_lambda;
    const int iteration = parameter->_iteration;
    const int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;
    const size_t absent_test_num = parameter->_absent_test_num;

    const double ratio = parameter->_ratio;
    const int itr = parameter->_itr;

    /* get the reordered training data points */
    const int train_queue_size = _trainWQueue->size(); 
    int* train_queue_wPos = &(*(_trainWQueue.get()))[0];
    int** train_queue_hPos = &(*(_trainHQueue.get()))[0];
    interm** train_queue_vVal = &(*(_trainVQueue.get()))[0];

    /* create the mutex for WData and HData */
    services::SharedPtr<currentMutex_t> mutex_w(new currentMutex_t[dim_w]);
    services::SharedPtr<currentMutex_t> mutex_h(new currentMutex_t[dim_h]);

    /* RMSE value for test dataset after each iteration */
    services::SharedPtr<interm> testRMSE(new interm[dim_test]);

    interm totalRMSE = 0;

    /*  Starting TBB based Training  */
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
    // const int dim_ratio = static_cast<const int>(ratio*dim_train);

    /* step is the stride of choosing tasks in a rotated way */
    // const int step = dim_train - dim_ratio;

    /* to use affinity_partitioner */
    static affinity_partitioner ap;

    // MFSGDTBB<interm, cpu> mfsgd(mtWDataPtr, mtHDataPtr, workWPos, workHPos, workV, dim_r, learningRate, lambda, mutex_w.get(), mutex_h.get(), Avx_explicit, step, dim_train);

    MFSGDTBB_TEST<interm, cpu> mfsgd_test(mtWDataPtr, mtHDataPtr, testWPos, testHPos, testV, dim_r, testRMSE.get(), mutex_w.get(), mutex_h.get(), Avx_explicit);

    /* Test MF-SGD before iteration */
    if (tbb_grainsize != 0)
    {
        /* use explicitly specified grainsize */
        parallel_for(blocked_range<int>(0, dim_test, tbb_grainsize), mfsgd_test);
    }
    else
    {
        /* use auto-partitioner by TBB */
        /* parallel_for(blocked_range<int>(0, dim_test), mfsgd_test, auto_partitioner()); */

        /* use affinity-partitioner by TBB */
        parallel_for(blocked_range<int>(0, dim_test), mfsgd_test, ap);

    }

    totalRMSE = 0;
    interm* testRMSE_ptr = testRMSE.get();

    for(int k=0;k<dim_test;k++)
        totalRMSE += testRMSE_ptr[k];

    totalRMSE = totalRMSE/(dim_test - absent_test_num);

    std::printf("RMSE before interation: %f\n", sqrt(totalRMSE));
    std::fflush(stdout);

    /* End of Test MF-SGD before iteration */
    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double train_time = 0;

    for(int j=0;j<iteration;j++)
    {
        // mfsgd.setItr(j);
        clock_gettime(CLOCK_MONOTONIC, &ts1);

        /* training MF-SGD */
        if (tbb_grainsize != 0)
        {
            // parallel_for(blocked_range<int>(0, dim_ratio, tbb_grainsize), mfsgd);
        }
        else
        {
            /* parallel_for(blocked_range<int>(0, dim_ratio), mfsgd, auto_partitioner()); */
            // parallel_for(blocked_range<int>(0, dim_ratio), mfsgd, ap);
        }

        clock_gettime(CLOCK_MONOTONIC, &ts2);

        /* get the training time for each iteration */
        diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
	    /* train_time += (double)(diff)/1000000L; */
	    train_time += static_cast<double>((diff)/1000000L);

        /* Test MF-SGD */
        if (tbb_grainsize != 0)
        {
            parallel_for(blocked_range<int>(0, dim_test, tbb_grainsize), mfsgd_test);
        }
        else
        {
            /* parallel_for(blocked_range<int>(0, dim_test), mfsgd_test, auto_partitioner()); */
            parallel_for(blocked_range<int>(0, dim_test), mfsgd_test, ap);
        }

        totalRMSE = 0;
        for(int k=0;k<dim_test;k++)
            totalRMSE += testRMSE_ptr[k];

        totalRMSE = totalRMSE/(dim_test - absent_test_num);
        std::printf("RMSE after interation %d: %f, train time: %f\n", j, sqrt(totalRMSE), static_cast<double>(diff/1000000L));
        std::fflush(stdout);
    }

    init.terminate();

    std::printf("Average training time per iteration: %f, total time: %f\n", train_time/iteration, train_time);
    std::fflush(stdout);

    return;

}/*}}}*/

/* A multi-threading version by using TBB */
template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDBatchKernel<interm, method, cpu>::compute_thr(int* workWPos, int* workHPos, interm* workV, const int dim_train,
                                                         int* testWPos, int* testHPos, interm* testV, const int dim_test,
                                                         interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter)
{/*{{{*/

    /* retrieve members of parameter */
    const int64_t dim_r = parameter->_Dim_r;
    const int64_t dim_w = parameter->_Dim_w;
    const int64_t dim_h = parameter->_Dim_h;
    const double learningRate = parameter->_learningRate;
    const double lambda = parameter->_lambda;
    const int iteration = parameter->_iteration;
    const int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;
    const size_t absent_test_num = parameter->_absent_test_num;

    const double ratio = parameter->_ratio;
    const int itr = parameter->_itr;

    /* create the mutex for WData and HData */
    services::SharedPtr<currentMutex_t> mutex_w(new currentMutex_t[dim_w]);
    services::SharedPtr<currentMutex_t> mutex_h(new currentMutex_t[dim_h]);

    /* RMSE value for test dataset after each iteration */
    services::SharedPtr<interm> testRMSE(new interm[dim_test]);

    interm totalRMSE = 0;

    /*  Starting TBB based Training  */
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
    const int dim_ratio = static_cast<const int>(ratio*dim_train);

    /* step is the stride of choosing tasks in a rotated way */
    const int step = dim_train - dim_ratio;

    /* to use affinity_partitioner */
    static affinity_partitioner ap;

    MFSGDTBB<interm, cpu> mfsgd(mtWDataPtr, mtHDataPtr, workWPos, workHPos, workV, dim_r, learningRate, lambda, mutex_w.get(), mutex_h.get(), Avx_explicit, step, dim_train);

    MFSGDTBB_TEST<interm, cpu> mfsgd_test(mtWDataPtr, mtHDataPtr, testWPos, testHPos, testV, dim_r, testRMSE.get(), mutex_w.get(), mutex_h.get(), Avx_explicit);

    /* Test MF-SGD before iteration */
    if (tbb_grainsize != 0)
    {
        /* use explicitly specified grainsize */
        parallel_for(blocked_range<int>(0, dim_test, tbb_grainsize), mfsgd_test);
    }
    else
    {
        /* use auto-partitioner by TBB */
        /* parallel_for(blocked_range<int>(0, dim_test), mfsgd_test, auto_partitioner()); */

        /* use affinity-partitioner by TBB */
        parallel_for(blocked_range<int>(0, dim_test), mfsgd_test, ap);

    }

    totalRMSE = 0;
    interm* testRMSE_ptr = testRMSE.get();

    for(int k=0;k<dim_test;k++)
        totalRMSE += testRMSE_ptr[k];

    totalRMSE = totalRMSE/(dim_test - absent_test_num);

    std::printf("RMSE before interation: %f\n", sqrt(totalRMSE));
    std::fflush(stdout);

    /* End of Test MF-SGD before iteration */
    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double train_time = 0;

    for(int j=0;j<iteration;j++)
    {
        mfsgd.setItr(j);
        clock_gettime(CLOCK_MONOTONIC, &ts1);

        /* training MF-SGD */
        if (tbb_grainsize != 0)
        {
            parallel_for(blocked_range<int>(0, dim_ratio, tbb_grainsize), mfsgd);
        }
        else
        {
            /* parallel_for(blocked_range<int>(0, dim_ratio), mfsgd, auto_partitioner()); */
            parallel_for(blocked_range<int>(0, dim_ratio), mfsgd, ap);
        }

        clock_gettime(CLOCK_MONOTONIC, &ts2);

        /* get the training time for each iteration */
        diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
	    /* train_time += (double)(diff)/1000000L; */
	    train_time += static_cast<double>((diff)/1000000L);

        /* Test MF-SGD */
        if (tbb_grainsize != 0)
        {
            parallel_for(blocked_range<int>(0, dim_test, tbb_grainsize), mfsgd_test);
        }
        else
        {
            /* parallel_for(blocked_range<int>(0, dim_test), mfsgd_test, auto_partitioner()); */
            parallel_for(blocked_range<int>(0, dim_test), mfsgd_test, ap);
        }

        totalRMSE = 0;
        for(int k=0;k<dim_test;k++)
            totalRMSE += testRMSE_ptr[k];

        totalRMSE = totalRMSE/(dim_test - absent_test_num);
        std::printf("RMSE after interation %d: %f, train time: %f\n", j, sqrt(totalRMSE), static_cast<double>(diff/1000000L));
        std::fflush(stdout);
    }

    init.terminate();

    std::printf("Average training time per iteration: %f, total time: %f\n", train_time/iteration, train_time);
    std::fflush(stdout);

    return;

}/*}}}*/


} // namespace daal::internal
}
}
} // namespace daal

#endif

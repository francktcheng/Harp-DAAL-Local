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
//  Implementation of mf_sgds
//--
*/

#ifndef __MF_SGD_KERNEL_BATCH_IMPL_I__
#define __MF_SGD_KERNEL_BATCH_IMPL_I__

#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "mf_sgd_default_impl.i"

#include "threading.h"
#include "task_scheduler_init.h"
#include "blocked_range.h"
#include "parallel_for.h"
#include "queuing_mutex.h"
#include <algorithm>
#include <math.h>       
#include <cstdlib> 
#include <iostream>
#include <time.h>

using namespace tbb;
using namespace daal::internal;
using namespace daal::services::internal;

// CPU intrinsics for Intel Compiler only
#if defined (__INTEL_COMPILER) && defined(__linux__) && defined(__x86_64__)
    #include <immintrin.h>
#endif

typedef queuing_mutex currentMutex_t;

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
namespace internal
{

template<typename interm, CpuType cpu>
void updateMF(interm *WMat,interm *HMat, interm* workV, int* seq, int idx, const long dim_r, const interm rate, const interm lambda);

template<typename interm, CpuType cpu>
void updateMF_explicit512(interm *WMat,interm *HMat, interm* workV, int* seq, int idx, const long dim_r, const interm rate, const interm lambda);

template<typename interm, CpuType cpu>
void computeRMSE(interm *WMat,interm *HMat, interm* testV, interm* testRMSE, int idx, const long dim_r);

template<typename interm, CpuType cpu>
void computeRMSE_explicit512(interm *WMat,interm *HMat, interm* testV, interm* testRMSE, int idx, const long dim_r);

/**
 *  \brief Kernel for mf_sgd mf_sgd calculation
 */
template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDBatchKernel<interm, method, cpu>::compute(NumericTable** TrainSet,NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par)
{
    MF_SGDBatchKernel<interm, method, cpu>::compute_thr(TrainSet, TestSet, r, par);
}


/* Max number of blocks depending on arch */
#if( __CPUID__(DAAL_CPU) >= __avx512_mic__ )
    #define DEF_MAX_BLOCKS 256
#else
    #define DEF_MAX_BLOCKS 128
#endif

/*
    Algorithm for parallel mf_sgd computation:
    -------------------------------------
    
*/
template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDBatchKernel<interm, method, cpu>::compute_thr(NumericTable** TrainSet, NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par)
{
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

    const int dim_train = TrainSet[0]->getNumberOfRows();
    const int dim_test = TestSet[0]->getNumberOfRows();

    /* ------------- Retrieve Training Data Set -------------*/

    FeatureMicroTable<int, readOnly, cpu> workflowW_ptr(TrainSet[0]);
    FeatureMicroTable<int, readOnly, cpu> workflowH_ptr(TrainSet[0]);
    FeatureMicroTable<interm, readOnly, cpu> workflow_ptr(TrainSet[0]);

    int *workWPos = 0;
    workflowW_ptr.getBlockOfColumnValues(0, 0, dim_train, &workWPos);

    int *workHPos = 0;
    workflowH_ptr.getBlockOfColumnValues(1, 0, dim_train, &workHPos);

    interm *workV;
    workflow_ptr.getBlockOfColumnValues(2,0,dim_train,&workV);

    /* debug */
    /*for( int j = 0; j< 20;j++)
    {
        std::cout<<"V: "<<j<<" wPos: "<<workWPos[j]<<" hPos: "<<workHPos[j]<<" val: "<<workV[j]<<std::endl;
    }*/

    /* ------------- Retrieve Test Data Set -------------*/

    FeatureMicroTable<int, readOnly, cpu> testW_ptr(TestSet[0]);
    FeatureMicroTable<int, readOnly, cpu> testH_ptr(TestSet[0]);
    FeatureMicroTable<interm, readOnly, cpu> test_ptr(TestSet[0]);

    int *testWPos = 0;
    testW_ptr.getBlockOfColumnValues(0, 0, dim_test, &testWPos);

    int *testHPos = 0;
    testH_ptr.getBlockOfColumnValues(1, 0, dim_test, &testHPos);

    interm *testV;
    test_ptr.getBlockOfColumnValues(2, 0, dim_test, &testV);

    /* debug */
    std::cout<<"Total test points: "<<dim_test<<std::endl;



    /* ---------------- Retrieve Model W ---------------- */
    BlockMicroTable<interm, readWrite, cpu> mtWDataTable(r[0]);

    /* debug */
    std::cout<<"model W row: "<<r[0]->getNumberOfRows()<<std::endl;
    std::cout<<"model W col: "<<r[0]->getNumberOfColumns()<<std::endl;

    /* interm** mtWDataPtr = new interm*[dim_w]; */
 
    /* for(int j = 0; j<dim_w;j++) */
    /* { */
    /*      mtWDataTable.getBlockOfRows(j, 1, &(mtWDataPtr[j])); */
    /* }   */

    interm* mtWDataPtr = 0;
    mtWDataTable.getBlockOfRows(0, dim_w, &mtWDataPtr);

    /* ---------------- Retrieve Model H ---------------- */
    BlockMicroTable<interm, readWrite, cpu> mtHDataTable(r[1]);

    // debug 
    // std::cout<<"model H row: "<<r[1]->getNumberOfRows()<<std::endl;
    // std::cout<<"model H col: "<<r[1]->getNumberOfColumns()<<std::endl;

    interm* mtHDataPtr = 0;
    mtHDataTable.getBlockOfRows(0, dim_h, &mtHDataPtr);

    /* create the mutex for WData and HData */
    currentMutex_t* mutex_w = new currentMutex_t[dim_w];
    currentMutex_t* mutex_h = new currentMutex_t[dim_h];

    /* RMSE value for test dataset after each iteration */
    interm* testRMSE = new interm[dim_test];

    interm totalRMSE = 0;

    /* ------------------- Starting TBB based Training  -------------------*/
    task_scheduler_init init(task_scheduler_init::deferred);

    if (thread_num != 0)
    {
        // use explicitly specified num of threads 
        // init = new task_scheduler_init(task_scheduler_init::deferred); 
        init.initialize(thread_num);
    }
    else
    {
        // use automatically generated threads by TBB 
        // init = new task_scheduler_init(task_scheduler_init::automatic); 

        init.initialize();
    }

    /* set up the sequence of workflow */
    int* seq = new int[dim_train];
    for(int j=0;j<dim_train;j++)
        seq[j] = j;


    MFSGDTBB<interm, cpu> mfsgd(mtWDataPtr, mtHDataPtr, workWPos, workHPos, workV, seq, dim_r, learningRate, lambda, mutex_w, mutex_h, Avx512_explicit);
    MFSGDTBB_TEST<interm, cpu> mfsgd_test(mtWDataPtr, mtHDataPtr, testWPos, testHPos, testV, dim_r, testRMSE, mutex_w, mutex_h, Avx512_explicit);

    /*---------------- Test MF-SGD before iteration ----------------*/


    if (tbb_grainsize != 0)
    {
        // use explicitly specified grainsize
        parallel_for(blocked_range<int>(0, dim_test, tbb_grainsize), mfsgd_test);
    }
    else
    {
        // use auto-partitioner by TBB
        parallel_for(blocked_range<int>(0, dim_test), mfsgd_test, auto_partitioner());
    }

    totalRMSE = 0;
    for(int k=0;k<dim_test;k++)
        totalRMSE += testRMSE[k];

    totalRMSE = totalRMSE/dim_test;

    printf("RMSE before interation: %f\n", sqrt(totalRMSE));

    /*---------------- End of Test MF-SGD before iteration ----------------*/
    int k, p;

    struct timespec ts1;
	struct timespec ts2;
    long diff;
    double train_time = 0;

    for(int j=0;j<iteration;j++)
    {

        clock_gettime(CLOCK_MONOTONIC, &ts1);

        /* training MF-SGD */
        if (tbb_grainsize != 0)
            parallel_for(blocked_range<int>(0, dim_train, tbb_grainsize), mfsgd);
        else
            parallel_for(blocked_range<int>(0, dim_train), mfsgd, auto_partitioner());

        clock_gettime(CLOCK_MONOTONIC, &ts2);

        /* get the training time for each iteration */
        diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
	    train_time += (double)(diff)/1000000L;

        /* Test MF-SGD */
        if (tbb_grainsize != 0)
            parallel_for(blocked_range<int>(0, dim_test, tbb_grainsize), mfsgd_test);
        else
            parallel_for(blocked_range<int>(0, dim_test), mfsgd_test, auto_partitioner());

        totalRMSE = 0;
        for(k=0;k<dim_test;k++)
            totalRMSE += testRMSE[k];

        totalRMSE = totalRMSE/dim_test;

        printf("RMSE after interation %d: %f, train time: %f\n", j, sqrt(totalRMSE), (double)(diff)/1000000L);

    }

    init.terminate();

    printf("Average training time per iteration: %f, total time: %f\n", train_time/iteration, train_time);

    /* ------------------- Finishing TBB based Training  -------------------*/

    delete[] mutex_w;
    delete[] mutex_h;

    delete[] seq;
    delete[] testRMSE;

    return;
}

template<typename interm, CpuType cpu>
MFSGDTBB<interm, cpu>::MFSGDTBB(
        interm* mtWDataTable,
        interm* mtHDataTable,
        int* workWPos,
        int* workHPos,
        interm *workV,
        int* seq,
        const long Dim,
        const interm learningRate,
        const interm lambda,
        currentMutex_t* mutex_w,
        currentMutex_t* mutex_h,
        const int Avx512_explicit

)
{
    _mtWDataTable = mtWDataTable;
    _mtHDataTable = mtHDataTable;
    
    _workWPos = workWPos;
    _workHPos = workHPos;

    _workV = workV;
    _seq = seq;
    _Dim = Dim;
    _learningRate = learningRate;
    _lambda = lambda;

    _mutex_w = mutex_w;
    _mutex_h = mutex_h;
    _Avx512_explicit = Avx512_explicit;
}

template<typename interm, CpuType cpu>
void MFSGDTBB<interm, cpu>::operator()( const blocked_range<int>& range ) const 
{

    interm *WMat = 0;
    interm *HMat = 0;

    interm Mult = 0;
    interm Err = 0;
    interm WMatVal = 0;
    interm HMatVal = 0;

    // using local variables
    interm* mtWDataTable = _mtWDataTable;
    interm* mtHDataTable = _mtHDataTable;

    int* workWPos = _workWPos;
    int* workHPos = _workHPos;
    int* seq = _seq;
    interm* workV = _workV;

    long Dim = _Dim;
    interm learningRate = _learningRate;
    interm lambda = _lambda;

    for( int i=range.begin(); i!=range.end(); ++i )
    {

        // using local variables
        WMat = mtWDataTable + workWPos[seq[i]]*Dim;
        HMat = mtHDataTable + workHPos[seq[i]]*Dim;

        /* currentMutex_t::scoped_lock lock_w(_mutex_w[_workWPos[_seq[i]]]); */
        /* currentMutex_t::scoped_lock lock_h(_mutex_h[_workHPos[_seq[i]]]); */
        
        if (_Avx512_explicit == 1)
            updateMF_explicit512<interm, cpu>(WMat, HMat, workV, seq, i, Dim, learningRate, lambda);
        else
            updateMF<interm, cpu>(WMat, HMat, workV, seq, i, Dim, learningRate, lambda);

        /* lock_w.release(); */
        /* lock_h.release(); */

    }

}

template<typename interm, CpuType cpu>
MFSGDTBB_TEST<interm, cpu>::MFSGDTBB_TEST(
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

)
{
    _mtWDataTable = mtWDataTable;
    _mtHDataTable = mtHDataTable;
    
    _testWPos = testWPos;
    _testHPos = testHPos;

    _testV = testV;
    _Dim = Dim;

    _testRMSE = testRMSE;

    _mutex_w = mutex_w;
    _mutex_h = mutex_h;

    _Avx512_explicit = Avx512_explicit;

}

template<typename interm, CpuType cpu>
void MFSGDTBB_TEST<interm, cpu>::operator()( const blocked_range<int>& range ) const 
{

    interm *WMat = 0;
    interm *HMat = 0;

    interm* mtWDataTable = _mtWDataTable;
    interm* mtHDataTable = _mtHDataTable;

    interm* testV = _testV;

    int* testWPos = _testWPos;
    int* testHPos = _testHPos;

    long Dim = _Dim;
    interm* testRMSE = _testRMSE;


    for( int i=range.begin(); i!=range.end(); ++i )
    {

        if (testWPos[i] != -1 && testHPos[i] != -1)
        {

            WMat = mtWDataTable + testWPos[i]*Dim;
            HMat = mtHDataTable + testHPos[i]*Dim;

            /* currentMutex_t::scoped_lock lock_w(_mutex_w[_testWPos[i]]); */
            /* currentMutex_t::scoped_lock lock_h(_mutex_h[_testHPos[i]]); */

            if (_Avx512_explicit == 1)
                computeRMSE_explicit512<interm, cpu>(WMat, HMat, testV, testRMSE, i, Dim);
            else
                computeRMSE<interm, cpu>(WMat, HMat, testV, testRMSE, i, Dim);

            /* lock_w.release(); */
            /* lock_h.release(); */

        }
        else
            testRMSE[i] = 0;

    }

}

/**
 * @brief compiler based vectorization according to different CpuType
 *
 * @tparam interm
 * @tparam cpu
 * @param WMat
 * @param HMat
 * @param workV
 * @param seq
 * @param idx
 * @param dim_r
 * @param rate
 * @param lambda
 */
template<typename interm, CpuType cpu>
void updateMF(interm *WMat,interm *HMat, interm* workV, int* seq, int idx, const long dim_r, const interm rate, const interm lambda)
{/*{{{*/

    interm Mult = 0;
    interm Err = 0;
    interm WMatVal = 0;
    interm HMatVal = 0;

    for(int p = 0; p<dim_r; p++)
        Mult += (WMat[p]*HMat[p]);

    Err = workV[seq[idx]] - Mult;

    for(int p = 0;p<dim_r;p++)
    {
        WMatVal = WMat[p];
        HMatVal = HMat[p];

        WMat[p] = WMat[p] + rate*(Err*HMatVal - lambda*WMatVal);
        HMat[p] = HMat[p] + rate*(Err*WMatVal - lambda*HMatVal);

    }

}/*}}}*/


/**
 * @brief if CpuType is avx512_mic, use explicit AVX512 instructions, 
 * otherwise use compiler based vectorization
 *
 * @tparam interm
 * @tparam cpu
 * @param WMat
 * @param HMat
 * @param workV
 * @param seq
 * @param idx
 * @param dim_r
 * @param rate
 * @param lambda
 */
template<typename interm, CpuType cpu>
void updateMF_explicit512(interm *WMat,interm *HMat, interm* workV, int* seq, int idx, const long dim_r, const interm rate, const interm lambda)
{/*{{{*/

    interm Mult = 0;
    interm Err = 0;
    interm WMatVal = 0;
    interm HMatVal = 0;

    for(int p = 0; p<dim_r; p++)
        Mult += (WMat[p]*HMat[p]);

    Err = workV[seq[idx]] - Mult;

    for(int p = 0;p<dim_r;p++)
    {
        WMatVal = WMat[p];
        HMatVal = HMat[p];

        WMat[p] = WMat[p] + rate*(Err*HMatVal - lambda*WMatVal);
        HMat[p] = HMat[p] + rate*(Err*WMatVal - lambda*HMatVal);

    }

}/*}}}*/

/**
 * @brief compiler based vectorization according to different CpuType
 *
 * @tparam interm
 * @tparam cpu
 * @param WMat
 * @param HMat
 * @param testV
 * @param testRMSE
 * @param idx
 * @param dim_r
 */
template<typename interm, CpuType cpu>
void computeRMSE(interm *WMat,interm *HMat, interm* testV, interm* testRMSE, int idx, const long dim_r)
{/*{{{*/
    int p;
    interm Mult = 0;
    interm Err;

    for(p = 0; p<dim_r; p++)
        Mult += (WMat[p]*HMat[p]);

    Err = testV[idx] - Mult;

    testRMSE[idx] = Err*Err;

}/*}}}*/

/**
 * @brief if CpuType is avx512_mic, use explicit AVX512 instructions, 
 * otherwise use compiler based vectorization
 *
 * @tparam interm
 * @tparam cpu
 * @param WMat
 * @param HMat
 * @param testV
 * @param testRMSE
 * @param idx
 * @param dim_r
 */
template<typename interm, CpuType cpu>
void computeRMSE_explicit512(interm *WMat,interm *HMat, interm* testV, interm* testRMSE, int idx, const long dim_r)
{/*{{{*/
    int p;
    interm Mult = 0;
    interm Err;

    for(p = 0; p<dim_r; p++)
        Mult += (WMat[p]*HMat[p]);

    Err = testV[idx] - Mult;

    testRMSE[idx] = Err*Err;

}/*}}}*/

// AVX512-MIC optimization via template specialization (Intel compiler only)
#if defined (__INTEL_COMPILER) && defined(__linux__) && defined(__x86_64__) && ( __CPUID__(DAAL_CPU) == __avx512_mic__ )
    #include "mf_sgd_default_batch_impl_avx512_mic.i"
#endif

} // namespace daal::internal
}
}
} // namespace daal

#endif

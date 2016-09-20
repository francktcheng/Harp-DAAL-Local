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

#include <cstdlib> 
#include <iostream>

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
    /* to be implemented */
    /* retrieve members of parameter */
    const Parameter *parameter = static_cast<const Parameter *>(par);
    const long dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    const double learningRate = parameter->_learningRate;
    const double lambda = parameter->_lambda;
    const int iteration = parameter->_iteration;
    const int thread_num = parameter->_thread_num;

    const int dim_train = TrainSet[0]->getNumberOfRows();
    const int dim_test = TestSet[0]->getNumberOfRows();

    /* ------------- Retrieve Training Data Set -------------*/

    /* size_t readFeatureIdx = 2; */

    /* BlockDescriptor<int> intBlock; */
    /* dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nObservations, readOnly, intBlock); */

    FeatureMicroTable<interm, readOnly, cpu> workflow_ptr(TrainSet[0]);
    FeatureMicroTable<int, readOnly, cpu> workflowW_ptr(TrainSet[0]);
    FeatureMicroTable<int, readOnly, cpu> workflowH_ptr(TrainSet[0]);

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

    /* ---------------- Retrieve Model W ---------------- */
    BlockMicroTable<interm, readWrite, cpu> mtWDataTable(r[0]);
    interm** mtWDataPtr = new interm*[dim_w];
 
    for(int j = 0; j<dim_w;j++)
    {
         mtWDataTable.getBlockOfRows(j, 1, &(mtWDataPtr[j]));
    }  

    /* ---------------- Retrieve Model H ---------------- */
    BlockMicroTable<interm, readWrite, cpu> mtHDataTable(r[1]);
    interm** mtHDataPtr = new interm*[dim_h];

    for(int j = 0; j<dim_h;j++)
    {
         mtHDataTable.getBlockOfRows(j, 1, &(mtHDataPtr[j]));
    }


    /* create the mutex for WData and HData */
    currentMutex_t* mutex_w = new currentMutex_t[dim_w];
    currentMutex_t* mutex_h = new currentMutex_t[dim_h];


    /* ------------------- Starting TBB based Training  -------------------*/
    /* task_scheduler_init init; */
    task_scheduler_init init(task_scheduler_init::deferred); 
    init.initialize(thread_num);
    
    /* set up the sequence of workflow */
    int* seq = new int[dim_train];
    for(int j=0;j<dim_train;j++)
        seq[j] = j;

    MFSGDTBB<interm, cpu> mfsgd(mtWDataPtr, mtHDataPtr, workWPos, workHPos, workV, seq, dim_r, learningRate, lambda, mutex_w, mutex_h);

    for(int j=0;j<iteration;j++)
    {

        /* shuffle the sequence */
        std::random_shuffle(&seq[0], &seq[dim_train-1]); 
        parallel_for(blocked_range<int>(0, dim_train, 100), mfsgd);

    }

    init.terminate();

    /* ------------------- Finishing TBB based Training  -------------------*/


    /* ------------------- Starting TBB based Testing  -------------------*/

    /* ------------------- Finishing TBB based Testing  -------------------*/

    delete[] mutex_w;
    delete[] mutex_h;

    delete[] mtWDataPtr;
    delete[] mtHDataPtr;

    delete[] seq;

    return;
}

template<typename interm, CpuType cpu>
MFSGDTBB<interm, cpu>::MFSGDTBB(
        interm** mtWDataTable,
        interm** mtHDataTable,
        int* workWPos,
        int* workHPos,
        interm *workV,
        int* seq,
        const long Dim,
        const interm learningRate,
        const interm lambda,
        currentMutex_t* mutex_w,
        currentMutex_t* mutex_h

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
}

template<typename interm, CpuType cpu>
void MFSGDTBB<interm, cpu>::operator()( const blocked_range<int>& range ) const 
{

    for( int i=range.begin(); i!=range.end(); ++i )
    {

        interm *WMat = 0;
        interm *HMat = 0;

        interm Mult = 0;
        interm Err = 0;
        interm WMatVal = 0;
        interm HMatVal = 0;

        WMat = _mtWDataTable[_workWPos[_seq[i]]];
        HMat = _mtHDataTable[_workHPos[_seq[i]]];

        currentMutex_t::scoped_lock lock_w(_mutex_w[_workWPos[_seq[i]]]);
        currentMutex_t::scoped_lock lock_h(_mutex_h[_workHPos[_seq[i]]]);


        for(int p = 0; p<_Dim; p++)
            Mult += (WMat[p]*HMat[p]);

        Err = _workV[_seq[i]] - Mult;

        for(int p = 0;p<_Dim;p++)
        {
            WMatVal = WMat[p];
            HMatVal = HMat[p];

            WMat[p] = WMat[p] + _learningRate*(Err*HMatVal + _lambda*WMatVal);
            HMat[p] = HMat[p] + _learningRate*(Err*WMatVal + _lambda*HMatVal);

        }

        lock_w.release();
        lock_h.release();

    }

}


} // namespace daal::internal
}
}
} // namespace daal

#endif

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

#include "task_scheduler_init.h"
#include "blocked_range.h"
#include "parallel_for.h"
#include "queuing_mutex.h"
#include "numeric_table.h"
#include "kernel.h"
#include <cstdlib> 
#include <cstdio> 

#include "mf_sgd_batch.h"
#include "tbb/tick_count.h"

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

/**
 * @brief computation kernel for mf_sgd batch mode 
 *
 * @tparam interm
 * @tparam method
 * @tparam cpu
 */
template<typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
class MF_SGDBatchKernel : public Kernel
{
public:

    /**
     * @brief compute and update W, H model by Training data
     *
     * @param[in] TrainSet  train dataset stored in an AOSNumericTable
     * @param[in] TestSet  test dataset stored in an AOSNumericTable
     * @param[in,out] r[] model W and H
     * @param[in] par
     */
    void compute(const NumericTable** TrainSet, const NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par);

    /**
     * @brief regroup the train dataset points by their row id
     *
     * @param trainWPos
     * @param trainHPos
     * @param trainV
     * @param train_num
     * @param parameter
     */
    void reorder(int* trainWPos, int* trainHPos, interm* trainV, const int train_num, const Parameter *parameter);

    /* a multi-threading version of compute implemented by TBB */
    void compute_thr(int* trainWPos, int* trainHPos, interm* trainV, const int train_num,
                     int* testWPos, int* testHPos, interm* testV, const int test_num,
                     interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter);

    /* a multi-threading version of compute implemented by OpenMP */
    void compute_openmp(int* trainWPos, int* trainHPos, interm* trainV, const int train_num,
                        int* testWPos, int* testHPos, interm* testV, const int test_num,
                        interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter);

    /* a multi-threading version of compute implemented by TBB with reordered training dataset points */
    void compute_thr_reordered(int* testWPos, int* testHPos, interm* testV, const int test_num,
                               interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter);

    /* a multi-threading version of compute implemented by OpenMP with reordered training dataset points */
    void compute_openmp_reordered(int* testWPos, int* testHPos, interm* testV, const int test_num,
                               interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter);

    void free_reordered();

private:

    services::SharedPtr<std::vector<int> > _trainWQueue;
    services::SharedPtr<std::vector<int> > _trainLQueue;
    services::SharedPtr<std::vector<int*> > _trainHQueue;
    services::SharedPtr<std::vector<interm*> > _trainVQueue;

};

/**
 * @brief computation kernel for mf_sgd distributed mode
 *
 * @tparam interm
 * @tparam method
 * @tparam cpu
 */
template<typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
class MF_SGDDistriKernel : public Kernel
{
public:

    /**
     * @brief compute and update W, H model by Training data
     *
     * @param[in] WPos row id of point in W model, stored in HomogenNumericTable 
     * @param[in] HPos col id of point in H model, stored in HomogenNumericTable
     * @param[in] Val  value of point, stored in HomogenNumericTable
     * @param[in,out] r[] model W and H
     * @param[in] par
     */
    // void compute(const NumericTable** WPos, const NumericTable** HPos, const NumericTable** Val, NumericTable** WPosTest, NumericTable** HPosTest, NumericTable** ValTest, 
    //         NumericTable *r[], const daal::algorithms::Parameter *par);

    void compute(NumericTable** WPos, const NumericTable** HPos, const NumericTable** Val, NumericTable** WPosTest, NumericTable** HPosTest, NumericTable** ValTest, 
            NumericTable *r[], Parameter *par, int* col_ids, interm** hMat_native_mem);

    /* a multi-threading version of training process implemented by TBB */
    void compute_train(int* workWPos, int* workHPos, interm* workV, const int dim_set, interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter);

    /* a multi-threading version of training process implemented by OpenMP */
    void compute_train_omp(int* workWPos, int* workHPos, interm* workV, const int dim_set, interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter);

    /* another multi-threading version of training process implemented by OpenMP */
    void compute_train2_omp(int* workWPos, int* workHPos, interm* workV, const int dim_set, interm* mtWDataPtr, interm* mtHDataPtr, Parameter *parameter, int* col_ids, interm** hMat_native_mem);

    /* a multi-threading version of testing process implemented by TBB */
    void compute_test(int* workWPos, int* workHPos, interm* workV, const int dim_set, interm* mtWDataPtr, interm* mtHDataPtr, interm* mtRMSEPtr,const Parameter *parameter);

    /* a multi-threading version of testing process implemented by OpenMP */
    void compute_test_omp(int* workWPos, int* workHPos, interm* workV, const int dim_set, interm* mtWDataPtr, interm* mtHDataPtr, interm* mtRMSEPtr,const Parameter *parameter);

    /* another multi-threading version of testing process implemented by OpenMP */
    void compute_test2_omp(int* workWPos, int* workHPos, interm* workV, const int dim_set, interm* mtWDataPtr, interm* mtHDataPtr, interm* mtRMSEPtr, Parameter *parameter, int* col_ids, interm** hMat_native_mem);

};

/**
 * @brief A TBB kernel for computing MF-SGD
 *
 * @tparam interm
 * @tparam cpu
 */
template<typename interm, CpuType cpu>
struct MFSGDTBB
{
   /* default constructor */ 
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
            const int Avx_explicit,
            const int step,
            const int dim_train
            );

	

	/**
	 * @brief operator used by parallel_for template
	 *
	 * @param[in] range range of parallel block to execute by a thread
	 */
    void operator()( const blocked_range<int>& range ) const; 

	
	/**
	 * @brief set up the id of iteration
	 * used in distributed mode
	 *
	 * @param itr
	 */
    void setItr(int itr) { _itr = itr;}
    
    void setTimeStart(tbb::tick_count timeStart) {_timeStart = timeStart;}

    void setTimeOut(double timeOut) {_timeOut = timeOut;}

    void setOrder(int* order) {_order = order;}

    interm* _mtWDataTable;  /* model W */
    interm* _mtHDataTable;  /* model H */

    int* _workWPos;         /* row id of point in W */
    int* _workHPos;		    /* col id of point in H */
    interm* _workV;         /* value of point */

    long _Dim;              /* dimension of vector in model W and H */
    interm _learningRate;
    interm _lambda;

    int _Avx_explicit;   /* 1 if use explicit avx intrincis 0 if use compiler vectorization */

    int _step;              /* stride of tasks if only part of tasks are executed */
    int _dim_train;         /* total number of tasks */
    int _itr;               /* iteration id  */
    tbb::tick_count _timeStart = tbb::tick_count::now();
    double _timeOut = 0;

    currentMutex_t* _mutex_w;
    currentMutex_t* _mutex_h;
    int* _order = NULL;

};

template<typename interm, CpuType cpu>
struct MFSGDTBBREORDER
{
   /* default constructor */ 
    MFSGDTBBREORDER(
            interm* mtWDataTable,
            interm* mtHDataTable,
            int* queueWPos,
            int* queueLength,
            int** queueHPos,
            interm** queueVVal,
            const long Dim,
            const interm learningRate,
            const interm lambda,
            currentMutex_t* mutex_w,
            currentMutex_t* mutex_h,
            const int Avx_explicit
            // const int step,
            // const int dim_train
            );

	/**
	 * @brief operator used by parallel_for template
	 *
	 * @param[in] range range of parallel block to execute by a thread
	 */
    void operator()( const blocked_range<int>& range ) const; 

	
	/**
	 * @brief set up the id of iteration
	 * used in distributed mode
	 *
	 * @param itr
	 */
    // void setItr(int itr) { _itr = itr;}
    
    // void setTimeStart(tbb::tick_count timeStart) {_timeStart = timeStart;}

    // void setTimeOut(double timeOut) {_timeOut = timeOut;}

    interm* _mtWDataTable;  /* model W */
    interm* _mtHDataTable;  /* model H */

    int* _queueWPos;        /* row id of W for each queue */
    int* _queueLength;		/* num of cols for each row */
    int** _queueHPos;       /* col ids of H for each queue */
    interm** _queueVVal;    /* value of points for each queue */

    long _Dim;              /* dimension of vector in model W and H */
    interm _learningRate;
    interm _lambda;

    int _Avx_explicit;   /* 1 if use explicit avx intrincis 0 if use compiler vectorization */

    // int _step;              /* stride of tasks if only part of tasks are executed */
    // int _dim_train;         /* total number of tasks */
    // int _itr;               /* iteration id  */
    // tbb::tick_count _timeStart = tbb::tick_count::now();
    // double _timeOut = 0;

    currentMutex_t* _mutex_w;
    currentMutex_t* _mutex_h;

};

/* MF_SGD_Test kernel implemented by TBB */
template<typename interm, CpuType cpu>
struct MFSGDTBB_TEST
{
    
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
            const int Avx_explicit
    );

    void operator()( const blocked_range<int>& range ) const; 

	interm* _mtWDataTable;   /* model W */
    interm* _mtHDataTable;   /* model H */

    int* _testWPos;          /* row id of point in W */
    int* _testHPos;          /* col id of point in H */
    interm* _testV;          /* value of point  */

    int _Dim;                /* dimension of model data */
    interm* _testRMSE;       /* RMSE value calculated for each training point */
    int _Avx_explicit;    /* 1 to use explicit avx intrinsics or 0 not */

    currentMutex_t* _mutex_w;
    currentMutex_t* _mutex_h;


};

//function used in pthread to copy data from javanumrictable 
template <typename interm>
struct SOADataCopy
{
    SOADataCopy(
            NumericTable* SOA_Table,
            int start_pos,
            int len,
            int nDim,
            BlockDescriptor<interm>** Descriptor,
            interm** nativeMem
            ): _SOA_Table(SOA_Table), _start_pos(start_pos), _len(len), _nDim(nDim), _Descriptor(Descriptor), _nativeMem(nativeMem) {}

    NumericTable* _SOA_Table;
    int _start_pos;
    int _len;
    int _nDim;
    BlockDescriptor<interm>** _Descriptor;
    interm** _nativeMem;

};

template <typename interm>
void* SOACopyBulkData(void* arg)
{
    internal::SOADataCopy<interm>* copyElem = static_cast<internal::SOADataCopy<interm>* >(arg);
    (copyElem->_SOA_Table)->getBlockOfColumnValuesBM(copyElem->_start_pos, copyElem->_len, 0, copyElem->_nDim, writeOnly, copyElem->_Descriptor);

    //assign ptr of blockDescriptors to nativeMem
    for(int k=0;k<copyElem->_len;k++)
    {
        int feature_idx = copyElem->_start_pos + k;
        (copyElem->_nativeMem)[feature_idx] = (copyElem->_Descriptor)[feature_idx]->getBlockPtr();

    }

    return NULL;
}

template <typename interm>
void* SOAReleaseBulkData(void* arg)
{
    internal::SOADataCopy<interm>* copyElem = static_cast<internal::SOADataCopy<interm>* >(arg);
    (copyElem->_SOA_Table)->releaseBlockOfColumnValuesBM(copyElem->_start_pos, copyElem->_len, copyElem->_Descriptor);

    //assign ptr of blockDescriptors to nativeMem
    // for(int k=0;k<copyElem->_len;k++)
    // {
    //     int feature_idx = copyElem->_start_pos + k;
    //     (copyElem->_nativeMem)[feature_idx] = (copyElem->_Descriptor)[feature_idx]->getBlockPtr();
    //
    // }

    return NULL;
}

} // namespace daal::internal
}
}
} // namespace daal

#endif

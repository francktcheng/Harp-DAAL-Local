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

#include "mf_sgd_batch.h"

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

    /* a multi-threading version of compute implemented by TBB */
    void compute_thr(const NumericTable** TrainSet,const NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par);

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
     * @param[in] TrainWPos row id of training point in W model, stored in HomogenNumericTable 
     * @param[in] TrainHPos col id of training point in H model, stored in HomogenNumericTable
     * @param[in] TrainVal  value of training point, stored in HomogenNumericTable
     * @param[in,out] r[] model W and H
     * @param[in] par
     */
    void compute(const NumericTable** TrainWPos, const NumericTable** TrainHPos, const NumericTable** TrainVal, NumericTable *r[], const daal::algorithms::Parameter *par);

    /* a multi-threading version of compute implemented by TBB */
    void compute_thr(const NumericTable** TrainWPos, const NumericTable** TrainHPos, const NumericTable** TrainVal, NumericTable *r[], const daal::algorithms::Parameter *par);

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
} // namespace daal::internal
}
}
} // namespace daal

#endif

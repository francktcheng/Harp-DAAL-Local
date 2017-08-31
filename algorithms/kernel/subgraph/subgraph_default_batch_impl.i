/* file: subgraph_dense_default_batch_impl.i */
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
//  Implementation of subgraphs batch mode
//--
*/

#ifndef __SUBGRAPH_KERNEL_BATCH_IMPL_I__
#define __SUBGRAPH_KERNEL_BATCH_IMPL_I__

#include <time.h>
#include <math.h>       
#include <algorithm>
#include <cstdlib> 
#include <cstdio> 
#include <iostream>
#include <vector>
#include <omp.h>
#include <immintrin.h>

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

#include "subgraph_default_impl.i"

using namespace tbb;
using namespace daal::internal;
using namespace daal::services::internal;

typedef queuing_mutex currentMutex_t;

// implementation of kernles within container
// kernels are defined in file: subgraph_default_kernel.h
namespace daal
{
namespace algorithms
{
namespace subgraph
{
namespace internal
{

template <typename interm, daal::algorithms::subgraph::Method method, CpuType cpu>
daal::services::interface1::Status subgraphBatchKernel<interm, method, cpu>::compute(const NumericTable** TrainSet, const NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par)
{/*{{{*/

    services::Status status;
    // /* retrieve members of parameter */
    // const Parameter *parameter = static_cast<const Parameter *>(par);
    // const int64_t dim_w = parameter->_Dim_w;
    // const int64_t dim_h = parameter->_Dim_h;
    //
    // const int dim_train = TrainSet[0]->getNumberOfRows();
    // const int dim_test = TestSet[0]->getNumberOfRows();
    //
    // /*  Retrieve Training Data Set */
    // FeatureMicroTable<int, readOnly, cpu> workflowW_ptr(TrainSet[0]);
    // FeatureMicroTable<int, readOnly, cpu> workflowH_ptr(TrainSet[0]);
    // FeatureMicroTable<interm, readOnly, cpu> workflow_ptr(TrainSet[0]);
    //
    // int *workWPos = 0;
    // workflowW_ptr.getBlockOfColumnValues(0, 0, dim_train, &workWPos);
    //
    // int *workHPos = 0;
    // workflowH_ptr.getBlockOfColumnValues(1, 0, dim_train, &workHPos);
    //
    // interm *workV;
    // workflow_ptr.getBlockOfColumnValues(2,0,dim_train,&workV);
    //
    // /*  Retrieve Test Data Set */
    // FeatureMicroTable<int, readOnly, cpu> testW_ptr(TestSet[0]);
    // FeatureMicroTable<int, readOnly, cpu> testH_ptr(TestSet[0]);
    // FeatureMicroTable<interm, readOnly, cpu> test_ptr(TestSet[0]);
    //
    // int *testWPos = 0;
    // testW_ptr.getBlockOfColumnValues(0, 0, dim_test, &testWPos);
    //
    // int *testHPos = 0;
    // testH_ptr.getBlockOfColumnValues(1, 0, dim_test, &testHPos);
    //
    // interm *testV;
    // test_ptr.getBlockOfColumnValues(2, 0, dim_test, &testV);
    //
    // /*  Retrieve Model W  */
    // BlockMicroTable<interm, readWrite, cpu> mtWDataTable(r[0]);
    //
    // /* screen print out the size of model data */
    // std::printf("model W row: %zu\n",r[0]->getNumberOfRows());
    // std::fflush(stdout);
    // std::printf("model W col: %zu\n",r[0]->getNumberOfColumns());
    // std::fflush(stdout);
    //
    // interm* mtWDataPtr = 0;
    // mtWDataTable.getBlockOfRows(0, dim_w, &mtWDataPtr);
    //
    // /*  Retrieve Model H  */
    // BlockMicroTable<interm, readWrite, cpu> mtHDataTable(r[1]);
    //
    // interm* mtHDataPtr = 0;
    // mtHDataTable.getBlockOfRows(0, dim_h, &mtHDataPtr);
    //
    // if (parameter->_isReorder == 1 )
    // {
    //     /* TBB re-order */
    //     subgraphBatchKernel<interm, method, cpu>::reorder(workWPos, workHPos, workV, dim_train, parameter);
    //     subgraphBatchKernel<interm, method, cpu>::compute_thr_reordered(testWPos, testHPos, testV, dim_test, mtWDataPtr, mtHDataPtr, parameter);
    //     subgraphBatchKernel<interm, method, cpu>::free_reordered();
    // }
    // else if (parameter->_isReorder == 2 )
    // {
    //     /* OpenMP re-order */
    //     subgraphBatchKernel<interm, method, cpu>::reorder(workWPos, workHPos, workV, dim_train, parameter);
    //     subgraphBatchKernel<interm, method, cpu>::compute_openmp_reordered(testWPos, testHPos, testV, dim_test, mtWDataPtr, mtHDataPtr, parameter);
    //     subgraphBatchKernel<interm, method, cpu>::free_reordered();
    // }
    // else if (parameter->_isReorder == 3 )
    // {
    //     /* OpenMP no-reorder */
    //     subgraphBatchKernel<interm, method, cpu>::compute_openmp(workWPos, workHPos, workV, dim_train, testWPos, testHPos, testV, dim_test, mtWDataPtr, mtHDataPtr, parameter);
    // }
    // else 
    // {
    //     /* TBB no-reorder */
    //     subgraphBatchKernel<interm, method, cpu>::compute_thr(workWPos, workHPos, workV, dim_train, testWPos, testHPos, testV, dim_test, mtWDataPtr, mtHDataPtr, parameter);
    // }

    return status;

}/*}}}*/

} // namespace daal::internal
}
}
} // namespace daal

#endif

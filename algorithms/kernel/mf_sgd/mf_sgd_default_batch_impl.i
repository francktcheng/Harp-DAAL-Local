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

using namespace daal::internal;
using namespace daal::services::internal;

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
    /* NumericTable *ntAi = const_cast<NumericTable *>(a[0]); */

    /* size_t  n = ntAi->getNumberOfColumns(); */
    /* size_t  m = ntAi->getNumberOfRows(); */

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
void MF_SGDBatchKernel<interm, method, cpu>::compute_thr(NumericTable** TrainSet,NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par)
{
    /* to be implemented */

    return;
}


} // namespace daal::internal
}
}
} // namespace daal

#endif

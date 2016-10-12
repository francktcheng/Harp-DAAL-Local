/* file: mf_sgd_default_ksnc_impl.i */
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
//  Implementation of knl snc mode mf_sgd method
//--
*/

#ifndef __MF_SGD_KERNEL_KSNC_IMPL_I__
#define __MF_SGD_KERNEL_KSNC_IMPL_I__

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

// template<typename interm, CpuType cpu>
// void updateMF(interm *WMat,interm *HMat, interm* workV, int* seq, int idx, const long dim_r, const interm rate, const interm lambda);
//
// template<typename interm, CpuType cpu>
// void updateMF_explicit512(interm *WMat,interm *HMat, interm* workV, int* seq, int idx, const long dim_r, const interm rate, const interm lambda);
//
// template<typename interm, CpuType cpu>
// void computeRMSE(interm *WMat,interm *HMat, interm* testV, interm* testRMSE, int idx, const long dim_r);
//
// template<typename interm, CpuType cpu>
// void computeRMSE_explicit512(interm *WMat,interm *HMat, interm* testV, interm* testRMSE, int idx, const long dim_r);

/**
 *  \brief Kernel for mf_sgd mf_sgd calculation
 */



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

















} // namespace daal::internal
}
}
} // namespace daal

#endif

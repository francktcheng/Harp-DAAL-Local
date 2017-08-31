/* file: subgraph_default_distri_impl.i */
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
//  Implementation of distributed mode subgraph method
//--
*/

#ifndef __SUBGRAPH_KERNEL_DISTRI_IMPL_I__
#define __SUBGRAPH_KERNEL_DISTRI_IMPL_I__

#include <time.h>
#include <math.h>       
#include <algorithm>
#include <cstdlib> 
#include <vector>
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
#include "tbb/concurrent_hash_map.h"
#include "tbb/concurrent_vector.h"

#include "blocked_range.h"
#include "parallel_for.h"
#include "queuing_mutex.h"

#include "subgraph_default_impl.i"

using namespace tbb;
using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace subgraph
{
namespace internal
{
    

template <typename interm, daal::algorithms::subgraph::Method method, CpuType cpu>
daal::services::interface1::Status subgraphDistriKernel<interm, method, cpu>::compute(NumericTable** WPos, 
                                                      NumericTable** HPos, 
                                                      NumericTable** Val, 
                                                      NumericTable** WPosTest,
                                                      NumericTable** HPosTest, 
                                                      NumericTable** ValTest, 
                                                      NumericTable *r[], Parameter* &parameter, int* &col_ids,
                                                      interm** &hMat_native_mem)
{

    services::Status status;
    return status;
}



} // namespace daal::internal
}
}
} // namespace daal

#endif

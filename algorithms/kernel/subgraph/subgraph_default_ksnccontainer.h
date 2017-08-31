/* file: subgraph_dense_ksnccontainer.h */
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
//  Implementation of subgraph calculation algorithm container for SNC-4 mode of KNL
//  TODO
//--
*/
#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <math.h>       
#include <random>

#include "numeric_table.h"
#include "service_rng.h"

#include "subgraph_types.h"
#include "subgraph_ksnc.h"
#include "subgraph_default_kernel.h"

namespace daal
{
namespace algorithms
{
namespace subgraph
{
    
template<typename interm, Method method, CpuType cpu>
KSNCContainer<interm, method, cpu>::KSNCContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::subgraphBatchKernel, interm, method);
}

template<typename interm, Method method, CpuType cpu>
KSNCContainer<interm, method, cpu>::~KSNCContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
daal::services::interface1::Status KSNCContainer<interm, method, cpu>::compute()
{
    /* TODO */

    services::Status status;
    return status;

}

template<typename interm, Method method, CpuType cpu>
daal::services::interface1::Status KSNCContainer<interm, method, cpu>::finalizeCompute()
{
    /* TODO */
    services::Status status;
    return status;

}

}
}
} // namespace daal

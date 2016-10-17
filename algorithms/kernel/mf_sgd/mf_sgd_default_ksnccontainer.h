/* file: mf_sgd_dense_ksnccontainer.h */
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
//  Implementation of mf_sgd calculation algorithm container.
//--
*/

//#include "mf_sgd.h"
#include "mf_sgd_types.h"
#include "mf_sgd_ksnc.h"
#include "mf_sgd_default_kernel.h"
#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <math.h>       
#include <random>

#include "numeric_table.h"
#include "service_rng.h"

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
    
// typedef std::mt19937 CppRNG;

/**
 *  \brief Initialize list of cholesky kernels with implementations for supported architectures
 */
template<typename interm, Method method, CpuType cpu>
KSNCContainer<interm, method, cpu>::KSNCContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MF_SGDBatchKernel, interm, method);
}

template<typename interm, Method method, CpuType cpu>
KSNCContainer<interm, method, cpu>::~KSNCContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
void KSNCContainer<interm, method, cpu>::compute()
{
    // prepare the computation

    

}

template<typename interm, Method method, CpuType cpu>
void KSNCContainer<interm, method, cpu>::finalizeCompute()
{
    // prepare the computation

    

}

}
}
} // namespace daal

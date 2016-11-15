/* file: mf_sgd_dense_districontainer.h */
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
#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <math.h>       
#include <random>
#include "numeric_table.h"
#include "service_rng.h"

#include "mf_sgd_types.h"
#include "mf_sgd_distri.h"
#include "mf_sgd_default_kernel.h"

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
    

/**
 *  @brief Initialize list of mf_sgd with implementations for supported architectures
 */
template<ComputeStep step, typename interm, Method method, CpuType cpu>
DistriContainer<step, interm, method, cpu>::DistriContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MF_SGDDistriKernel, interm, method);
}

template<ComputeStep step, typename interm, Method method, CpuType cpu>
DistriContainer<step, interm, method, cpu>::~DistriContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<ComputeStep step, typename interm, Method method, CpuType cpu>
void DistriContainer<step, interm, method, cpu>::compute()
{
    // prepare the computation
    Input *input = static_cast<Input *>(_in);
    DistributedPartialResult *result = static_cast<DistributedPartialResult *>(_pres);

    const NumericTable *a0 = static_cast<const NumericTable *>(input->get(wPos).get());
    const NumericTable *a1 = static_cast<const NumericTable *>(input->get(hPos).get());
    const NumericTable *a2 = static_cast<const NumericTable *>(input->get(val).get());

    const NumericTable **WPos = &a0;
    const NumericTable **HPos = &a1;
    const NumericTable **Val = &a2;

    daal::algorithms::Parameter *par = _par;
    NumericTable *r[3];

    r[0] = static_cast<NumericTable *>(result->get(presWMat).get());
    r[1] = static_cast<NumericTable *>(result->get(presHMat).get());

    if ((static_cast<Parameter*>(_par))->_isTrain)
        r[2] = NULL;
    else
        r[2] = static_cast<NumericTable *>(result->get(presRMSE).get());

    daal::services::Environment::env &env = *_env;

    /* invoke the MF_SGDBatchKernel */
    __DAAL_CALL_KERNEL(env, internal::MF_SGDDistriKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), compute, WPos, HPos, Val, r, par);
   
}

template<ComputeStep step, typename interm, Method method, CpuType cpu>
void DistriContainer<step, interm, method, cpu>::finalizeCompute() {}

}
}
} // namespace daal

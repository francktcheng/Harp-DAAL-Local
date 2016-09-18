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

#include "mf_sgd_batch.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
namespace internal
{

template<typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
class MF_SGDBatchKernel : public Kernel
{
public:

    void compute(NumericTable** TrainSet,NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par);

    void compute_thr(NumericTable** TrainSet,NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par);

};


} // namespace daal::internal
}
}
} // namespace daal

#endif

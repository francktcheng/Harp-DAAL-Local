/* file: mf_sgd_dense_default_impl.i */
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

#ifndef __MF_SGD_UTILS_IMPL_I__
#define __MF_SGD_UTILS_IMPL_I__

#include "service_blas.h"
#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

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

/*
    assumed n < m
  Input:
    a_q at input : a[m][lda_q] -> A (m x n)
  Output:
    a_q at output: q[m][lda_q] -> Qn(m x n) = n leading columns of orthogonal Q
    r   at output: r[n][ldr  ] -> R (n x n) = upper triangular matrix written in lower triangular (fortran is evil)

*/
template <typename interm, CpuType cpu>
ServiceStatus compute_mf_sgd_on_one_node( MKL_INT m, MKL_INT n, interm *a_q, MKL_INT lda_q, interm *r,
                                      MKL_INT ldr )
{
    

    return SERV_ERR_OK;
}




} // namespace daal::internal
}
}
} // namespace daal

#endif

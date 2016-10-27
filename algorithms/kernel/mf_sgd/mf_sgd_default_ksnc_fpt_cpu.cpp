/* file: mf_sgd_dense_default_ksnc_fpt_cpu.cpp */
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
//  Instantiation of mf_sgd algorithm classes for SNC-4 mode of KNL
//--
*/

#include "mf_sgd_default_kernel.h"
#include "mf_sgd_default_ksnc_impl.i"
#include "mf_sgd_default_ksnccontainer.h"

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
namespace interface1
{
template class KSNCContainer<DAAL_FPTYPE, daal::algorithms::mf_sgd::defaultSGD, DAAL_CPU>;
}
namespace internal
{
}
}
}
}

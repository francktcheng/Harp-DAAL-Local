/* file: subgraph_dense_default_batch_fpt.cpp */
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
//  Instantiation of subgraph algorithm and types methods.
//  in subgraph_default_batch.h
//--
*/

#include "subgraph_default_batch.h"

namespace daal
{
namespace algorithms
{
namespace subgraph
{
namespace interface1
{

template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT void Result::free_mem<DAAL_FPTYPE>(size_t r, size_t w, size_t h);

template DAAL_EXPORT void Result::allocateImpl<DAAL_FPTYPE>(size_t r, size_t w, size_t h );
template DAAL_EXPORT void Result::freeImpl<DAAL_FPTYPE>(size_t r, size_t w, size_t h );

template DAAL_EXPORT void Result::allocateImpl_cache_aligned<DAAL_FPTYPE>(size_t r, size_t w, size_t h );
template DAAL_EXPORT void Result::freeImpl_cache_aligned<DAAL_FPTYPE>(size_t r, size_t w, size_t h );

template DAAL_EXPORT void Result::allocateImpl_hbw_mem<DAAL_FPTYPE>(size_t r, size_t w, size_t h );
template DAAL_EXPORT void Result::freeImpl_hbw_mem<DAAL_FPTYPE>(size_t r, size_t w, size_t h );


}// namespace interface1
}// namespace subgraph
}// namespace algorithms
}// namespace daal

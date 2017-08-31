/* file: subgraph_dense_default_batch.h */
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
//  Implementation of subgraph algorithm and types methods.
//--
*/
#ifndef __SUBGRAPH_DEFAULT_BATCH__
#define __SUBGRAPH_DEFAULT_BATCH__

#include <stdlib.h>     
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <time.h>

#include "subgraph_types.h"
#include "service_rng.h"
#include <tbb/cache_aligned_allocator.h>

using namespace tbb;

namespace daal
{
namespace algorithms
{
namespace subgraph
{
namespace interface1
{

// ------------------------------ impl of Result classs ------------------------------
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{/*{{{*/

    // cast algorithm::input to subgraph::input
    const Input *in = static_cast<const Input *>(input);
    // cast algorithm::parameter to subgraph::parameter
    const Parameter *par = static_cast<const Parameter *>(parameter);
    
    // func to allocate memory space for result class
    // allocateImpl_cache_aligned<algorithmFPType>(Dim_r, Dim_w, Dim_h);

}/*}}}*/

template <typename algorithmFPType>
DAAL_EXPORT void Result::free_mem(size_t r, size_t w, size_t h)
{/*{{{*/

    // func to manually free mem in result
    // freeImpl_cache_aligned<algorithmFPType>(r, w, h);

}/*}}}*/


/**
 * @brief allocate memory for Result
 *
 * @tparam algorithmFPType
 * @param r
 * @param w
 * @param h
 *
 * @return 
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocateImpl(size_t r, size_t w, size_t h)
{/*{{{*/

    //allocate numeric tables and assign them to Argument
    
    
    // /* allocate model W */
    // if (r == 0 || w == 0)
    // {
    //     Argument::set(resWMat, data_management::SerializationIfacePtr());
    // }
    // else
    // {
    //     algorithmFPType* w_data = (algorithmFPType*)malloc(sizeof(algorithmFPType)*r*w);
    //     Argument::set(resWMat, data_management::SerializationIfacePtr(
    //                       new data_management::HomogenNumericTable<algorithmFPType>(w_data, r, w)));
    // }
    //
    // /* allocate model H */
    // if (r == 0 || h == 0)
    // {
    //     Argument::set(resHMat, data_management::SerializationIfacePtr());
    // }
    // else
    // {
    //     algorithmFPType* h_data = (algorithmFPType*)malloc(sizeof(algorithmFPType)*r*h);
    //     Argument::set(resHMat, data_management::SerializationIfacePtr(
    //                       new data_management::HomogenNumericTable<algorithmFPType>(h_data, r, h)));
    // }

}/*}}}*/

template <typename algorithmFPType>
DAAL_EXPORT void Result::freeImpl(size_t r, size_t w, size_t h)
{/*{{{*/

    //retrieve numeric tables and free allocated memory
        
    // data_management::HomogenNumericTable<algorithmFPType>* wMat = (data_management::HomogenNumericTable<algorithmFPType>*)Argument::get(resWMat).get();
        // free(wMat->getArray());
        //
        // data_management::HomogenNumericTable<algorithmFPType>* hMat = (data_management::HomogenNumericTable<algorithmFPType>*)Argument::get(resHMat).get();
        // free(hMat->getArray());

}/*}}}*/

/**
 * @brief An alternative way to allocate memory aligned in cache
 *
 * @tparam algorithmFPType
 * @param r
 * @param w
 * @param h
 *
 * @return 
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocateImpl_cache_aligned(size_t r, size_t w, size_t h)
{/*{{{*/

    // /* allocate model W */
    // if (r == 0 || w == 0)
    // {
    //     Argument::set(resWMat, data_management::SerializationIfacePtr());
    // }
    // else
    // {
    //     algorithmFPType* w_data = cache_aligned_allocator<algorithmFPType>().allocate(r*w);
    //     Argument::set(resWMat, data_management::SerializationIfacePtr(
    //                       new data_management::HomogenNumericTable<algorithmFPType>(w_data, r, w)));
    // }
    //
    // /* allocate model H */
    // if (r == 0 || h == 0)
    // {
    //     Argument::set(resHMat, data_management::SerializationIfacePtr());
    // }
    // else
    // {
    //     algorithmFPType* h_data = cache_aligned_allocator<algorithmFPType>().allocate(r*h);
    //     Argument::set(resHMat, data_management::SerializationIfacePtr(
    //                       new data_management::HomogenNumericTable<algorithmFPType>(h_data, r, h)));
    // }

}/*}}}*/

template <typename algorithmFPType>
DAAL_EXPORT void Result::freeImpl_cache_aligned(size_t r, size_t w, size_t h)
{/*{{{*/

        // data_management::HomogenNumericTable<algorithmFPType>* wMat = (data_management::HomogenNumericTable<algorithmFPType>*)Argument::get(resWMat).get();
        // cache_aligned_allocator<algorithmFPType>().deallocate(wMat->getArray(), r*w);
        //
        // data_management::HomogenNumericTable<algorithmFPType>* hMat = (data_management::HomogenNumericTable<algorithmFPType>*)Argument::get(resHMat).get();
        // cache_aligned_allocator<algorithmFPType>().deallocate(hMat->getArray(), r*h);

}/*}}}*/

/**
 * @brief A third way to allocate memory in MCDRAM for KNL
 *
 * @tparam algorithmFPType
 * @param r
 * @param w
 * @param h
 *
 * @return 
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocateImpl_hbw_mem(size_t r, size_t w, size_t h)
{/*{{{*/

    // algorithmFPType* w_data;
    // algorithmFPType* h_data;
    //
    // /* allocate model W */
    // if (r == 0 || w == 0)
    // {
    //     Argument::set(resWMat, data_management::SerializationIfacePtr());
    // }
    // else
    // {
    //     // algorithmFPType* w_data = (algorithmFPType*)hbw_malloc(sizeof(algorithmFPType)*r*w);
    //     // int ret = hbw_posix_memalign((void**)&w_data, 64, sizeof(algorithmFPType)*r*w); 
    //     Argument::set(resWMat, data_management::SerializationIfacePtr(
    //                       new data_management::HomogenNumericTable<algorithmFPType>(w_data, r, w)));
    // }
    //
    // /* allocate model H */
    // if (r == 0 || h == 0)
    // {
    //     Argument::set(resHMat, data_management::SerializationIfacePtr());
    // }
    // else
    // {
    //     // algorithmFPType* h_data = (algorithmFPType*)hbw_malloc(sizeof(algorithmFPType)*r*h);
    //     // int ret = hbw_posix_memalign((void**)&h_data, 64, sizeof(algorithmFPType)*r*h); 
    //     Argument::set(resHMat, data_management::SerializationIfacePtr(
    //                       new data_management::HomogenNumericTable<algorithmFPType>(h_data, r, h)));
    // }

}/*}}}*/

template <typename algorithmFPType>
DAAL_EXPORT void Result::freeImpl_hbw_mem(size_t r, size_t w, size_t h)
{/*{{{*/

        // data_management::HomogenNumericTable<algorithmFPType>* wMat = (data_management::HomogenNumericTable<algorithmFPType>*)Argument::get(resWMat).get();
        // hbw_free(wMat->getArray());

        // data_management::HomogenNumericTable<algorithmFPType>* hMat = (data_management::HomogenNumericTable<algorithmFPType>*)Argument::get(resHMat).get();
        // hbw_free(hMat->getArray());

}/*}}}*/


}// namespace interface1
}// namespace subgraph
}// namespace algorithms
}// namespace daal

#endif

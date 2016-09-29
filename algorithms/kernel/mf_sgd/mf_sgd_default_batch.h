/* file: mf_sgd_dense_default_batch.h */
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
//  Implementation of mf_sgd algorithm and types methods.
//--
*/
#ifndef __MF_SGD_DEFAULT_BATCH__
#define __MF_SGD_DEFAULT_BATCH__

#include "mf_sgd_types.h"
#include <time.h>

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
namespace interface1
{

/**
 * Allocates memory for storing final results of the mf_sgd decomposition algorithm
 * \param[in] input     Pointer to input object
 * \param[in] parameter Pointer to parameter
 * \param[in] method    Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    // convert algorithm::Input to mf_sgd::Input
    const Input *in = static_cast<const Input *>(input);
    // convert algorithm::Parameter to mf_sgd::Parameter
    const Parameter *par = static_cast<const Parameter *>(parameter);
    
    size_t Dim_r = par->_Dim_r;
    size_t Dim_w = par->_Dim_w;
    size_t Dim_h = par->_Dim_h;

    // allocate NumericTable of model W and H
    allocateImpl<algorithmFPType>(Dim_r, Dim_w, Dim_h);

}


/**
 * Allocates memory for storing final results of the mf_sgd decomposition algorithm
 * \tparam     algorithmFPType  Data type to be used for storage in resulting HomogenNumericTable
 * \param[in]  r  feature dimension of model W and H 
 * \param[in]  w  Number of rows in model W 
 * \param[in]  h  Number of cols in model H 
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocateImpl(size_t r, size_t w, size_t h)
{
    // allocate model W
    if (r == 0 || w == 0)
    {
        Argument::set(resWMat, data_management::SerializationIfacePtr());
    }
    else
    {
        Argument::set(resWMat, data_management::SerializationIfacePtr(
                          new data_management::HomogenNumericTable<algorithmFPType>(r, w, data_management::NumericTable::doAllocate)));
    }


    // allocate model H
    if (r == 0 || h == 0)
    {
        Argument::set(resHMat, data_management::SerializationIfacePtr());
    }
    else
    {
        Argument::set(resHMat, data_management::SerializationIfacePtr(
                          new data_management::HomogenNumericTable<algorithmFPType>(r, h, data_management::NumericTable::doAllocate)));
    }

}

// template <typename algorithmFPType>


}// namespace interface1
}// namespace mf_sgd
}// namespace algorithms
}// namespace daal

#endif

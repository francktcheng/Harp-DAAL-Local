/* file: mf_sgd_distri_result.cpp */
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
//  Implementation of mf_sgd classes.
//--
*/

#include "algorithms/mf_sgd/mf_sgd_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
namespace interface1
{

/** Default constructor */
DistributedPartialResult::DistributedPartialResult() : daal::algorithms::PartialResult(2) {}

/**
 * Returns the result of the mf_sgd decomposition algorithm
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
NumericTablePtr DistributedPartialResult::get(DistributedPartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the mf_sgd decomposition algorithm
 * \param[in] id    Identifier of the result
 * \param[in] value Pointer to the result
 */
void DistributedPartialResult::set(DistributedPartialResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

void DistributedPartialResult::check(const daal::algorithms::Parameter *parameter, int method) const
{

    //to be implemented



}

/**
 * Checks final results of the algorithm
 * \param[in] input  Pointer to input objects
 * \param[in] par    Pointer to parameters
 * \param[in] method Computation method
 */
void DistributedPartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    //to be implemented

    // const Input *algInput = static_cast<const Input *>(input);
    //
    // size_t nVectors_Train = algInput->get(dataTrain)->getNumberOfRows();
    // size_t nFeatures_Train = algInput->get(dataTrain)->getNumberOfColumns();
    //
    // size_t nVectors_Test = algInput->get(dataTest)->getNumberOfRows();
    // size_t nFeatures_Test = algInput->get(dataTest)->getNumberOfColumns();
    //
    // int unexpectedLayouts = (int)packed_mask;

    // To Do check with Train and Test
    // if(!checkNumericTable(get(resWMat).get(), this->_errors.get(), "resWMat", unexpectedLayouts, 0, nFeatures, nVectors)) { return; }
    // if(!checkNumericTable(get(resHMat).get(), this->_errors.get(), "resHMat", unexpectedLayouts, 0, nFeatures, nFeatures)) { return; }
}


} // namespace interface1
} // namespace mf_sgd
} // namespace algorithm
} // namespace daal

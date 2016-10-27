/* file: mf_sgd_result.cpp */
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
Result::Result() : daal::algorithms::Result(2) {}

NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void Result::set(ResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const {}


} // namespace interface1
} // namespace mf_sgd
} // namespace algorithm
} // namespace daal

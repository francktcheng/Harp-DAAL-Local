/* file: subgraph_input.cpp */
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
//  Implementation of subgraph classes.
//  non-template func
//--
*/

#include "algorithms/subgraph/subgraph_types.h"
#include "service_micro_table.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace subgraph
{
namespace interface1
{

Input::Input() : daal::algorithms::Input(8) {}

NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void Input::set(InputId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

size_t Input::getNumberOfColumns(InputId id) const
{
    NumericTablePtr dataTable = get(id);
    if(dataTable)
        return dataTable->getNumberOfColumns();
    else
        return 0;
}

size_t Input::getNumberOfRows(InputId id) const
{
    NumericTablePtr dataTable = get(id);
    if(dataTable)
        return dataTable->getNumberOfRows();
    else
        return 0;
}

daal::services::interface1::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const {services::Status s; return s;}


} // namespace interface1
} // namespace subgraph
} // namespace algorithm
} // namespace daal

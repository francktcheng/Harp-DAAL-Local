/* file: mf_sgd_input.cpp */
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
// Input::Input() : daal::algorithms::Input(1) {}
Input::Input() : daal::algorithms::Input(2) {}

/**
 * Returns input object of the mf_sgd decomposition algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets input object for the mf_sgd decomposition algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] value Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

size_t Input::getNumberOfColumns(InputId id) const
{
    NumericTablePtr dataTable = get(id);
    if(dataTable)
    {
        return dataTable->getNumberOfColumns();
    }
    else
    {
        this->_errors->add(Error::create(ErrorNullNumericTable, ArgumentName, dataStr()));
        return 0;
    }
}

size_t Input::getNumberOfRows(InputId id) const
{
    NumericTablePtr dataTable = get(id);
    if(dataTable)
    {
        return dataTable->getNumberOfRows();
    }
    else
    {
        this->_errors->add(Error::create(ErrorNullNumericTable, ArgumentName, dataStr()));
        return 0;
    }
}

/**
 * Checks parameters of the algorithm
 * \param[in] parameter Pointer to the parameters
 * \param[in] method Computation method
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    // First check the Training data
    // NumericTablePtr dataTable_Train = get(dataTrain);
    // if(!checkNumericTable(dataTable.get(), this->_errors.get(), dataStr())) { return; }

    // DAAL_CHECK_EX(dataTable->getNumberOfColumns() <= dataTable->getNumberOfRows(), ErrorIncorrectNumberOfRows, ArgumentName, dataStr());

    // First check the Test data
    // NumericTablePtr dataTable_Test = get(dataTest);
    // if(!checkNumericTable(dataTable.get(), this->_errors.get(), dataStr())) { return; }

    // DAAL_CHECK_EX(dataTable->getNumberOfColumns() <= dataTable->getNumberOfRows(), ErrorIncorrectNumberOfRows, ArgumentName, dataStr());
}

} // namespace interface1
} // namespace mf_sgd
} // namespace algorithm
} // namespace daal

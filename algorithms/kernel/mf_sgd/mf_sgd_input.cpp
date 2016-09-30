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
#include "service_micro_table.h"
#include "service_rng.h"
#include <stdlib.h>     
#include <map>

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

template <typename algorithmFPType>
void Input::generate_points(daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Test, long num_Test,  long row_num_w, long col_num_h)
{/*{{{*/

    // srand((unsigned)time(0)); 
	
	daal::internal::UniformRng<algorithmFPType, daal::sse2> rng(time(0));
    
    algorithmFPType value;

	long counts_train = 0;
	long counts_test = 0;

	long i, j;
	for(i=0;i<row_num_w;i++)
	{
		for (j=0;j<col_num_h;j++) 
		{
			if (i == j)
			{
				// put diagonal item into train dataset
				if (counts_train < num_Train)
				{
					points_Train[counts_train].wPos = i;
					points_Train[counts_train].hPos = j;

                    rng.uniform(1, 0.0, 1.0, &value);
					// points_Train[counts_train].val = (10.0*((algorithmFPType)rand()/(RAND_MAX) + 0.5));
					points_Train[counts_train].val = 10.0*value;
					counts_train++;
				}
			}
			else
			{
				if ( ((algorithmFPType)rand())/RAND_MAX > 0.2)
				{
					// put item into train dataset
					if (counts_train < num_Train)
					{
						points_Train[counts_train].wPos = i;
						points_Train[counts_train].hPos = j;
                        
                        rng.uniform(1, 0.0, 1.0, &value);
						// points_Train[counts_train].val = (10.0*((algorithmFPType)rand()/(RAND_MAX) +0.5));
					    points_Train[counts_train].val = 10.0*value;
						counts_train++;
					}
					else if (counts_test < num_Test)
					{
						points_Test[counts_test].wPos = i;
						points_Test[counts_test].hPos = j;

                        rng.uniform(1, 0.0, 1.0, &value);
						// points_Test[counts_test].val = (10.0*((algorithmFPType)rand()/(RAND_MAX) +0.5));
						points_Test[counts_test].val = 10.0*value;
						counts_test++;
					}
				}
				else
				{
					// put item into test dataset
					if (counts_test < num_Test)
					{
						points_Test[counts_test].wPos = i;
						points_Test[counts_test].hPos = j;

                        rng.uniform(1, 0.0, 1.0, &value);
						// points_Test[counts_test].val = (10.0*((algorithmFPType)rand()/(RAND_MAX) +0.5));
						points_Test[counts_test].val = 10.0*value;
						counts_test++;
					}
					else if (counts_train < num_Train)
					{
						points_Train[counts_train].wPos = i;
						points_Train[counts_train].hPos = j;

                        rng.uniform(1, 0.0, 1.0, &value);
						// points_Train[counts_train].val = (10.0*((algorithmFPType)rand()/(RAND_MAX) +0.5));
						points_Train[counts_train].val = 10.0*value;
						counts_train++;
					}
				}
			}
		}
	}


}/*}}}*/

template <typename algorithmFPType>
void Input::convert_format(daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Test, 
            long num_Test, data_management::NumericTablePtr trainTable, data_management::NumericTablePtr testTable, long &row_num_w, long &col_num_h)
{

    std::map<long, long> vMap_row_w; // a hashmap where key is the row id of input matrix, value is the row position in model W 
    std::map<long, long> vMap_col_h; // a hashmap where key is the col id of input matrix, value is the row position in model H 

	daal::internal::BlockMicroTable<algorithmFPType, readOnly, daal::sse2> trainMic(trainTable.get());
	daal::internal::BlockMicroTable<algorithmFPType, readOnly, daal::sse2> testMic(testTable.get());

    algorithmFPType* train_ptr = 0;
    algorithmFPType* test_ptr = 0;

    long row_pos = 0;
    long col_pos = 0;

    long row_vpoint = 0;
    long col_vpoint = 0;
    long i;
    long row_id;
    long col_id;
    algorithmFPType val;

    //convert Train Dataset
    for (i = 0; i < num_Train; i++) {
        
        trainMic.getBlockOfRows(i, 1, &train_ptr);
        row_id = (long)train_ptr[0];
        col_id = (long)train_ptr[1];
        val = train_ptr[2];

        if (vMap_row_w.find(row_id) == vMap_row_w.end())
        {
            //not found row id
            vMap_row_w[row_id] = row_pos;
            row_vpoint = row_pos;
            row_pos++;
        }
        else
            row_vpoint = vMap_row_w[row_id];

        if (vMap_col_h.find(col_id) == vMap_col_h.end())
        {
            //not found col id
            vMap_col_h[col_id] = col_pos;
            col_vpoint = col_pos;
            col_pos++;
        }
        else
            col_vpoint = vMap_col_h[col_id];

        points_Train[i].wPos = row_vpoint;
        points_Train[i].hPos = col_vpoint;
        points_Train[i].val = val;

    }

    row_num_w = row_pos;
    col_num_h = col_pos;

    //convert Test dataset
    for (i = 0; i < num_Test; i++) {

        testMic.getBlockOfRows(i, 1, &test_ptr);
        row_id = (long)test_ptr[0];
        col_id = (long)test_ptr[1];
        val = test_ptr[2];

        if (vMap_row_w.find(row_id) == vMap_row_w.end())
        {
            //not found row id
            row_vpoint = -1;
        }
        else
            row_vpoint = vMap_row_w[row_id];


        if (vMap_col_h.find(col_id) == vMap_col_h.end())
        {
            //not found row id
            col_vpoint = -1;
        }
        else
            col_vpoint = vMap_col_h[col_id];

        points_Test[i].wPos = row_vpoint;
        points_Test[i].hPos = col_vpoint;
        points_Test[i].val = val;

    }

}

template void Input::generate_points(daal::algorithms::mf_sgd::VPoint<double>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<double>* points_Test, long num_Test,  long row_num_w, long col_num_h);

template void Input::generate_points(daal::algorithms::mf_sgd::VPoint<float>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<float>* points_Test, long num_Test,  long row_num_w, long col_num_h);

template void Input::convert_format(daal::algorithms::mf_sgd::VPoint<double>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<double>* points_Test, 
            long num_Test, data_management::NumericTablePtr trainTable, data_management::NumericTablePtr testTable, long &row_num_w, long &col_num_h);

template void Input::convert_format(daal::algorithms::mf_sgd::VPoint<float>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<float>* points_Test, 
            long num_Test, data_management::NumericTablePtr trainTable, data_management::NumericTablePtr testTable, long &row_num_w, long &col_num_h);

} // namespace interface1
} // namespace mf_sgd
} // namespace algorithm
} // namespace daal

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
#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

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
            long num_Test, std::unordered_map<long, std::vector<mf_sgd::VPoint<algorithmFPType>*>*> &map_train, 
            std::unordered_map<long, std::vector<mf_sgd::VPoint<algorithmFPType>*>*> &map_test, long &row_num_w, long &col_num_h)
{/*{{{*/

    std::unordered_map<long, long> vMap_row_w;
    std::unordered_map<long, long> vMap_col_h;

    long row_pos_itr = 0;
    long col_pos_itr = 0;

    long row_pos = 0;
    long col_pos = 0;
    long entry_itr = 0;

    long row_id;
    long col_id;
    algorithmFPType val;

    typename std::unordered_map<long, std::vector<mf_sgd::VPoint<algorithmFPType>*>*>::iterator it_map;
    typename std::vector<mf_sgd::VPoint<algorithmFPType>*>::iterator it_vec;

    //iteration over train data map
    for (it_map = map_train.begin(); it_map != map_train.end(); ++it_map) 
    {

        if (it_map->second->empty() == false)
        {
            for(it_vec = it_map->second->begin(); it_vec < it_map->second->end();it_vec++)
            {
                row_id = (*it_vec)->wPos;
                col_id = (*it_vec)->hPos;
                val = (*it_vec)->val;

                if (vMap_row_w.find(row_id) == vMap_row_w.end())
                {
                    //not found row id
                    vMap_row_w[row_id] = row_pos_itr;
                    row_pos = row_pos_itr;
                    row_pos_itr++;
                }
                else
                    row_pos = vMap_row_w[row_id];

                if (vMap_col_h.find(col_id) == vMap_col_h.end())
                {
                    //not found col id
                    vMap_col_h[col_id] = col_pos_itr;
                    col_pos = col_pos_itr;
                    col_pos_itr++;
                }
                else
                    col_pos = vMap_col_h[col_id];

                points_Train[entry_itr].wPos = row_pos;
                points_Train[entry_itr].hPos = col_pos;
                points_Train[entry_itr].val = val;

                entry_itr++;

            }

        }

    }

    row_num_w = row_pos_itr;
    col_num_h = col_pos_itr;


    // iteration over test data map
    entry_itr = 0;

    for (it_map = map_test.begin(); it_map != map_test.end(); ++it_map) 
    {

        if (it_map->second->empty() == false)
        {
            for(it_vec = it_map->second->begin(); it_vec < it_map->second->end();it_vec++)
            {
                row_id = (*it_vec)->wPos;
                col_id = (*it_vec)->hPos;
                val = (*it_vec)->val;

                if (vMap_row_w.find(row_id) == vMap_row_w.end())
                {
                    //not found row id
                    row_pos = -1;
                }
                else
                    row_pos = vMap_row_w[row_id];

                if (vMap_col_h.find(col_id) == vMap_col_h.end())
                {
                    //not found col id
                    col_pos = -1;
                }
                else
                    col_pos = vMap_col_h[col_id];

                points_Test[entry_itr].wPos = row_pos;
                points_Test[entry_itr].hPos = col_pos;
                points_Test[entry_itr].val = val;

                entry_itr++;

            }

        }

    }


}/*}}}*/

/**
 * @brief load dataset from CSV files into a map
 * CSV files are space delimited
 *
 * @tparam algorithmFPType
 * @param filename
 * @param map
 * @param lineContainer
 */
template <typename algorithmFPType>
void Input::loadData(std::string filename, std::unordered_map<long, std::vector<mf_sgd::VPoint<algorithmFPType>*>*> &map, long &num_points, std::vector<long>* lineContainer)
{/*{{{*/
    
    long row_id;
    long col_id;
    algorithmFPType val;

    num_points = 0;

    std::string line;

    std::ifstream infile(filename);

    while (std::getline(infile, line)) 
    {
        infile >> row_id >> col_id >> val;

        //processing a line
        mf_sgd::VPoint<algorithmFPType>* item = new mf_sgd::VPoint<algorithmFPType>();
        item->wPos = row_id;
        item->hPos = col_id;
        item->val = val;

        num_points++;

        if (map.find(row_id) == map.end())
        {
            //not found row id
            std::vector<mf_sgd::VPoint<algorithmFPType>*>* vec = new std::vector<mf_sgd::VPoint<algorithmFPType>*>();
            vec->push_back(item);
            map[row_id] = vec;

            if (lineContainer != NULL)
                lineContainer->push_back(row_id);

        }
        else
            map[row_id]->push_back(item);

    }

    infile.close();

}/*}}}*/


/**
 * @brief free up the memory allocated within the map structure
 *
 * @tparam algorithmFPType
 * @param map
 */
template <typename algorithmFPType>
void Input::freeData(std::unordered_map<long, std::vector<mf_sgd::VPoint<algorithmFPType>*>*> &map)
{/*{{{*/
        typename std::unordered_map<long, std::vector<mf_sgd::VPoint<algorithmFPType>*>*>::iterator it_map;
        typename std::vector<mf_sgd::VPoint<algorithmFPType>*>::iterator it_vec;
   
        for (it_map = map.begin(); it_map != map.end(); ++it_map) 
        {

            if (it_map->second->empty() == false)
            {
                // std::vector<mf_sgd::VPoint<algorithmFPType>*>::iterator it_vec;
                // VEC_Itr it_vec;
                for(it_vec = it_map->second->begin(); it_vec < it_map->second->end();it_vec++)
                {
                    delete *it_vec;
                }

                delete it_map->second;

            }

        }

        map.clear();

}/*}}}*/

template void Input::generate_points(daal::algorithms::mf_sgd::VPoint<double>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<double>* points_Test, long num_Test,  long row_num_w, long col_num_h);

template void Input::generate_points(daal::algorithms::mf_sgd::VPoint<float>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<float>* points_Test, long num_Test,  long row_num_w, long col_num_h);

template void Input::convert_format(daal::algorithms::mf_sgd::VPoint<double>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<double>* points_Test, 
            long num_Test, std::unordered_map<long, std::vector<mf_sgd::VPoint<double>*>*> &map_train, 
            std::unordered_map<long, std::vector<mf_sgd::VPoint<double>*>*> &map_test, long &row_num_w, long &col_num_h);

template void Input::convert_format(daal::algorithms::mf_sgd::VPoint<float>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<float>* points_Test, 
            long num_Test, std::unordered_map<long, std::vector<mf_sgd::VPoint<float>*>*> &map_train, 
            std::unordered_map<long, std::vector<mf_sgd::VPoint<float>*>*> &map_test, long &row_num_w, long &col_num_h);

template void Input::loadData(std::string filename, std::unordered_map<long, std::vector<mf_sgd::VPoint<double>*>*> &map, long &num_points, std::vector<long>* lineContainer);

template void Input::loadData(std::string filename, std::unordered_map<long, std::vector<mf_sgd::VPoint<float>*>*> &map, long &num_points, std::vector<long>* lineContainer);

template void Input::freeData(std::unordered_map<long, std::vector<mf_sgd::VPoint<double>*>*> &map);

template void Input::freeData(std::unordered_map<long, std::vector<mf_sgd::VPoint<float>*>*> &map);

} // namespace interface1
} // namespace mf_sgd
} // namespace algorithm
} // namespace daal

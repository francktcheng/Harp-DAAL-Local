/* file: mf_sgd_dense_batch.cpp */
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
!  Content:
!    C++ example of computing mf_sgd decomposition in the batch processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-mf_sgd_BATCH"></a>
 * \example mf_sgd_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include <cstdlib> 
#include <ctime> 
#include <time.h>

using namespace std;
using namespace daal;
using namespace daal::algorithms;


// struct for a data point
struct VPoint 
{
    int wPos;
    int hPos;
    double val;
};

// A function to generate the input data
void mf_sgd_dataGenerator(VPoint* points_Train, const size_t num_Train, VPoint* points_Test, const size_t num_Test, const long row_num_w, const long col_num_h);

/**
 * $V = W H$
 * 1) Calculate the error for each training point V_{i,j}
 * $E_{ij} = V_{ij} - \sum_{k=0}^r W_{ik} H_{kj}$
 *
 * 2) Update model W by training point V_{i,j}  
 * $W_{i*} = W_{i*} - learningRate\cdot (E_{ij} \cdot H_{*j} + \lambda \cdot W_{i*})$
 *
 * 3) Update model H by training point V_{i,j}
 * $H_{*j} = H_{*j} - learningRate\cdot (E_{ij} \cdot W_{i*} + \lambda \cdot H_{*j})$
 */
int main(int argc, char *argv[])
{
    // checkArguments(argc, argv, 1, &datasetFileName);

	// parameters of SGD training
    const double learningRate = 0.05;
    const double lambda = 0.002;
    const int iteration = 10;		//num of iterations in SGD training
    const int threads = 40;			// threads used by TBB

	// dimension of model W and model H
    const long r_dim = 10;
    const long row_num_w = 1000;
    const long col_num_h = 1000;

    const long col_num_w = r_dim;
    const long row_num_h = r_dim;

	// size of training dataset and test dataset
    size_t num_Train = row_num_w + 0.6*(row_num_w*col_num_h - row_num_w);
    size_t num_Test = 0.002*(row_num_w*col_num_h- row_num_w);
    const size_t field_v = 3;
		
	// generate the Train and Test datasets
	VPoint* points_Train = new VPoint[num_Train];
	VPoint* points_Test = new VPoint[num_Test];

	mf_sgd_dataGenerator(points_Train, num_Train, points_Test, num_Test, row_num_w, col_num_h);

	printf("num_Train: %d\n", num_Train);
	printf("num_Test: %d\n", num_Test);

    /* Create a new dictionary and fill it with the information about data */
    NumericTableDictionary newDict_Train(field_v);
    NumericTableDictionary newDict_Test(field_v);

    /* Add a feature type to the dictionary */
    newDict_Train[0].featureType = data_feature_utils::DAAL_CONTINUOUS;
    newDict_Train[1].featureType = data_feature_utils::DAAL_CONTINUOUS;
    newDict_Train[2].featureType = data_feature_utils::DAAL_CONTINUOUS;

    newDict_Test[0].featureType = data_feature_utils::DAAL_CONTINUOUS;
    newDict_Test[1].featureType = data_feature_utils::DAAL_CONTINUOUS;
    newDict_Test[2].featureType = data_feature_utils::DAAL_CONTINUOUS;

    services::SharedPtr<AOSNumericTable> dataTable_Train(new AOSNumericTable(points_Train, field_v, num_Train));
    services::SharedPtr<AOSNumericTable> dataTable_Test(new AOSNumericTable(points_Test, field_v, num_Test));

    /* Assign the new dictionary to an existing numeric table */
    dataTable_Train->setDictionary(&newDict_Train);
    dataTable_Test->setDictionary(&newDict_Test);

    /* Add data to the numeric table */
    dataTable_Train->setFeature<int> (0, DAAL_STRUCT_MEMBER_OFFSET(VPoint, wPos));
    dataTable_Train->setFeature<int> (1, DAAL_STRUCT_MEMBER_OFFSET(VPoint, hPos));
    dataTable_Train->setFeature<double> (2, DAAL_STRUCT_MEMBER_OFFSET(VPoint, val));

    /* Add data to the numeric table */
    dataTable_Test->setFeature<int> (0, DAAL_STRUCT_MEMBER_OFFSET(VPoint, wPos));
    dataTable_Test->setFeature<int> (1, DAAL_STRUCT_MEMBER_OFFSET(VPoint, hPos));
    dataTable_Test->setFeature<double> (2, DAAL_STRUCT_MEMBER_OFFSET(VPoint, val));

    /* Create an algorithm to compute mf_sgd decomposition */
    // use default template value: double and defaultSGD
    mf_sgd::Batch<> algorithm;

    // algorithm.input.set(mf_sgd::dataTrain, dataSource.getNumericTable());
    algorithm.input.set(mf_sgd::dataTrain, dataTable_Train);
    algorithm.input.set(mf_sgd::dataTest, dataTable_Test);

    algorithm.parameter.setParameter(learningRate, lambda, r_dim, row_num_w, col_num_h, iteration, threads);

    /* Compute mf_sgd decomposition */
	clock_t start_compute = clock();

    algorithm.compute();

	clock_t stop_compute = clock();

	double compute_time = (double)(stop_compute - start_compute)*1000.0/CLOCKS_PER_SEC;
	
    services::SharedPtr<mf_sgd::Result> res = algorithm.getResult();

	printf("Computation Time elapsed in ms: %f\n", compute_time);

    /* Print the results */
    // printNumericTable(res->get(mf_sgd::resWMat), "Model W Matrix:", 10, 10, 10);
    // printNumericTable(res->get(mf_sgd::resHMat), "Model H Matrix:", 10, 10, 10);

	delete[] points_Train;
	delete[] points_Test;

    return 0;
}

void mf_sgd_dataGenerator(VPoint* points_Train, const size_t num_Train, VPoint* points_Test, const size_t num_Test, const long row_num_w, const long col_num_h)
{/*{{{*/

	srand((unsigned)time(0)); 
	
	int counts_train = 0;
	int counts_test = 0;

	int i, j;
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
					points_Train[counts_train].val = (10.0*((double)rand()/(RAND_MAX) + 0.5));
					counts_train++;
				}
			}
			else
			{
				if ( ((double)rand())/RAND_MAX > 0.2)
				{
					// put item into train dataset
					if (counts_train < num_Train)
					{
						points_Train[counts_train].wPos = i;
						points_Train[counts_train].hPos = j;
						points_Train[counts_train].val = (10.0*((double)rand()/(RAND_MAX) +0.5));
						counts_train++;
					}
					else if (counts_test < num_Test)
					{
						points_Test[counts_test].wPos = i;
						points_Test[counts_test].hPos = j;
						points_Test[counts_test].val = (10.0*((double)rand()/(RAND_MAX) +0.5));
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
						points_Test[counts_test].val = (10.0*((double)rand()/(RAND_MAX) +0.5));
						counts_test++;
					}
					else if (counts_train < num_Train)
					{
						points_Train[counts_train].wPos = i;
						points_Train[counts_train].hPos = j;
						points_Train[counts_train].val = (10.0*((double)rand()/(RAND_MAX) +0.5));
						counts_train++;
					}
				}
			}
		}
	}

}/*}}}*/

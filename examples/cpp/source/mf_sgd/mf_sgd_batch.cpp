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
#include <stdlib.h> 

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace daal;
using namespace daal::algorithms;

//parameters of SGD training
double learningRate = 0.005;
double lambda = 0.002;
int iteration = 10;		    //num of iterations in SGD training
int threads = 20;			// threads used by TBB
int tbb_grainsize = 10000;   //grainsize for TBB parallel_for 
int isAvx512 = 1;

// dimension of model W and model H
long r_dim = 1000;

long row_num_w = 1000;
long col_num_w = r_dim;

long row_num_h = r_dim;
long col_num_h = 1000;

// dimension of training and testing matrices
long num_Train;
long num_Test;
const size_t field_v = 3;

//choose the precision
typedef float sgd_float;
// typedef double sgd_float;

// input dataset files in csv format
// string trainDataFile = "../../../data/batch/movielens-train.mm";
// string testDataFile = "../../../data/batch/movielens-test.mm";

// string trainDataFile = "../../../data/batch/yahoomusic-train.mm";
// string testDataFile = "../../../data/batch/yahoomusic-test.mm";

//absolute path used in VTune
// string trainDataFile = "/home/langshichen/Lib/__release_lnx/daal/examples/data/batch/movielens-train.mm";
// string testDataFile = "/home/langshichen/Lib/__release_lnx/daal/examples/data/batch/movielens-test.mm";

string trainDataFile = "/home/langshichen/Lib/__release_lnx/daal/examples/data/batch/yahoomusic-train.mm";
string testDataFile = "/home/langshichen/Lib/__release_lnx/daal/examples/data/batch/yahoomusic-test.mm";

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
    
	if (argc > 1)
		learningRate = atof(argv[1]);

	if (argc > 2)
		lambda = atof(argv[2]);

	if (argc > 3)
		iteration = atoi(argv[3]);

	if (argc > 4)
		threads = atoi(argv[4]);

	if (argc > 5)
		tbb_grainsize = atoi(argv[5]);

	if (argc > 6)
		r_dim = atol(argv[6]);

    if (argc > 7)
        isAvx512 = atoi(argv[7]);

	if (argc > 8)
		row_num_w = atol(argv[8]);

	if (argc > 9)
		col_num_h = atol(argv[9]);

    col_num_w = r_dim;
    row_num_h = r_dim;

	mf_sgd::VPoint<sgd_float>* points_Train;
	mf_sgd::VPoint<sgd_float>* points_Test;

    // Create an algorithm to compute mf_sgd decomposition 
    mf_sgd::Batch<sgd_float, mf_sgd::defaultSGD> algorithm;

    struct timespec ts1;
	struct timespec ts2;
    long diff;
    double diff_ms;

	if (argc > 8)
	{
		// generate the dataset
		// size of training dataset and test dataset
		num_Train = row_num_w + 0.6*(row_num_w*col_num_h - row_num_w);
		num_Test = 0.002*(row_num_w*col_num_h- row_num_w);

		// generate the Train and Test datasets
		points_Train = new mf_sgd::VPoint<sgd_float>[num_Train];
		points_Test = new mf_sgd::VPoint<sgd_float>[num_Test];

		algorithm.input.generate_points<sgd_float>(points_Train, num_Train, points_Test, num_Test, row_num_w, col_num_h);

		printf("Train set num of Points: %d\n", num_Train);
		printf("Test set num of Points: %d\n", num_Test);
		printf("Model W Rows: %d\n", row_num_w);
		printf("Model H Columns: %d\n", col_num_h);
		printf("Model Dimension: %d\n", r_dim);

	}
	else
	{
        
        //----------------- Start reading datasets from csv file --------------------------
        std::unordered_map<long, std::vector<mf_sgd::VPoint<sgd_float>*>*> map_train;
        std::unordered_map<long, std::vector<mf_sgd::VPoint<sgd_float>*>*> map_test;

        //load train set data
	    clock_gettime(CLOCK_MONOTONIC, &ts1);

        algorithm.input.loadData(trainDataFile, map_train, num_Train, NULL);

        clock_gettime(CLOCK_MONOTONIC, &ts2);

        diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
	    diff_ms = (double)(diff)/1000000L;

        printf("Finish loading Train Data using %f ms\n", diff_ms);

        //load test set data
	    clock_gettime(CLOCK_MONOTONIC, &ts1);

        algorithm.input.loadData(testDataFile, map_test, num_Test, NULL);

        clock_gettime(CLOCK_MONOTONIC, &ts2);

        diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
	    diff_ms = (double)(diff)/1000000L;

        printf("Finish loading Test Data using %f ms\n", diff_ms);


        //----------------- End reading datasets from csv file --------------------------

        points_Train = new mf_sgd::VPoint<sgd_float>[num_Train];
		points_Test = new mf_sgd::VPoint<sgd_float>[num_Test];

		// a function to convert the row id and column id
        printf("Start Converting Sparse Data\n");

	    clock_gettime(CLOCK_MONOTONIC, &ts1);

		algorithm.input.convert_format(points_Train, num_Train, points_Test, num_Test, map_train, map_test, row_num_w, col_num_h);

        clock_gettime(CLOCK_MONOTONIC, &ts2);
        diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
	    diff_ms = (double)(diff)/1000000L;

        algorithm.input.freeData(map_train);
        algorithm.input.freeData(map_test);

        printf("Finish Converting Sparse Data using %f ms\n", diff_ms);
        printf("Train set num of Points: %d\n", num_Train);
		printf("Test set num of Points: %d\n", num_Test);
		printf("Model W Rows: %d\n", row_num_w);
		printf("Model H Columns: %d\n", col_num_h);
		printf("Model Dimension: %d\n", r_dim);
	
	}

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
    dataTable_Train->setFeature<long> (0, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, wPos));
    dataTable_Train->setFeature<long> (1, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, hPos));
    dataTable_Train->setFeature<sgd_float> (2, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, val));

    /* Add data to the numeric table */
    dataTable_Test->setFeature<long> (0, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, wPos));
    dataTable_Test->setFeature<long> (1, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, hPos));
    dataTable_Test->setFeature<sgd_float> (2, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, val));

	// debug
	// printNumericTable(dataTable_Train, "Train dataset:", 5, 3, 10);
	// printNumericTable(dataTable_Test, "Test dataset:", 5, 3, 10);

    algorithm.input.set(mf_sgd::dataTrain, dataTable_Train);
    algorithm.input.set(mf_sgd::dataTest, dataTable_Test);

    algorithm.parameter.setParameter(learningRate, lambda, r_dim, row_num_w, col_num_h, iteration, threads, tbb_grainsize, isAvx512);

    /* Compute mf_sgd decomposition */

    algorithm.compute();

    /* Print the results */
    // printNumericTable(res->get(mf_sgd::resWMat), "Model W Matrix:", 10, 10, 10);
    // printNumericTable(res->get(mf_sgd::resHMat), "Model H Matrix:", 10, 10, 10);

	delete[] points_Train;
	delete[] points_Test;

    return 0;
}









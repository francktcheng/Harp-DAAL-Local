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
#include <cstdlib> 
#include <ctime> 
#include <time.h>
#include <stdlib.h> 
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* default parameters of SGD training */
const double learningRate = 0.005;
const double lambda = 0.002;
const size_t iteration = 10;		    /* num of iterations in SGD training */
const size_t threads = 0;			    /* specify the num of threads used by TBB, when 0 TBB chooses automatic threads num  */
const size_t tbb_grainsize = 0;			/* 0 by auto partitioner of TBB or user specified tbb grainsize   */
const size_t Avx_explicit = 0;			/* 0 to use compiler generated vectorization codes, 1 to use explicit intel intrinsic codes */

/* dimension of vectors in model W and model H */
const int64_t r_dim = 128;

/* input dataset files in csv format */
const string trainDataFile = "../../../data/batch/movielens-train.mm";
const string testDataFile = "../../../data/batch/movielens-test.mm";

/* choose the precision */
// typedef float sgd_float;
typedef double sgd_float;

int main(int argc, char *argv[])
{

	/* Create an algorithm to compute mf_sgd decomposition  */
	mf_sgd::Batch<sgd_float, mf_sgd::defaultSGD> algorithm;

	/* map to hold imported csv data */
	std::unordered_map<int64_t, std::vector<mf_sgd::VPoint<sgd_float>*>*> map_train;
	std::unordered_map<int64_t, std::vector<mf_sgd::VPoint<sgd_float>*>*> map_test;

	const int num_Train = algorithm.input.loadData(trainDataFile, NULL, map_train);
	const int num_Test = algorithm.input.loadData(testDataFile, NULL, map_test);

	services::SharedPtr<mf_sgd::VPoint<sgd_float> > points_Train(new mf_sgd::VPoint<sgd_float>[num_Train]);
	services::SharedPtr<mf_sgd::VPoint<sgd_float> > points_Test(new mf_sgd::VPoint<sgd_float>[num_Test]);

	int64_t row_num_w = 0;
	int64_t col_num_h = 0;
	algorithm.input.convert_format(map_train, map_test, num_Train, num_Test, points_Train.get(), points_Test.get(), row_num_w, col_num_h);

	algorithm.input.freeData(map_train);
	algorithm.input.freeData(map_test);

	/* Create a new dictionary and fill it with the information about data */
	const size_t field_v = 3; 
	NumericTableDictionary newDict_Train(field_v);
	NumericTableDictionary newDict_Test(field_v);

	/* Add a feature type to the dictionary */
	newDict_Train[0].featureType = data_feature_utils::DAAL_CONTINUOUS;
	newDict_Train[1].featureType = data_feature_utils::DAAL_CONTINUOUS;
	newDict_Train[2].featureType = data_feature_utils::DAAL_CONTINUOUS;

	newDict_Test[0].featureType = data_feature_utils::DAAL_CONTINUOUS;
	newDict_Test[1].featureType = data_feature_utils::DAAL_CONTINUOUS;
	newDict_Test[2].featureType = data_feature_utils::DAAL_CONTINUOUS;

	services::SharedPtr<AOSNumericTable> dataTable_Train(new AOSNumericTable(points_Train.get(), field_v, num_Train));
	services::SharedPtr<AOSNumericTable> dataTable_Test(new AOSNumericTable(points_Test.get(), field_v, num_Test));

	/* Assign the new dictionary to an existing numeric table */
	dataTable_Train->setDictionary(&newDict_Train);
	dataTable_Test->setDictionary(&newDict_Test);

	/* Add training data to the numeric table */
	dataTable_Train->setFeature<int64_t> (0, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, wPos));
	dataTable_Train->setFeature<int64_t> (1, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, hPos));
	dataTable_Train->setFeature<sgd_float> (2, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, val));

	/* Add test data to the numeric table */
	dataTable_Test->setFeature<int64_t> (0, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, wPos));
	dataTable_Test->setFeature<int64_t> (1, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, hPos));
	dataTable_Test->setFeature<sgd_float> (2, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, val));

	algorithm.input.set(mf_sgd::dataTrain, dataTable_Train);
	algorithm.input.set(mf_sgd::dataTest, dataTable_Test);

	algorithm.parameter.setParameter(learningRate, lambda, r_dim, row_num_w, col_num_h, iteration, threads, tbb_grainsize, Avx_explicit);

	algorithm.compute();

	services::SharedPtr<mf_sgd::Result> res = algorithm.getResult();

	/* Print the results first 10 rows and 10 columns */
	printNumericTable(res->get(mf_sgd::resWMat), "Model W Matrix:", 10, 10, 10);
	printNumericTable(res->get(mf_sgd::resHMat), "Model H Matrix:", 10, 10, 10);

	return 0;
}









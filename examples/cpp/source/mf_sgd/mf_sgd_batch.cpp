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


using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string datasetFileName = "../data/batch/mf_sgd.csv";

struct VPoint 
{
    int wPos;
    int hPos;
    double val;
};

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    // FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 // DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
	// internal numericType == HomogenNumericTable
	// To Do: use AOSNumericTable to install VPoints
    // load from dataSource or create a table with random values
    // dataSource.loadDataBlock();

    // AOS format input data 

    const size_t num_Train = 5;
    const size_t num_Test = 2;
    const size_t field_v = 3;

    const double learningRate = 0.05;
    const double lambda = 0.002;
    const int iteration = 10;
    const int threads = 40;

    VPoint points_Train[num_Train] =
    {
        {1, 2, 100.1},
        {2, 4, 200.2},
        {3, 5, 350.3},
        {4, 1, 470.4},
        {5, 2, 270.5}
    };

    VPoint points_Test[num_Test] =
    {
        {4, 3, 150.1},
        {2, 3, 343.2}
    };

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

    // dimension of model W and model H
    const long r_dim = 1000;
    const long row_num_w = 20000;
    const long col_num_w = r_dim;

    const long row_num_h = r_dim;
    const long col_num_h = 20000;

    /* Create an algorithm to compute mf_sgd decomposition */
    // use default template value: double and defaultSGD
    mf_sgd::Batch<> algorithm;

    // algorithm.input.set(mf_sgd::dataTrain, dataSource.getNumericTable());
    algorithm.input.set(mf_sgd::dataTrain, dataTable_Train);
    algorithm.input.set(mf_sgd::dataTest, dataTable_Test);

    algorithm.parameter.setParameter(learningRate, lambda, r_dim, row_num_w, col_num_h, iteration, threads);

    /* Compute mf_sgd decomposition */
    algorithm.compute();

    services::SharedPtr<mf_sgd::Result> res = algorithm.getResult();

    /* Print the results */
    printNumericTable(res->get(mf_sgd::resWMat), "Model W Matrix:", 10);
    printNumericTable(res->get(mf_sgd::resHMat), "Model H Matrix:");

    return 0;
}

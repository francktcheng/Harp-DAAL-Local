/* file: mf_sgd_ksnc.cpp */
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
!    C++ example of computing mf_sgd decomposition in the SNC mode of KNL 
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-mf_sgd_BATCH"></a>
 * \example mf_sgd_ksnc.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"
#include <cstdlib> 
#include <ctime> 
#include <time.h>
#include <stdlib.h> 

#include <cstring>
#include <fstream>
#include <iostream>
#include <exception>
#include <string>
#include <vector>

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

typedef float sgd_float;
// typedef double sgd_float;
typedef std::vector<mf_sgd::VPoint<sgd_float>*>* LineContainer; 

const int mpi_size = 4; //using the SNC-4 mode of KNL

// input dataset files in distributed mode 
string trainDataFile = "../../../data/distributed/movielens-snc-train/movielens-train0";
string testDataFile = "../../../data/distributed/movielens-snc-test/movielens-test0";

// string trainDataFile = "../../../data/distributed/yahoomusic-snc-train/yahoomusic-train0";
// string testDataFile = "../../../data/distributed/yahoomusic-snc-test/yahoomusic-test0";

// string trainDataFile = "../../../data/batch/yahoomusic-train.csv";
// string testDataFile = "../../../data/batch/yahoomusic-test.csv";
//
// string trainDataFile = "/home/langshichen/Lib/__release_lnx/daal/examples/data/batch/movielens-train.csv";
// string testDataFile = "/home/langshichen/Lib/__release_lnx/daal/examples/data/batch/movielens-test.csv";

// string trainDataFile = "/home/langshichen/Lib/__release_lnx/daal/examples/data/batch/yahoomusic-train.csv";
// string testDataFile = "/home/langshichen/Lib/__release_lnx/daal/examples/data/batch/yahoomusic-test.csv";

// read in binary file
typedef int mf_int;
typedef float mf_float;
typedef long long mf_long;

struct mf_problem
{
    mf_int m;
    mf_int n;
    mf_long nnz;
    mf_sgd::VPoint_bin *R;
};

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
     //--------------------- Start MPI process ------------------------------//
    MPI_Init(&argc,&argv);

    MPI_Status status;
    MPI_Request request;
   
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

    int NbProcs, rank;

    try
    {
        //check the num of mpi procs
        MPI_Comm_size(MPI_COMM_WORLD,&NbProcs);
        if (NbProcs != mpi_size)
            throw std::exception();
    }
    catch(const std::exception&)  // Consider using a custom exception type for intentional
    {                             
        std::cerr<<"The MPI does not create the correct number of process for SNC-4 mode"<<std::endl;
        return EXIT_FAILURE;
    }

    //get the mpi rank
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    // Create an algorithm to compute mf_sgd decomposition 
    // use default template value: double and defaultSGD
    // mf_sgd::KSNC<double, mf_sgd::defaultSGD> algorithm;
    mf_sgd::KSNC<sgd_float, mf_sgd::defaultSGD> algorithm;

    //read in distributed files by MPI procs
	mf_sgd::VPoint<sgd_float>* points_Train;
	mf_sgd::VPoint<sgd_float>* points_Test;
    
    struct timespec ts1;
	struct timespec ts2;
    long diff;
    double diff_ms;

    std::unordered_map<long, LineContainer> map_train;
    std::unordered_map<long, LineContainer> map_test;

    std::vector<long> vec_line_train;
    std::vector<long> vec_line_test;

	if (argc > 8)
	{
		// generate the dataset in parallel
		// // size of training dataset and test dataset
		// num_Train = row_num_w + 0.6*(row_num_w*col_num_h - row_num_w);
		// num_Test = 0.002*(row_num_w*col_num_h- row_num_w);
        //
		// // generate the Train and Test datasets
		// points_Train = new mf_sgd::VPoint<sgd_float>[num_Train];
		// points_Test = new mf_sgd::VPoint<sgd_float>[num_Test];
        //
		// algorithm.input.generate_points<sgd_float>(points_Train, num_Train, points_Test, num_Test, row_num_w, col_num_h);
        //
		// printf("Train set num of Points: %d\n", num_Train);
		// printf("Test set num of Points: %d\n", num_Test);
		// printf("Model W Rows: %d\n", row_num_w);
		// printf("Model H Columns: %d\n", col_num_h);
		// printf("Model Dimension: %d\n", r_dim);

	}
	else
	{
		// load the dataset file in parallel

        //----------------- Starting reading datasets from binary file --------------------------
        
        printf("Start loading Data into DAAL's NumericTable on rank: %d\n", rank);

        std::string distri_train_file = trainDataFile + std::to_string(rank);
        std::string distri_test_file = testDataFile + std::to_string(rank);
        long num_Train;
        long num_Test;

        //read in files and sorted in row id
        
        //load training data
        MPI_Barrier(MPI_COMM_WORLD);
	    clock_gettime(CLOCK_MONOTONIC, &ts1);

        algorithm.input.loadData(distri_train_file, map_train, num_Train,  &vec_line_train);

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC, &ts2);

        diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
	    diff_ms = (double)(diff)/1000000L;

        printf("Finish loading Train Data %d rows %d points into DAAL's NumericTable on rank: %d, using %f ms\n", map_train.size(), num_Train, rank, diff_ms);
        printf("Train Vec line num for rank: %d is %d\n", rank, vec_line_train.size());
        // algorithm.input.freeDistri(map_train);

        //load test data
        MPI_Barrier(MPI_COMM_WORLD);
	    clock_gettime(CLOCK_MONOTONIC, &ts1);

        algorithm.input.loadData(distri_test_file, map_test, num_Test, &vec_line_test);

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC, &ts2);

        diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
	    diff_ms = (double)(diff)/1000000L;

        printf("Finish loading Test Data %d rows and %d points into DAAL's NumericTable on rank: %d, using %f ms\n", map_test.size(), num_Test, rank, diff_ms);
        printf("Test Vec line num for rank: %d is %d\n", rank, vec_line_test.size());
        // algorithm.input.freeDistri(map_test);

        //debug check the value in map_test
        // for (std::unordered_map<long, LineContainer>::iterator it = map_test.begin(); it != map_test.end(); ++it) 
        // {
        //
        //     long first = it->first;
        //     LineContainer second = it->second; 
        //
        //     if (second->empty() == false)
        //     {
        //         std::cout<<"row id: "<<first<<" elements num: "<<second->size()<<std::endl;
        //     }
        //
        // }

        //----------------- End reading datasets from binary file --------------------------

        // points_Train = new mf_sgd::VPoint<sgd_float>[num_Train];
		// points_Test = new mf_sgd::VPoint<sgd_float>[num_Test];

	    // clock_gettime(CLOCK_MONOTONIC, &ts1);
        //
        // clock_gettime(CLOCK_MONOTONIC, &ts2);
        // diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
	    // diff_ms = (double)(diff)/1000000L;

        // printf("Finish Converting Sparse Data using %f ms\n", diff_ms);
        // printf("Train set num of Points: %d\n", num_Train);
		// printf("Test set num of Points: %d\n", num_Test);
		// printf("Model W Rows: %d\n", row_num_w);
		// printf("Model H Columns: %d\n", col_num_h);
		// printf("Model Dimension: %d\n", r_dim);
        //
        // delete[] train_binary;
        // delete[] test_binary;
        
	
	}

    // --------------------------------------- Start of shuffling the train and test data based on row_id ---------------------------------------
    
    //first allgather the number of rows from all ranks
    
    long* rows_num_train_send = new long[NbProcs];
    long* rows_num_test_send = new long[NbProcs];
     
    long* rows_num_train_recv = new long[NbProcs];
    long* rows_num_test_recv = new long[NbProcs];
   
    for(int j=0;j<NbProcs;j++)
    {
        rows_num_train_send[j] = map_train.size();
        rows_num_test_send[j] = map_test.size();
    }

    rows_num_train_recv[rank] = map_train.size();
    rows_num_test_recv[rank] = map_test.size();
    
    MPI_Alltoall(rows_num_train_send, 1, MPI_LONG, rows_num_train_recv, 1, MPI_LONG, MPI_COMM_WORLD);

    MPI_Alltoall(rows_num_test_send, 1, MPI_LONG, rows_num_test_recv, 1, MPI_LONG, MPI_COMM_WORLD);

    // check the result of rows_num_train/test_recv buffers
    for (int j = 0; j < NbProcs; j++) {
        printf("Train Rank: %d receive %d rows from rank %d\n", rank, rows_num_train_recv[j], j);
        printf("Test Rank: %d receive %d rows from rank %d\n", rank, rows_num_test_recv[j], j);
    }

    //send and receive row ids 
    MPI_Barrier(MPI_COMM_WORLD);

    long** recvBuf_train = new long*[NbProcs - 1];
    long** recvBuf_test = new long*[NbProcs - 1];

    if (rank == 0)
    {

        for(int j= 0; j< NbProcs-1; j++)
        {
            recvBuf_train[j] = new long[rows_num_train_recv[j+1]];
            recvBuf_test[j] = new long[rows_num_test_recv[j+1]];
        }

    }

    //for train set
    if (rank != 0)
    {
        const long* sendbuf_train = vec_line_train.data();
        int sendSize_train = vec_line_train.size();
        //send row_ids to root rank 0
        MPI_Send(sendbuf_train, sendSize_train, MPI_LONG, 0, 1, MPI_COMM_WORLD);
    }
    else
    {
        for(int j=0;j<NbProcs-1;j++)
          MPI_Recv(recvBuf_train[j], rows_num_train_recv[j+1], MPI_LONG, j+1, 1, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //for train set
    if (rank != 0)
    {
        const long* sendbuf_test = vec_line_test.data();
        int sendSize_test = vec_line_test.size();
        //send row_ids to root rank 0
        MPI_Send(sendbuf_test, sendSize_test, MPI_LONG, 0, 1, MPI_COMM_WORLD);
    }
    else
    {
        for(int j=0;j<NbProcs-1;j++)
          MPI_Recv(recvBuf_test[j], rows_num_test_recv[j+1], MPI_LONG, j+1, 1, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //check the recvBuf_train
    if (rank == 0)
    {
        for (int j = 0; j < 10; j++) 
        {
            printf("Rank 0: recvBuf_train[0][%d]: %d\n", j, recvBuf_train[0][j]);
            printf("Rank 0: recvBuf_test[0][%d]: %d\n", j, recvBuf_test[0][j]);
        }
    }

    //merge the row_ids by rank 0
    if (rank == 0)
    {

        std::unordered_map<long, long> Map_row_merge_train;
        long index_abs_train = 0;

        //first add rows of rank 0
        for(int j = 0; j<map_train.size();j++)
        {

            if ( Map_row_merge_train.find(vec_line_train[j]) == Map_row_merge_train.end())
            {
                Map_row_merge_train[vec_line_train[j]] = index_abs_train; 
                index_abs_train++;
            }

        }

        //add rows from other ranks
        for (int j = 0; j < NbProcs-1; j++) 
        {
            long* recvbuf_train = recvBuf_train[j];
            int recvbuf_train_size = rows_num_train_recv[j+1];

            for(int k= 0; k<recvbuf_train_size;k++)
            {

                if ( Map_row_merge_train.find(recvbuf_train[k]) == Map_row_merge_train.end())
                {
                     Map_row_merge_train[recvbuf_train[k]] = index_abs_train; 
                     index_abs_train++;
                }

            }
            
        }

        //debug check the size of rows
        printf("Train dataset: total rows: %d\n", Map_row_merge_train.size());

    }
    

    // --------------------------------------- End of shuffling the train and test data based on row_id ---------------------------------------

    /* Create a new dictionary and fill it with the information about data */
    // NumericTableDictionary newDict_Train(field_v);
    // NumericTableDictionary newDict_Test(field_v);

    /* Add a feature type to the dictionary */
    // newDict_Train[0].featureType = data_feature_utils::DAAL_CONTINUOUS;
    // newDict_Train[1].featureType = data_feature_utils::DAAL_CONTINUOUS;
    // newDict_Train[2].featureType = data_feature_utils::DAAL_CONTINUOUS;
    //
    // newDict_Test[0].featureType = data_feature_utils::DAAL_CONTINUOUS;
    // newDict_Test[1].featureType = data_feature_utils::DAAL_CONTINUOUS;
    // newDict_Test[2].featureType = data_feature_utils::DAAL_CONTINUOUS;
    //
    // services::SharedPtr<AOSNumericTable> dataTable_Train(new AOSNumericTable(points_Train, field_v, num_Train));
    // services::SharedPtr<AOSNumericTable> dataTable_Test(new AOSNumericTable(points_Test, field_v, num_Test));

    /* Assign the new dictionary to an existing numeric table */
    // dataTable_Train->setDictionary(&newDict_Train);
    // dataTable_Test->setDictionary(&newDict_Test);

    /* Add data to the numeric table */
    // dataTable_Train->setFeature<long> (0, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, wPos));
    // dataTable_Train->setFeature<long> (1, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, hPos));
    // dataTable_Train->setFeature<sgd_float> (2, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, val));

    /* Add data to the numeric table */
    // dataTable_Test->setFeature<long> (0, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, wPos));
    // dataTable_Test->setFeature<long> (1, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, hPos));
    // dataTable_Test->setFeature<sgd_float> (2, DAAL_STRUCT_MEMBER_OFFSET(mf_sgd::VPoint<sgd_float>, val));

	// debug
	// printNumericTable(dataTable_Train, "Train dataset:", 5, 3, 10);
	// printNumericTable(dataTable_Test, "Test dataset:", 5, 3, 10);

    // algorithm.input.set(mf_sgd::dataTrain, dataTable_Train);
    // algorithm.input.set(mf_sgd::dataTest, dataTable_Test);

    // algorithm.parameter.setParameter(learningRate, lambda, r_dim, row_num_w, col_num_h, iteration, threads, tbb_grainsize, isAvx512);

    /* Compute mf_sgd decomposition */
	// struct timespec ts1;
	// struct timespec ts2;

	// clock_gettime(CLOCK_MONOTONIC, &ts1);

    // algorithm.compute();

	// clock_gettime(CLOCK_MONOTONIC, &ts2);

	// long diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
	// double compute_time = (double)(diff)/1000000L;
	
    // services::SharedPtr<mf_sgd::Result> res = algorithm.getResult();
	// printf("Computation Time elapsed in ms: %f\n", compute_time);

    /* Print the results */
    // printNumericTable(res->get(mf_sgd::resWMat), "Model W Matrix:", 10, 10, 10);
    // printNumericTable(res->get(mf_sgd::resHMat), "Model H Matrix:", 10, 10, 10);

	// delete[] points_Train;
	// delete[] points_Test;

    MPI_Finalize();
    return 0;
}









/* file: mf_sgd_default_distri_impl.i */
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
//  Implementation of distributed mode mf_sgd method
//--
*/

#ifndef __MF_SGD_KERNEL_DISTRI_IMPL_I__
#define __MF_SGD_KERNEL_DISTRI_IMPL_I__

#include <time.h>
#include <math.h>       
#include <algorithm>
#include <cstdlib> 
#include <vector>
#include <iostream>
#include <cstdio> 
#include <algorithm>
#include <ctime>        
#include <omp.h>
#include <immintrin.h>

#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "threading.h"
#include "tbb/tick_count.h"
#include "task_scheduler_init.h"
#include "tbb/concurrent_hash_map.h"
#include "tbb/concurrent_vector.h"

#include "blocked_range.h"
#include "parallel_for.h"
#include "queuing_mutex.h"

#include "mf_sgd_default_impl.i"

using namespace tbb;
using namespace daal::internal;
using namespace daal::services::internal;


typedef queuing_mutex currentMutex_t;
typedef tbb::concurrent_hash_map<int, int> ConcurrentMap;
typedef tbb::concurrent_hash_map<int, std::vector<int> > ConcurrentVectorMap;

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
namespace internal
{
    
struct omp_task
{
    int _col_pos;
    int _len;
    int* _task_ids;
    omp_task(int col_pos, int len, int* task_ids): _col_pos(col_pos), _len(len), _task_ids(task_ids){}
};

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute(NumericTable** WPos, 
                                                      const NumericTable** HPos, 
                                                      const NumericTable** Val, 
                                                      NumericTable** WPosTest,
                                                      NumericTable** HPosTest, 
                                                      NumericTable** ValTest, 
                                                      NumericTable *r[], Parameter *parameter, int* col_ids)
{/*{{{*/

    /* retrieve members of parameter */
    /* const Parameter *parameter = static_cast<const Parameter *>(par); */
    long dim_w = parameter->_Dim_w;
    long dim_h = parameter->_Dim_h;
    int dim_set = Val[0]->getNumberOfRows();

    const int isTrain = parameter->_isTrain;
    const int isSGD2 = parameter->_sgd2;
    const int dim_r = parameter->_Dim_r;

    if (isSGD2 == 0)
    {
        /* ------------- Retrieve Data Set -------------*/
        FeatureMicroTable<int, readOnly, cpu> workflowW_ptr(WPos[0]);
        FeatureMicroTable<int, readOnly, cpu> workflowH_ptr(HPos[0]);
        FeatureMicroTable<interm, readOnly, cpu> workflow_ptr(Val[0]);

        int *workWPos = 0;
        workflowW_ptr.getBlockOfColumnValues(0, 0, dim_set, &workWPos);

        int *workHPos = 0;
        workflowH_ptr.getBlockOfColumnValues(0, 0, dim_set, &workHPos);

        interm *workV;
        workflow_ptr.getBlockOfColumnValues(0, 0, dim_set, &workV);

        /* ---------------- Retrieve Model W ---------------- */
        BlockMicroTable<interm, readWrite, cpu> mtWDataTable(r[0]);

        interm* mtWDataPtr = 0;
        mtWDataTable.getBlockOfRows(0, dim_w, &mtWDataPtr);

        /* ---------------- Retrieve Model H ---------------- */
        BlockMicroTable<interm, readWrite, cpu> mtHDataTable(r[1]);

        interm* mtHDataPtr = 0;
        mtHDataTable.getBlockOfRows(0, dim_h, &mtHDataPtr);

        if (isTrain == 1)
        {
            /* MF_SGDDistriKernel<interm, method, cpu>::compute_train(workWPos,workHPos,workV, dim_set, mtWDataPtr, mtHDataPtr, parameter);  */
            MF_SGDDistriKernel<interm, method, cpu>::compute_train_omp(workWPos,workHPos,workV, dim_set, mtWDataPtr, mtHDataPtr, parameter); 
        }
        else
        {
            /* ---------------- Retrieve RMSE for Test dataset --------- */
            interm* mtRMSEPtr = 0;
            BlockMicroTable<interm, readWrite, cpu> mtRMSETable(r[2]);
            mtRMSETable.getBlockOfRows(0, 1, &mtRMSEPtr);
            /* MF_SGDDistriKernel<interm, method, cpu>::compute_test(workWPos,workHPos,workV, dim_set, mtWDataPtr,mtHDataPtr, mtRMSEPtr, parameter); */
            MF_SGDDistriKernel<interm, method, cpu>::compute_test_omp(workWPos,workHPos,workV, dim_set, mtWDataPtr,mtHDataPtr, mtRMSEPtr, parameter);
        }

    }
    else if (isTrain == 1)
    {

        /* dim_set is the number of training points */
        FeatureMicroTable<int, readOnly, cpu> workflowW_ptr(WPos[0]);
        FeatureMicroTable<int, readOnly, cpu> workflowH_ptr(HPos[0]);
        FeatureMicroTable<interm, readOnly, cpu> workflow_ptr(Val[0]);

        dim_set =  Val[0]->getNumberOfRows();

        int *workWPos = 0;
        workflowW_ptr.getBlockOfColumnValues(0, 0, dim_set, &workWPos);

        int *workHPos = 0;
        workflowH_ptr.getBlockOfColumnValues(0, 0, dim_set, &workHPos);

        interm *workV;
        workflow_ptr.getBlockOfColumnValues(0, 0, dim_set, &workV);

        dim_w = r[3]->getNumberOfRows();
        dim_h = r[1]->getNumberOfRows();

        /* ---------------- Retrieve Model W ---------------- */
        BlockMicroTable<interm, readWrite, cpu> mtWDataTable(r[3]);

        interm* mtWDataPtr = 0;
        mtWDataTable.getBlockOfRows(0, dim_w, &mtWDataPtr);

        /* ---------------- Retrieve Model H ---------------- */
        BlockMicroTable<interm, readWrite, cpu> mtHDataTable(r[1]);

        interm* mtHDataPtr = 0;
        mtHDataTable.getBlockOfRows(0, dim_h, &mtHDataPtr);

        MF_SGDDistriKernel<interm, method, cpu>::compute_train2_omp(workWPos,workHPos,workV, dim_set, mtWDataPtr, mtHDataPtr, parameter, col_ids); 

    }
    else
    {
        /* isSGD2 test process */

        /* dim_set is the number of training points */
        FeatureMicroTable<int, readOnly, cpu> workflowW_ptr(WPosTest[0]);
        FeatureMicroTable<int, readOnly, cpu> workflowH_ptr(HPosTest[0]);
        FeatureMicroTable<interm, readOnly, cpu> workflow_ptr(ValTest[0]);

        dim_set =  ValTest[0]->getNumberOfRows();

        int *workWPos = 0;
        workflowW_ptr.getBlockOfColumnValues(0, 0, dim_set, &workWPos);

        int *workHPos = 0;
        workflowH_ptr.getBlockOfColumnValues(0, 0, dim_set, &workHPos);

        interm *workV;
        workflow_ptr.getBlockOfColumnValues(0, 0, dim_set, &workV);

        dim_w = r[3]->getNumberOfRows();
        dim_h = r[1]->getNumberOfRows();

        /* ---------------- Retrieve Model W ---------------- */
        BlockMicroTable<interm, readWrite, cpu> mtWDataTable(r[3]);

        interm* mtWDataPtr = 0;
        mtWDataTable.getBlockOfRows(0, dim_w, &mtWDataPtr);

        /* ---------------- Retrieve Model H ---------------- */
        BlockMicroTable<interm, readWrite, cpu> mtHDataTable(r[1]);

        interm* mtHDataPtr = 0;
        mtHDataTable.getBlockOfRows(0, dim_h, &mtHDataPtr);

        interm* mtRMSEPtr = 0;
        BlockMicroTable<interm, readWrite, cpu> mtRMSETable(r[2]);
        mtRMSETable.getBlockOfRows(0, 1, &mtRMSEPtr);

        MF_SGDDistriKernel<interm, method, cpu>::compute_test2_omp(workWPos,workHPos,workV, dim_set, mtWDataPtr,mtHDataPtr, mtRMSEPtr, parameter, col_ids);

    }

}/*}}}*/

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute_train(int* workWPos, 
                                                            int* workHPos, 
                                                            interm* workV, 
                                                            const int dim_set,
                                                            interm* mtWDataPtr, 
                                                            interm* mtHDataPtr, 
                                                            const Parameter *parameter)
{/*{{{*/

    /* retrieve members of parameter */
    const long dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    const double learningRate = parameter->_learningRate;
    const double lambda = parameter->_lambda;
    const int iteration = parameter->_iteration;
    const int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;

    const double ratio = parameter->_ratio;
    const int itr = parameter->_itr;
    double timeout = parameter->_timeout;

    /* create the mutex for WData and HData */
    services::SharedPtr<currentMutex_t> mutex_w(new currentMutex_t[dim_w]);
    services::SharedPtr<currentMutex_t> mutex_h(new currentMutex_t[dim_h]);

    /* ------------------- Starting TBB based Training  -------------------*/
    task_scheduler_init init(task_scheduler_init::deferred);

    if (thread_num != 0)
    {
        /* use explicitly specified num of threads  */
        init.initialize(thread_num);
    }
    else
    {
        /* use automatically generated threads by TBB  */
        init.initialize();
    }

    /* if ratio != 1, dim_ratio is the ratio of computed tasks */
    int dim_ratio = (int)(ratio*dim_set);

    /* step is the stride of choosing tasks in a rotated way */
    const int step = dim_set - dim_ratio;

    /* randomize the execution order */
    /* services::SharedPtr<int> random_index(new int[dim_set]);  */
    /* int* random_index_ptr = random_index.get(); */
    /* for(int j=0;j<dim_set;j++) */
    /*     random_index_ptr[j] = j; */

    /* std::srand ( unsigned (std::time(0))); */
    /* std::random_shuffle(random_index_ptr,random_index_ptr + dim_set - 1); */

    MFSGDTBB<interm, cpu> mfsgd(mtWDataPtr, mtHDataPtr, workWPos, workHPos, workV, dim_r, learningRate, lambda, mutex_w.get(), mutex_h.get(), Avx_explicit, step, dim_set);
    mfsgd.setItr(itr);
    mfsgd.setTimeStart(tbb::tick_count::now());
    mfsgd.setTimeOut(timeout);

    /* test randomize order */
    /* mfsgd.setOrder(random_index_ptr); */

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double train_time = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    /* training MF-SGD */
    if (tbb_grainsize != 0)
        parallel_for(blocked_range<int>(0, dim_ratio, tbb_grainsize), mfsgd);
    else
        parallel_for(blocked_range<int>(0, dim_ratio), mfsgd, auto_partitioner());

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    /* get the training time for each iteration */
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    train_time += (double)(diff)/1000000L;

    init.terminate();
    std::printf("Training time this iteration: %f\n", train_time);
    std::fflush(stdout);

    return;

}/*}}}*/

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute_train_omp(int* workWPos, 
                                                                int* workHPos, 
                                                                interm* workV, 
                                                                const int dim_set,
                                                                interm* mtWDataPtr, 
                                                                interm* mtHDataPtr, 
                                                                const Parameter *parameter)
{/*{{{*/

#ifdef _OPENMP

    /* retrieve members of parameter */
    const long dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    const double learningRate = parameter->_learningRate;
    const double lambda = parameter->_lambda;
    const int iteration = parameter->_iteration;
    int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;

    const double ratio = parameter->_ratio;
    const int itr = parameter->_itr;
    double timeout = parameter->_timeout;

    /* create the mutex for WData and HData */
    /* services::SharedPtr<currentMutex_t> mutex_w(new currentMutex_t[dim_w]); */
    /* services::SharedPtr<currentMutex_t> mutex_h(new currentMutex_t[dim_h]); */

    /* omp_lock_t* mutex_w_ptr = mutex_w.get(); */
    /* for(int j=0; j<dim_w;j++) */
         /* omp_init_lock(&(mutex_w_ptr[j])); */

    /* omp_lock_t* mutex_h_ptr = mutex_h.get(); */
    /* for(int j=0; j<dim_h;j++) */
         /* omp_init_lock(&(mutex_h_ptr[j])); */

    /* ------------------- Starting OpenMP based Training  -------------------*/
    int num_thds_max = omp_get_max_threads();
    std::printf("Max threads number: %d\n", num_thds_max);
    std::fflush(stdout);

    if (thread_num == 0)
        thread_num = num_thds_max;

    /* if ratio != 1, dim_ratio is the ratio of computed tasks */
    int dim_ratio = (int)(ratio*dim_set);

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double train_time = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    /* training MF-SGD */
    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(int k=0;k<dim_ratio;k++)
    {

        interm *WMat = 0;
        interm *HMat = 0;

        interm Mult = 0;
        interm Err = 0;
        interm WMatVal = 0;
        interm HMatVal = 0;
        int p = 0;

        interm* mtWDataLocal = mtWDataPtr;
        interm* mtHDataLocal = mtHDataPtr;

        int* workWPosLocal = workWPos;
        int* workHPosLocal = workHPos;
        interm* workVLocal = workV;

        long Dim = dim_r;
        interm learningRateLocal = learningRate;
        interm lambdaLocal = lambda;

        WMat = mtWDataLocal + workWPosLocal[k]*Dim;
        HMat = mtHDataLocal + workHPosLocal[k]*Dim;


        for(p = 0; p<Dim; p++)
            Mult += (WMat[p]*HMat[p]);


        Err = workVLocal[k] - Mult;

        for(p = 0;p<Dim;p++)
        {

            WMatVal = WMat[p];
            HMatVal = HMat[p];

            WMat[p] = WMatVal + learningRateLocal*(Err*HMatVal - lambdaLocal*WMatVal);
            HMat[p] = HMatVal + learningRateLocal*(Err*WMatVal - lambdaLocal*HMatVal);

        }


    }

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    /* get the training time for each iteration */
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    train_time += (double)(diff)/1000000L;

    /* init.terminate(); */
    std::printf("Training time this iteration: %f\n", train_time);
    std::fflush(stdout);

#else

    std::printf("Error: OpenMP module is not enabled\n");
    std::fflush(stdout);

#endif

    return;

}/*}}}*/

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute_train2_omp(int* workWPos, 
                                                                int* workHPos, 
                                                                interm* workV, 
                                                                const int dim_set,
                                                                interm* mtWDataPtr, 
                                                                interm* mtHDataPtr, 
                                                                const Parameter *parameter,
                                                                int* col_ids)
{/*{{{*/

#ifdef _OPENMP

    /* retrieve members of parameter */
    const int dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    const double learningRate = parameter->_learningRate;
    const double lambda = parameter->_lambda;
    const int iteration = parameter->_iteration;
    int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;

    const double ratio = parameter->_ratio;
    const int itr = parameter->_itr;
    double timeout = parameter->_timeout;

    /* ------------------- Starting OpenMP based Training  -------------------*/
    int num_thds_max = omp_get_max_threads();
    std::printf("Max threads number: %d\n", num_thds_max);
    std::fflush(stdout);

    if (thread_num == 0)
        thread_num = num_thds_max;

    /* if ratio != 1, dim_ratio is the ratio of computed tasks */
    /* int dim_ratio = (int)(ratio*dim_set); */

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double train_time = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    /* shared vars */
    /* ConcurrentMap* map_w = parameter->_wMat_map; */
    ConcurrentMap* map_h = parameter->_hMat_map;
    ConcurrentVectorMap* map_train = parameter->_train_map;

    //store the col pos of each sub-task queue
    /* std::vector<int>* task_queue_colPos = new std::vector<int>(); */

    //store the size of each sub-task queue
    /* std::vector<int>* task_queue_size = new std::vector<int>(); */

    //store the pointer to each sub-task queue
    /* std::vector<int*>* task_queue_ids = new std::vector<int*>(); */
    std::vector<omp_task*>* task_queue = new std::vector<omp_task*>();

    const int tasks_queue_len = 50;

    for(int k=0;k<dim_h;k++)
    {
        int col_id = col_ids[k];
        int col_pos = -1;
        ConcurrentMap::accessor pos_h; 
        if (map_h->find(pos_h, col_id))
            col_pos = pos_h->second;
        else
            continue;

        ConcurrentVectorMap::accessor pos_train; 
        std::vector<int>* sub_tasks_ptr = NULL;
        if (map_train->find(pos_train, col_id))
        {
             sub_tasks_ptr = &(pos_train->second);
        }

        if (sub_tasks_ptr != NULL)
        {
            int tasks_size = (int)sub_tasks_ptr->size();
            int itr = 0; 

            while (((itr+1)*tasks_queue_len) <= tasks_size)
            {
                task_queue->push_back(new omp_task(col_pos, tasks_queue_len, &(*sub_tasks_ptr)[itr*tasks_queue_len]));
                //task_queue_colPos->push_back(col_pos);
                //task_queue_size->push_back(tasks_queue_len);
                //task_queue_ids->push_back(&(*sub_tasks_ptr)[itr*tasks_queue_len]);
                itr++;
            }

            //add the last sub task queue
            int residue = tasks_size - itr*tasks_queue_len;
            if (residue > 0)
            {
                task_queue->push_back(new omp_task(col_pos, residue, &(*sub_tasks_ptr)[itr*tasks_queue_len]));
                //task_queue_colPos->push_back(col_pos);
                //task_queue_size->push_back(residue);
                //task_queue_ids->push_back(&(*sub_tasks_ptr)[itr*tasks_queue_len]);
            }
        }

    }

    //int task_queues_num = (int)task_queue_ids->size();
    //int* queue_cols_ptr = &(*task_queue_colPos)[0];
    //int* queue_size_ptr = &(*task_queue_size)[0];
    //int** queue_ids_ptr = &(*task_queue_ids)[0];

    //shuffle the tasks
    //std::random_shuffle(task_queue->begin(), task_queue->end());

    int task_queues_num = (int)task_queue->size();
    omp_task** task_queue_array = &(*task_queue)[0]; 

    std::printf("Col num: %ld, Tasks num: %d\n", dim_h, task_queues_num);
    std::fflush(stdout);

    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(int k=0;k<task_queues_num;k++)
    {
        int* workWPosLocal = workWPos; 
        int* workHPosLocal = workHPos; 
        interm* workVLocal = workV; 

        interm *WMat = 0; 
        interm HMat[dim_r];

        interm Mult = 0;
        interm Err = 0;
        interm WMatVal = 0;
        interm HMatVal = 0;
        int p = 0;

        interm* mtWDataLocal = mtWDataPtr;
        interm* mtHDataLocal = mtHDataPtr + 1; // consider the sentinel element 

        int stride_w = dim_r;
        int stride_h = dim_r + 1; // h matrix has a sentinel as the first element of each row 

        interm learningRateLocal = learningRate;
        interm lambdaLocal = lambda;

        omp_task* elem = task_queue_array[k];
        //int col_pos = queue_cols_ptr[k];
        //int squeue_size = queue_size_ptr[k];
        //int* ids_ptr = queue_ids_ptr[k];
        int col_pos = elem->_col_pos;
        int squeue_size = elem->_len;
        int* ids_ptr = elem->_task_ids;

        /* HMat = mtHDataLocal + col_pos*stride_h; */
        //data copy 
        memcpy(HMat, mtHDataLocal+col_pos*stride_h, dim_r*sizeof(interm));

        for(int j=0;j<squeue_size;j++)
        {
            int data_id = ids_ptr[j];
            int row_pos = workWPosLocal[data_id];
            Mult = 0;
            Err = 0;

            WMat = mtWDataLocal + row_pos*stride_w;

            for(p = 0; p<dim_r; p++)
                Mult += (WMat[p]*HMat[p]);

            Err = workVLocal[data_id] - Mult;

            for(p = 0;p<dim_r;p++)
            {
                WMatVal = WMat[p];
                HMatVal = HMat[p];

                WMat[p] = WMatVal + learningRateLocal*(Err*HMatVal - lambdaLocal*WMatVal);
                HMat[p] = HMatVal + learningRateLocal*(Err*WMatVal - lambdaLocal*HMatVal);

            }

        }

        memcpy(mtHDataLocal+col_pos*stride_h, HMat, dim_r*sizeof(interm));
        delete elem;

    }


    //delete task_queue_colPos;
    //delete task_queue_size;
    //delete task_queue_ids;
    delete task_queue;

    /* training MF-SGD */
    /* for(int k=0;k<dim_set;k++) */
    /* std::vector<int>* task_queue = new std::vector<int>(); */
    /* long total_tasks_num = 0; */
    /* //loop over col_ids to get the total number of tasks */
    /* for(int k=0;k<dim_h;k++) */
    /* { */
    /*     int col_id = col_ids[k]; */
    /*     ConcurrentVectorMap::accessor pos_train;  */
    /*     if (map_train->find(pos_train, col_id)) */
    /*     { */
    /*         total_tasks_num += pos_train->second.size(); */
    /*     } */
    /* } */
    /*  */
    /* int* total_tasks = (int*)calloc(total_tasks_num, sizeof(int)); */
    /* int* total_tasks_colPos = (int*)calloc(total_tasks_num, sizeof(int)); */
    /*  */
    /* //loop over col_ids to copy the tasks ids */
    /* int copy_pos = 0; */
    /* for(int k=0;k<dim_h;k++) */
    /* { */
    /*     int col_id = col_ids[k]; */
    /*     int seg_size = 0; */
    /*     int col_pos = -1; */
    /*     ConcurrentVectorMap::accessor pos_train;  */
    /*     if (map_train->find(pos_train, col_id)) */
    /*     { */
    /*         seg_size = (int)pos_train->second.size(); */
    /*         memcpy(&total_tasks[copy_pos], &(pos_train->second)[0], seg_size*sizeof(int)); */
    /*          */
    /*         ConcurrentMap::accessor pos_h; */
    /*         if (map_h->find(pos_h, col_id)) */
    /*         { */
    /*             col_pos = pos_h->second; */
    /*         } */
    /*  */
    /*         for(int l=0;l<seg_size;l++) */
    /*             total_tasks_colPos[copy_pos+l] = col_pos; */
    /*   */
    /*         copy_pos += seg_size; */
    /*     } */
    /*  */
    /* } */

    /* std::printf("Training Points Number: %ld\n", total_tasks_num); */
    /* std::fflush(stdout); */

    /* #pragma omp parallel for schedule(guided) num_threads(thread_num)  */
    /* for(int k=0;k<total_tasks_num;k++) */
    /* { */
    /*  */
    /*     int* workWPosLocal = workWPos;  */
    /*     int* workHPosLocal = workHPos;  */
    /*     interm* workVLocal = workV;  */
    /*  */
    /*     interm *WMat = 0;  */
    /*     interm *HMat = 0; */
    /*  */
    /*     interm Mult = 0; */
    /*     interm Err = 0; */
    /*     interm WMatVal = 0; */
    /*     interm HMatVal = 0; */
    /*     int p = 0; */
    /*  */
    /*     interm* mtWDataLocal = mtWDataPtr; */
    /*     interm* mtHDataLocal = mtHDataPtr + 1; // consider the sentinel element  */
    /*  */
    /*     int stride_w = dim_r; */
    /*     int stride_h = dim_r + 1; // h matrix has a sentinel as the first element of each row  */
    /*  */
    /*     interm learningRateLocal = learningRate; */
    /*     interm lambdaLocal = lambda; */
    /*  */
    /*     int task_id = total_tasks[k]; */
    /*      */
    /*     int col_pos = total_tasks_colPos[k]; */
    /*  */
    /*     int row_pos = workWPosLocal[task_id]; */
    /*     WMat = mtWDataLocal + row_pos*stride_w; */
    /*     HMat = mtHDataLocal + col_pos*stride_h; */
    /*  */
    /*     for(p = 0; p<dim_r; p++) */
    /*         Mult += (WMat[p]*HMat[p]); */
    /*  */
    /*     Err = workVLocal[task_id] - Mult; */
    /*  */
    /*     for(p = 0;p<dim_r;p++) */
    /*     { */
    /*         WMatVal = WMat[p]; */
    /*         HMatVal = HMat[p]; */
    /*  */
    /*         WMat[p] = WMatVal + learningRateLocal*(Err*HMatVal - lambdaLocal*WMatVal); */
    /*         HMat[p] = HMatVal + learningRateLocal*(Err*WMatVal - lambdaLocal*HMatVal); */
    /*  */
    /*     } */
    /*  */
    /* } */

    /* free(total_tasks); */
    /* free(total_tasks_colPos); */

    /* #pragma omp parallel for schedule(guided) num_threads(thread_num)  */
    /* for(int k=0;k<dim_h;k++) */
    /* { */
    /*      */
    /*     int* workWPosLocal = workWPos;  */
    /*     int* workHPosLocal = workHPos;  */
    /*     interm* workVLocal = workV;  */
    /*  */
    /*     interm *WMat = 0;  */
    /*     interm *HMat = 0; */
    /*  */
    /*     interm Mult = 0; */
    /*     interm Err = 0; */
    /*     interm WMatVal = 0; */
    /*     interm HMatVal = 0; */
    /*     int p = 0; */
    /*  */
    /*     interm* mtWDataLocal = mtWDataPtr; */
    /*     interm* mtHDataLocal = mtHDataPtr + 1; // consider the sentinel element  */
    /*  */
    /*     int stride_w = dim_r; */
    /*     int stride_h = dim_r + 1; // h matrix has a sentinel as the first element of each row  */
    /*  */
    /*     interm learningRateLocal = learningRate; */
    /*     interm lambdaLocal = lambda; */
    /*  */
    /*     int col_id = col_ids[k]; */
    /*     int col_pos = -1; */
    /*     ConcurrentMap::accessor posH;  */
    /*     if (map_h->find(posH, col_id)) */
    /*         col_pos = posH->second; */
    /*  */
    /*     ConcurrentVectorMap::accessor pos_train;  */
    /*     if (map_train->find(pos_train, col_id) && col_pos != -1) */
    /*     { */
    /*  */
    /*         std::vector<int>* cols_ptr = &(pos_train->second); */
    /*  */
    /*         HMat = mtHDataLocal + col_pos*stride_h; */
    /*         for(std::vector<int>::iterator it = cols_ptr->begin(); it != cols_ptr->end(); ++it)  */
    /*         { */
    /*             int train_pos = (*it); */
    /*             int row_pos = workWPosLocal[train_pos]; */
    /*             WMat = mtWDataLocal + row_pos*stride_w; */
    /*  */
    /*             Mult = 0; */
    /*             Err = 0; */
    /*             for(p = 0; p<dim_r; p++) */
    /*                 Mult += (WMat[p]*HMat[p]); */
    /*  */
    /*             Err = workVLocal[train_pos] - Mult; */
    /*  */
    /*             for(p = 0;p<dim_r;p++) */
    /*             { */
    /*                 WMatVal = WMat[p]; */
    /*                 HMatVal = HMat[p]; */
    /*  */
    /*                 WMat[p] = WMatVal + learningRateLocal*(Err*HMatVal - lambdaLocal*WMatVal); */
    /*                 HMat[p] = HMatVal + learningRateLocal*(Err*WMatVal - lambdaLocal*HMatVal); */
    /*  */
    /*             } */
    /*  */
    /*         } */
    /*  */
    /*     } */
    /*  */
    /* } */

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    /* get the training time for each iteration */
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    train_time += (double)(diff)/1000000L;

    /* init.terminate(); */
    std::printf("Training time this iteration: %f\n", train_time);
    std::fflush(stdout);

#else

    std::printf("Error: OpenMP module is not enabled\n");
    std::fflush(stdout);

#endif

    return;

}/*}}}*/

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute_test(int* workWPos, 
                                                            int* workHPos, 
                                                            interm* workV, 
                                                            const int dim_set,
                                                            interm* mtWDataPtr, 
                                                            interm* mtHDataPtr, 
                                                            interm* mtRMSEPtr,
                                                            const Parameter *parameter)
{/*{{{*/

    /* retrieve members of parameter */
    const long dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    const int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;

    /* create the mutex for WData and HData */
    services::SharedPtr<currentMutex_t> mutex_w(new currentMutex_t[dim_w]);
    services::SharedPtr<currentMutex_t> mutex_h(new currentMutex_t[dim_h]);

    /* RMSE value for test dataset */
    services::SharedPtr<interm> testRMSE(new interm[dim_set]);

    /* ------------------- Starting TBB based Training  -------------------*/
    task_scheduler_init init(task_scheduler_init::deferred);

    if (thread_num != 0)
    {
        /* use explicitly specified num of threads  */
        init.initialize(thread_num);
    }
    else
    {
        /* use automatically generated threads by TBB  */
        init.initialize();
    }

    MFSGDTBB_TEST<interm, cpu> mfsgd_test(mtWDataPtr, mtHDataPtr, workWPos, workHPos, workV, dim_r, testRMSE.get(), mutex_w.get(), mutex_h.get(), Avx_explicit);

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double test_time = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    /* test dataset compute RMSE for MF-SGD */
    if (tbb_grainsize != 0)
        parallel_for(blocked_range<int>(0, dim_set, tbb_grainsize), mfsgd_test);
    else
        parallel_for(blocked_range<int>(0, dim_set), mfsgd_test, auto_partitioner());

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    test_time += (double)(diff)/1000000L;

    init.terminate();

    interm totalRMSE = 0;
    interm* testRMSE_ptr = testRMSE.get();

    for(int k=0;k<dim_set;k++)
        totalRMSE += testRMSE_ptr[k];

    mtRMSEPtr[0] = totalRMSE;

    return;

}/*}}}*/

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute_test_omp(int* workWPos, 
                                                               int* workHPos, 
                                                               interm* workV, 
                                                               const int dim_set,
                                                               interm* mtWDataPtr, 
                                                               interm* mtHDataPtr, 
                                                               interm* mtRMSEPtr,
                                                               const Parameter *parameter)
{/*{{{*/

#ifdef _OPENMP

    /* retrieve members of parameter */
    const long dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;

    /* create the mutex for WData and HData */
    services::SharedPtr<omp_lock_t> mutex_w(new omp_lock_t[dim_w]);
    services::SharedPtr<omp_lock_t> mutex_h(new omp_lock_t[dim_h]);

    /* RMSE value for test dataset */
    services::SharedPtr<interm> testRMSE(new interm[dim_set]);

    omp_lock_t* mutex_w_ptr = mutex_w.get();
    for(int j=0; j<dim_w;j++)
         omp_init_lock(&(mutex_w_ptr[j]));

    omp_lock_t* mutex_h_ptr = mutex_h.get();
    for(int j=0; j<dim_h;j++)
         omp_init_lock(&(mutex_h_ptr[j]));

    /* ------------------- Starting OpenMP based testing  -------------------*/
    int num_thds_max = omp_get_max_threads();
    std::printf("Max threads number: %d\n", num_thds_max);
    std::fflush(stdout);

    if (thread_num == 0)
        thread_num = num_thds_max;

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double test_time = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(int k = 0; k<dim_set; k++)
    {

        interm *WMat = 0;
        interm *HMat = 0;

        interm* mtWDataLocal = mtWDataPtr;
        interm* mtHDataLocal = mtHDataPtr;

        int* testWPosLocal = workWPos;
        int* testHPosLocal = workHPos;

        interm* testVLocal = workV;

        long Dim = dim_r;
        interm* testRMSELocal = testRMSE.get();

        int p = 0;
        interm Mult = 0;
        interm Err = 0;

        if (testWPosLocal[k] != -1 && testHPosLocal[k] != -1)
        {

            omp_set_lock(&(mutex_w_ptr[testWPosLocal[k]]));
            omp_set_lock(&(mutex_h_ptr[testHPosLocal[k]]));

            WMat = mtWDataLocal + testWPosLocal[k]*Dim;
            HMat = mtHDataLocal + testHPosLocal[k]*Dim;

            for(p = 0; p<Dim; p++)
                Mult += (WMat[p]*HMat[p]);

            Err = testVLocal[k] - Mult;

            testRMSELocal[k] = Err*Err;

            omp_unset_lock(&(mutex_w_ptr[testWPosLocal[k]]));
            omp_unset_lock(&(mutex_h_ptr[testHPosLocal[k]]));

        }
        else
            testRMSELocal[k] = 0;

    }

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    test_time += (double)(diff)/1000000L;

    interm totalRMSE = 0;
    interm* testRMSE_ptr = testRMSE.get();

    for(int k=0;k<dim_set;k++)
        totalRMSE += testRMSE_ptr[k];

    mtRMSEPtr[0] = totalRMSE;

#else

    std::printf("Error: OpenMP module is not enabled\n");
    std::fflush(stdout);

#endif

    return;

}/*}}}*/

template <typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
void MF_SGDDistriKernel<interm, method, cpu>::compute_test2_omp(int* workWPos, 
                                                               int* workHPos, 
                                                               interm* workV, 
                                                               const int dim_set,
                                                               interm* mtWDataPtr, 
                                                               interm* mtHDataPtr, 
                                                               interm* mtRMSEPtr,
                                                               Parameter *parameter,
                                                               int* col_ids)

{/*{{{*/

#ifdef _OPENMP

    /* retrieve members of parameter */
    const int dim_r = parameter->_Dim_r;
    const long dim_w = parameter->_Dim_w;
    const long dim_h = parameter->_Dim_h;
    int thread_num = parameter->_thread_num;
    const int tbb_grainsize = parameter->_tbb_grainsize;
    const int Avx_explicit = parameter->_Avx_explicit;

    /* create the mutex for WData and HData */
    services::SharedPtr<omp_lock_t> mutex_w(new omp_lock_t[dim_w]);
    services::SharedPtr<omp_lock_t> mutex_h(new omp_lock_t[dim_h]);

    /* RMSE value for test dataset */
    /* services::SharedPtr<interm> testRMSE(new interm[dim_set]); */
    /* interm* testRMSELocal = testRMSE.get(); */

    omp_lock_t* mutex_w_ptr = mutex_w.get();
    for(int j=0; j<dim_w;j++)
         omp_init_lock(&(mutex_w_ptr[j]));

    omp_lock_t* mutex_h_ptr = mutex_h.get();
    for(int j=0; j<dim_h;j++)
         omp_init_lock(&(mutex_h_ptr[j]));

    /* ------------------- Starting OpenMP based testing  -------------------*/
    int num_thds_max = omp_get_max_threads();
    std::printf("Max threads number: %d\n", num_thds_max);
    std::fflush(stdout);

    if (thread_num == 0)
        thread_num = num_thds_max;

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double test_time = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    /* shared vars */
    /* ConcurrentMap* map_w = parameter->_wMat_map; */
    ConcurrentMap* map_h = parameter->_hMat_map;
    ConcurrentVectorMap* map_test = parameter->_test_map;

    //store the col pos of each sub-task queue
    std::vector<int>* task_queue_colPos = new std::vector<int>();

    //store the size of each sub-task queue
    std::vector<int>* task_queue_size = new std::vector<int>();

    //store the pointer to each sub-task queue
    std::vector<int*>* task_queue_ids = new std::vector<int*>();
    
    const int tasks_queue_len = 20;

    for(int k=0;k<dim_h;k++)
    {
        int col_id = col_ids[k];
        int col_pos = -1;
        ConcurrentMap::accessor pos_h; 
        if (map_h->find(pos_h, col_id))
            col_pos = pos_h->second;
        else
            continue;

        ConcurrentVectorMap::accessor pos_test; 
        std::vector<int>* sub_tasks_ptr = NULL;
        if (map_test->find(pos_test, col_id))
        {
             sub_tasks_ptr = &(pos_test->second);
        }

        if (sub_tasks_ptr != NULL)
        {
            int tasks_size = (int)sub_tasks_ptr->size();
            int itr = 0; 

            while (((itr+1)*tasks_queue_len) <= tasks_size)
            {
                task_queue_colPos->push_back(col_pos);
                task_queue_size->push_back(tasks_queue_len);
                task_queue_ids->push_back(&(*sub_tasks_ptr)[itr*tasks_queue_len]);
                itr++;
            }

            //add the last sub task queue
            int residue = tasks_size - itr*tasks_queue_len;
            if (residue > 0)
            {
                task_queue_colPos->push_back(col_pos);
                task_queue_size->push_back(residue);
                task_queue_ids->push_back(&(*sub_tasks_ptr)[itr*tasks_queue_len]);
            }
        }

    }

    int task_queues_num = (int)task_queue_ids->size();
    int* queue_cols_ptr = &(*task_queue_colPos)[0];
    int* queue_size_ptr = &(*task_queue_size)[0];
    int** queue_ids_ptr = &(*task_queue_ids)[0];

    std::printf("Col num: %ld, Test Tasks num: %d\n", dim_h, task_queues_num);
    std::fflush(stdout);

    //RMSE value from computed points
    interm totalRMSE = 0;
    
    //the effective computed num of test V points
    int numTestV = 0;

    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(int k=0;k<task_queues_num;k++)
    {
        int* workWPosLocal = workWPos; 
        int* workHPosLocal = workHPos; 
        interm* workVLocal = workV; 

        interm *WMat = 0; 
        interm HMat[dim_r];

        interm Mult = 0;
        interm Err = 0;
        interm WMatVal = 0;
        interm HMatVal = 0;
        int p = 0;

        interm* mtWDataLocal = mtWDataPtr;
        interm* mtHDataLocal = mtHDataPtr + 1; // consider the sentinel element 

        int stride_w = dim_r;
        int stride_h = dim_r + 1; // h matrix has a sentinel as the first element of each row 

        int col_pos = queue_cols_ptr[k];
        int squeue_size = queue_size_ptr[k];
        int* ids_ptr = queue_ids_ptr[k];


        omp_set_lock(&(mutex_h_ptr[col_pos]));

        //---------- copy hmat data ---------------
        memcpy(HMat, mtHDataLocal+col_pos*stride_h, dim_r*sizeof(interm));

        omp_unset_lock(&(mutex_h_ptr[col_pos]));

        for(int j=0;j<squeue_size;j++)
        {
            int data_id = ids_ptr[j];
            int row_pos = workWPosLocal[data_id];
            if (row_pos < 0)
                continue;

            Mult = 0;

            omp_set_lock(&(mutex_w_ptr[row_pos]));
            WMat = mtWDataLocal + row_pos*stride_w;

            for(p = 0; p<dim_r; p++)
                Mult += (WMat[p]*HMat[p]);

            omp_unset_lock(&(mutex_w_ptr[row_pos]));

            Err = workVLocal[data_id] - Mult;

            #pragma omp atomic
            totalRMSE += (Err*Err);

            #pragma omp atomic
            numTestV++;

        }


    }

    delete task_queue_colPos;
    delete task_queue_size;
    delete task_queue_ids;

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    test_time += (double)(diff)/1000000L;

    mtRMSEPtr[0] = totalRMSE;

    parameter->setTestV(numTestV);

    std::printf("local RMSE value: %f, test time: %f\n", totalRMSE, test_time);
    std::fflush(stdout);

#else

    std::printf("Error: OpenMP module is not enabled\n");
    std::fflush(stdout);

#endif

    return;

}/*}}}*/

} // namespace daal::internal
}
}
} // namespace daal

#endif

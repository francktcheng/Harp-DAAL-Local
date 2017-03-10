/* file: mf_sgd_dense_default_kernel.h */
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
//  Declaration of template function that calculate mf_sgds.
//--
*/

#ifndef __MF_SGD_FPK_H__
#define __MF_SGD_FPK_H__

#include "task_scheduler_init.h"
#include "blocked_range.h"
#include "parallel_for.h"
#include "queuing_mutex.h"
#include "numeric_table.h"
#include "kernel.h"
#include <cstdlib> 
#include <cstdio> 
#include <assert.h>
#include <random>
#include <omp.h>

#include "service_rng.h"
#include "services/daal_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "mf_sgd_batch.h"
#include "tbb/tick_count.h"

using namespace tbb;
using namespace daal::data_management;

typedef queuing_mutex currentMutex_t;

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{

typedef tbb::concurrent_hash_map<int, int> ConcurrentModelMap;
typedef tbb::concurrent_hash_map<int, std::vector<int> > ConcurrentDataMap;

namespace internal
{

/**
 * @brief computation kernel for mf_sgd batch mode 
 *
 * @tparam interm
 * @tparam method
 * @tparam cpu
 */
template<typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
class MF_SGDBatchKernel : public Kernel
{
public:

    /**
     * @brief compute and update W, H model by Training data
     *
     * @param[in] TrainSet  train dataset stored in an AOSNumericTable
     * @param[in] TestSet  test dataset stored in an AOSNumericTable
     * @param[in,out] r[] model W and H
     * @param[in] par
     */
    void compute(const NumericTable** TrainSet, const NumericTable** TestSet,
                 NumericTable *r[], const daal::algorithms::Parameter *par);

    /**
     * @brief regroup the train dataset points by their row id
     *
     * @param trainWPos
     * @param trainHPos
     * @param trainV
     * @param train_num
     * @param parameter
     */
    void reorder(int* trainWPos, int* trainHPos, interm* trainV, const int train_num, const Parameter *parameter);

    // a multi-threading version of compute implemented by TBB 
    void compute_thr(int* trainWPos, int* trainHPos, interm* trainV, const int train_num,
                     int* testWPos, int* testHPos, interm* testV, const int test_num,
                     interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter);

    // a multi-threading version of compute implemented by OpenMP 
    void compute_openmp(int* trainWPos, int* trainHPos, interm* trainV, const int train_num,
                        int* testWPos, int* testHPos, interm* testV, const int test_num,
                        interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter);

    // a multi-threading version of compute implemented by TBB with reordered training dataset points 
    void compute_thr_reordered(int* testWPos, int* testHPos, interm* testV, const int test_num,
                               interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter);

    // a multi-threading version of compute implemented by OpenMP with reordered training dataset points 
    void compute_openmp_reordered(int* testWPos, int* testHPos, interm* testV, const int test_num,
                               interm* mtWDataPtr, interm* mtHDataPtr, const Parameter *parameter);

    void free_reordered();

private:

    services::SharedPtr<std::vector<int> > _trainWQueue;
    services::SharedPtr<std::vector<int> > _trainLQueue;
    services::SharedPtr<std::vector<int*> > _trainHQueue;
    services::SharedPtr<std::vector<interm*> > _trainVQueue;

};

/**
 * @brief computation kernel for mf_sgd distributed mode
 *
 * @tparam interm
 * @tparam method
 * @tparam cpu
 */
template<typename interm, daal::algorithms::mf_sgd::Method method, CpuType cpu>
class MF_SGDDistriKernel : public Kernel
{
public:

    /**
     * @brief compute and update W, H model by Training data
     *
     * @param[in] WPos row id of point in W model, stored in HomogenNumericTable 
     * @param[in] HPos col id of point in H model, stored in HomogenNumericTable
     * @param[in] Val  value of point, stored in HomogenNumericTable
     * @param[in,out] r[] model W and H
     * @param[in] par
     */
    void compute(NumericTable** WPos, NumericTable** HPos, NumericTable** Val, NumericTable** WPosTest, NumericTable** HPosTest, NumericTable** ValTest, 
            NumericTable *r[], Parameter* &par, int* &col_ids, interm** &hMat_native_mem);

    // another multi-threading version of training process implemented by OpenMP 
    void compute_train_omp(int* &workWPos, int* &workHPos, interm* &workV, const int dim_set,
                           interm* &mtWDataPtr, int* &col_ids, interm** &hMat_native_mem, Parameter* &parameter);

    // another multi-threading version of testing process implemented by OpenMP 
    void compute_test_omp(int* workWPos, int* workHPos, interm* workV, const int dim_set, interm* mtWDataPtr, interm* mtRMSEPtr, Parameter *parameter, int* col_ids, interm** hMat_native_mem);

};

/**
 * @brief A TBB kernel for computing MF-SGD
 *
 * @tparam interm
 * @tparam cpu
 */
template<typename interm, CpuType cpu>
struct MFSGDTBB
{
    // default constructor  
    MFSGDTBB(
            interm* mtWDataTable,
            interm* mtHDataTable,
            int* workWPos,
            int* workHPos,
            interm *workV,
            const long Dim,
            const interm learningRate,
            const interm lambda,
            currentMutex_t* mutex_w,
            currentMutex_t* mutex_h,
            const int Avx_explicit,
            const int step,
            const int dim_train
            );

	/**
	 * @brief operator used by parallel_for template
	 *
	 * @param[in] range range of parallel block to execute by a thread
	 */
    void operator()( const blocked_range<int>& range ) const; 
	
	/**
	 * @brief set up the id of iteration
	 * used in distributed mode
	 *
	 * @param itr
	 */
    void setItr(int itr) { _itr = itr;}
    
    void setTimeStart(tbb::tick_count timeStart) {_timeStart = timeStart;}

    void setTimeOut(double timeOut) {_timeOut = timeOut;}

    void setOrder(int* order) {_order = order;}

    // model W 
    interm* _mtWDataTable;  

    // model H 
    interm* _mtHDataTable;  

    // row id of point in W 
    int* _workWPos;         

    // col id of point in H 
    int* _workHPos;		    

    // value of point 
    interm* _workV;         

    // dimension of vector in model W and H 
    long _Dim;              
    interm _learningRate;
    interm _lambda;

    // 1 if use explicit avx intrincis 0 if use compiler vectorization 
    int _Avx_explicit;   

    // stride of tasks if only part of tasks are executed 
    int _step;              
    // total number of tasks 
    int _dim_train;         
    // iteration id  
    int _itr;               
    tbb::tick_count _timeStart = tbb::tick_count::now();

    double _timeOut = 0;

    currentMutex_t* _mutex_w;
    currentMutex_t* _mutex_h;

    int* _order = NULL;

};

template<typename interm, CpuType cpu>
struct MFSGDTBBREORDER
{
    // default constructor  
    MFSGDTBBREORDER(
            interm* mtWDataTable,
            interm* mtHDataTable,
            int* queueWPos,
            int* queueLength,
            int** queueHPos,
            interm** queueVVal,
            const long Dim,
            const interm learningRate,
            const interm lambda,
            currentMutex_t* mutex_w,
            currentMutex_t* mutex_h,
            const int Avx_explicit
            );

	/**
	 * @brief operator used by parallel_for template
	 *
	 * @param[in] range range of parallel block to execute by a thread
	 */
    void operator()( const blocked_range<int>& range ) const; 

    /* model W */
    interm* _mtWDataTable;  
    /* model H */
    interm* _mtHDataTable;  

    // row id of W for each queue 
    int* _queueWPos;        

    // num of cols for each row 
    int* _queueLength;		

    // col ids of H for each queue 
    int** _queueHPos;       

    // value of points for each queue 
    interm** _queueVVal;    

    // dimension of vector in model W and H 
    long _Dim;              
    interm _learningRate;
    interm _lambda;

    // 1 if use explicit avx intrincis 0 if use compiler vectorization 
    int _Avx_explicit;   

    currentMutex_t* _mutex_w;
    currentMutex_t* _mutex_h;

};

/* MF_SGD_Test kernel implemented by TBB */
template<typename interm, CpuType cpu>
struct MFSGDTBB_TEST
{
    
    MFSGDTBB_TEST(

            interm* mtWDataTable,
            interm* mtHDataTable,
            int* testWPos,
            int* testHPos,
            interm *testV,
            const long Dim,
            interm* testRMSE,
            currentMutex_t* mutex_w,
            currentMutex_t* mutex_h,
            const int Avx_explicit
    );

    void operator()( const blocked_range<int>& range ) const; 

    // model W 
	interm* _mtWDataTable;   
    
    // model H 
    interm* _mtHDataTable;   

    // row id of point in W 
    int* _testWPos;          

    // col id of point in H 
    int* _testHPos;          

    // value of point  
    interm* _testV;          

    // dimension of model data 
    int _Dim;                

    // RMSE value calculated for each training point 
    interm* _testRMSE;       

    // 1 to use explicit avx intrinsics or 0 not 
    int _Avx_explicit;    

    currentMutex_t* _mutex_w;
    currentMutex_t* _mutex_h;


};

/**
 * @brief A function to copy data between JavaNumericTable 
 * and native memory space in parallel
 *
 * @tparam interm
 */
template <typename interm>
struct SOADataCopy
{
    SOADataCopy(
            NumericTable* SOA_Table,
            int start_pos,
            int len,
            int nDim,
            BlockDescriptor<interm>** Descriptor,
            interm** nativeMem
            ): _SOA_Table(SOA_Table), _start_pos(start_pos), _len(len), _nDim(nDim), _Descriptor(Descriptor), _nativeMem(nativeMem) {}

    NumericTable* _SOA_Table;
    int _start_pos;
    int _len;
    int _nDim;
    BlockDescriptor<interm>** _Descriptor;
    interm** _nativeMem;

};

/**
 * @brief Function to retrieve data from JavaNumericTable
 *
 * @tparam interm
 * @param arg
 *
 * @return 
 */
template <typename interm>
void* SOACopyBulkData(void* arg)
{/*{{{*/
    internal::SOADataCopy<interm>* copyElem = static_cast<internal::SOADataCopy<interm>* >(arg);
    (copyElem->_SOA_Table)->getBlockOfColumnValuesBM(copyElem->_start_pos, copyElem->_len, 0, copyElem->_nDim, writeOnly, copyElem->_Descriptor);

    //assign ptr of blockDescriptors to nativeMem
    for(int k=0;k<copyElem->_len;k++)
    {
        int feature_idx = copyElem->_start_pos + k;
        (copyElem->_nativeMem)[feature_idx] = (copyElem->_Descriptor)[feature_idx]->getBlockPtr();

    }

    return NULL;
}/*}}}*/

/**
 * @brief Funtion to release data to JavaNumericTable 
 *
 * @tparam interm
 * @param arg
 *
 * @return 
 */
template <typename interm>
void* SOAReleaseBulkData(void* arg)
{/*{{{*/
    internal::SOADataCopy<interm>* copyElem = static_cast<internal::SOADataCopy<interm>* >(arg);
    (copyElem->_SOA_Table)->releaseBlockOfColumnValuesBM(copyElem->_start_pos, copyElem->_len, copyElem->_Descriptor);
    return NULL;
}/*}}}*/

/**
 * @brief generate W matrix model data 
 *
 * @tparam interm
 * @tparam cpu
 * @param r[]
 * @param par
 * @param result
 * @param dim_r
 * @param thread_num
 */
template<typename interm, CpuType cpu>
void wMat_generate_distri(NumericTable *r[], mf_sgd::Parameter* &par, mf_sgd::DistributedPartialResult* &result, size_t dim_r, int thread_num)
{/*{{{*/

    //construct the wMat_hashtable
    //use size_t to avoid overflow of wMat_bytes
    size_t wMat_size = r[0]->getNumberOfRows();
    size_t wMat_bytes = wMat_size*dim_r*sizeof(interm); 

    // std::printf("Start constructing wMat_map, size:%zd, %zd, bytes: %zd\n", wMat_size, dim_r, wMat_bytes);
    // std::fflush(stdout);
    interm* wMat_body = (interm*)daal::services::daal_malloc(wMat_bytes);
    assert(wMat_body != NULL);

    par->_wMat_map = new ConcurrentModelMap(wMat_size);
    assert(par->_wMat_map != NULL);

    interm scale = 1.0/sqrt(static_cast<interm>(dim_r));

    daal::internal::FeatureMicroTable<int, readWrite, cpu> wMat_index(r[0]);
    int* wMat_index_ptr = 0;
    wMat_index.getBlockOfColumnValues(0, 0, wMat_size, &wMat_index_ptr);

#ifdef _OPENMP

    if (thread_num == 0)
        thread_num = omp_get_max_threads();

    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(size_t k=0;k<wMat_size;k++)
    {
        ConcurrentModelMap::accessor pos; 
        if(par->_wMat_map->insert(pos, wMat_index_ptr[k]))
        {
            pos->second = k;
        }

        pos.release();
        daal::internal::UniformRng<interm, daal::sse2> rng1(time(0));
        //k*dim_r is size_t to avoid integer overflow
        rng1.uniform(dim_r, 0.0, scale, &wMat_body[k*dim_r]);
    }

#else

    // a serial version 
    daal::internal::UniformRng<interm, daal::sse2> rng1(time(0));

    for(size_t k=0;k<wMat_size;k++)
    {
        ConcurrentModelMap::accessor pos; 
        if(par->_wMat_map->insert(pos, wMat_index_ptr[k]))
        {
            pos->second = k;
        }
        pos.release();
        //randomize the kth row in the memory space
        rng1.uniform(dim_r, 0.0, scale, &wMat_body[k*dim_r]);
    }

#endif

    par->_wMatFinished = 1;
    result->set(presWData, data_management::NumericTablePtr(new HomogenNumericTable<interm>(wMat_body, dim_r, wMat_size)));

}/*}}}*/

/**
 * @brief generate training dataset hashmap 
 *
 * @tparam interm
 * @tparam cpu
 * @param r[]
 * @param a0
 * @param a1
 * @param par
 * @param dim_r
 * @param thread_num
 */
template<typename interm, CpuType cpu>
void train_generate_distri(NumericTable *r[], NumericTable* &a0, NumericTable* &a1, mf_sgd::Parameter* &par, size_t dim_r, int thread_num)
{/*{{{*/

    par->_train_map = new ConcurrentDataMap();
    assert(par->_train_map != NULL);

    size_t train_size = a0->getNumberOfRows();
    daal::internal::FeatureMicroTable<int, readWrite, cpu> train_wPos(a0);
    daal::internal::FeatureMicroTable<int, readWrite, cpu> train_hPos(a1);

    int* train_wPos_ptr = 0;
    train_wPos.getBlockOfColumnValues(0, 0, train_size, &train_wPos_ptr);
    int* train_hPos_ptr = 0;
    train_hPos.getBlockOfColumnValues(0, 0, train_size, &train_hPos_ptr);

#ifdef _OPENMP

    if (thread_num == 0)
        thread_num = omp_get_max_threads();

    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(size_t k=0;k<train_size;k++)
    {
        ConcurrentModelMap::accessor pos_w;
        ConcurrentDataMap::accessor pos_train;

        //  replace row id by row position 
        int row_id = train_wPos_ptr[k];
        if (par->_wMat_map->find(pos_w, row_id))
        {
            train_wPos_ptr[k] = pos_w->second;
        }
        else
            train_wPos_ptr[k] = -1;

        pos_w.release();

        // construct the training data queue indexed by col id 
        int col_id = train_hPos_ptr[k];
        par->_train_map->insert(pos_train, col_id);
        pos_train->second.push_back(k);

        pos_train.release();

    }

#else

    for(size_t k=0;k<train_size;k++)
    {
        ConcurrentModelMap::accessor pos_w;
        ConcurrentModelMap::accessor pos_train;

        // replace row id by row position 
        int row_id = train_wPos_ptr[k];
        if (par->_wMat_map->find(pos_w, row_id))
        {
            train_wPos_ptr[k] = pos_w->second;
        }
        else
            train_wPos_ptr[k] = -1;

        pos_w.release();

        // construct the training data queue indexed by col id 
        int col_id = train_hPos_ptr[k];
        par->_train_map->insert(pos_train, col_id);
        pos_train->second.push_back(k);
        pos_train.release();

    }

#endif

    par->_trainMapFinished = 1;

    //transfer data from train_Map to train_list
    size_t trainMapSize = par->_train_map->size();

    //put in the number of local columns
    par->_train_list_len = trainMapSize;

    //put in the col_ids
    par->_train_list_ids = new int[trainMapSize];
    assert(par->_train_list_ids != NULL);

    //put in the subqueue lenght
    par->_train_sub_len = new int[trainMapSize];
    assert(par->_train_sub_len != NULL);

    par->_train_list = new int *[trainMapSize];
    assert(par->_train_list);

    ConcurrentDataMap::iterator itr = par->_train_map->begin();
    int subqueueSize = 0;
    size_t traverse_itr = 0;
    while (itr != par->_train_map->end())
    {
        (par->_train_list_ids)[traverse_itr] = itr->first;
        subqueueSize = itr->second.size();
        (par->_train_sub_len)[traverse_itr] = subqueueSize;

        (par->_train_list)[traverse_itr] = new int[subqueueSize];

        std::memcpy((par->_train_list)[traverse_itr], &(itr->second)[0], subqueueSize*sizeof(int) );

        itr++;
        traverse_itr++;
    }

    //dump hashmap after transferring data to par->_train_list
    delete par->_train_map;
    par->_train_map = NULL;

    // std::printf("Finish constructing train_map\n");
    // std::fflush(stdout);

}/*}}}*/

/**
 * @brief generate test dataset hashmap
 *
 * @tparam interm
 * @tparam cpu
 * @param r[]
 * @param a3
 * @param a4
 * @param par
 * @param dim_r
 * @param thread_num
 */
template<typename interm, CpuType cpu>
void test_generate_distri(NumericTable *r[], NumericTable* &a3, NumericTable* &a4, mf_sgd::Parameter* &par, size_t dim_r, int thread_num)
{/*{{{*/

    // std::printf("Start constructing test_map\n");
    // std::fflush(stdout);

    par->_test_map = new ConcurrentDataMap();
    assert(par->_test_map != NULL);

    size_t test_size = a3->getNumberOfRows();
    daal::internal::FeatureMicroTable<int, readWrite, cpu> test_wPos(a3);
    daal::internal::FeatureMicroTable<int, readWrite, cpu> test_hPos(a4);

    int* test_wPos_ptr = 0;
    test_wPos.getBlockOfColumnValues(0, 0, test_size, &test_wPos_ptr);
    int* test_hPos_ptr = 0;
    test_hPos.getBlockOfColumnValues(0, 0, test_size, &test_hPos_ptr);

#ifdef _OPENMP

    if (thread_num == 0)
        thread_num = omp_get_max_threads();

    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(size_t k=0;k<test_size;k++)
    {
        ConcurrentModelMap::accessor pos_w;
        ConcurrentDataMap::accessor pos_test;

        // replace row id by row position 
        int row_id = test_wPos_ptr[k];
        if (par->_wMat_map->find(pos_w, row_id))
        {
            test_wPos_ptr[k] = pos_w->second;
        }
        else
            test_wPos_ptr[k] = -1;

        pos_w.release();

        // construct the test data queue indexed by col id 
        int col_id = test_hPos_ptr[k];
        par->_test_map->insert(pos_test, col_id);
        pos_test->second.push_back(k);
        pos_test.release();

    }

#else

    //a sequential version
    for(size_t k=0;k<test_size;k++)
    {
        ConcurrentModelMap::accessor pos_w;
        ConcurrentDataMap::accessor pos_test;

        // replace row id by row position 
        int row_id = test_wPos_ptr[k];
        if (par->_wMat_map->find(pos_w, row_id))
        {
            test_wPos_ptr[k] = pos_w->second;
        }
        else
            test_wPos_ptr[k] = -1;

        pos_w.release();

        // construct the test data queue indexed by col id 
        int col_id = test_hPos_ptr[k];
        par->_test_map->insert(pos_test, col_id);
        pos_test->second.push_back(k);

        pos_test.release();

    }


#endif

    par->_testMapFinished = 1;
    //transfer data from test_Map to test_list
    size_t testMapSize = par->_test_map->size();

    //put in the number of local columns
    par->_test_list_len = testMapSize;

    //put in the col_ids
    par->_test_list_ids = new int[testMapSize];
    assert(par->_test_list_ids != NULL);

    //put in the subqueue lenght
    par->_test_sub_len = new int[testMapSize];
    assert(par->_test_sub_len != NULL);

    par->_test_list = new int *[testMapSize];
    assert(par->_test_list != NULL);

    ConcurrentDataMap::iterator itr = par->_test_map->begin();
    int subqueueSize = 0;
    size_t traverse_itr = 0;
    while (itr != par->_test_map->end())
    {
        (par->_test_list_ids)[traverse_itr] = itr->first;
        subqueueSize = itr->second.size();
        (par->_test_sub_len)[traverse_itr] = subqueueSize;

        (par->_test_list)[traverse_itr] = new int[subqueueSize];

        std::memcpy((par->_test_list)[traverse_itr], &(itr->second)[0], subqueueSize*sizeof(int) );

        itr++;
        traverse_itr++;
    }

    //dump hashmap after transferring to par->_test_list
    delete par->_test_map;
    par->_test_map = NULL;

    // std::printf("Finish constructing test_map\n");
    // std::fflush(stdout);

}/*}}}*/

/**
 * @brief generate H matrix model data from JavaNumericTable 
 *
 * @tparam interm
 * @tparam cpu
 * @param r[]
 * @param par
 * @param dim_r
 * @param thread_num
 * @param col_ids
 * @param hMat_native_mem
 * @param hMat_blk_array
 * @param copylist
 */
template<typename interm, CpuType cpu>
void hMat_generate(NumericTable *r[], mf_sgd::Parameter* &par, size_t dim_r, int thread_num, int* &col_ids,
    interm** &hMat_native_mem, BlockDescriptor<interm>** &hMat_blk_array, internal::SOADataCopy<interm>** &copylist)
{/*{{{*/

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double hMat_time = 0;

    // std::printf("Start constructing h_map\n");
    // std::fflush(stdout);

    int hMat_rowNum = par->_Dim_h;
    assert(hMat_rowNum <= (r[1]->getNumberOfColumns()));

    // should be dim_r + 1, there is a sentinel to record the col id 
    int hMat_colNum = r[1]->getNumberOfRows(); 
    assert(hMat_colNum == dim_r + 1);

    col_ids = (int*)calloc(hMat_rowNum, sizeof(int));
    assert(col_ids != NULL);

    hMat_native_mem = new interm *[hMat_rowNum];
    assert(hMat_native_mem != NULL);

    hMat_blk_array = new BlockDescriptor<interm> *[hMat_rowNum];
    assert(hMat_blk_array != NULL);

    copylist = new internal::SOADataCopy<interm> *[thread_num];
    assert(copylist != NULL);

    for(int k=0;k<hMat_rowNum;k++)
        hMat_blk_array[k] = new BlockDescriptor<interm>();

    int res = hMat_rowNum%thread_num;
    int cpy_len = (int)((hMat_rowNum - res)/thread_num);
    int last_cpy_len = cpy_len + res;

    for(int k=0;k<thread_num-1;k++)
        copylist[k] = new internal::SOADataCopy<interm>(r[1], k*cpy_len, cpy_len, hMat_colNum, hMat_blk_array, hMat_native_mem);

    copylist[thread_num-1] = new internal::SOADataCopy<interm>(r[1], (thread_num-1)*cpy_len, last_cpy_len, hMat_colNum, hMat_blk_array, hMat_native_mem);

    // std::printf("Start converting h_map\n");
    // std::fflush(stdout);

    //---------------------------------- start doing a parallel data conversion by using pthread----------------------------------
    clock_gettime(CLOCK_MONOTONIC, &ts1);

    #pragma omp parallel for schedule(static) num_threads(thread_num) 
    for(int k=0;k<thread_num;k++)
    {
        internal::SOACopyBulkData<interm>(copylist[k]);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    hMat_time = (double)(diff)/1000000L;

    // std::printf("Loading hMat time: %f\n", hMat_time);
    // std::fflush(stdout);
    par->_jniDataConvertTime += (size_t)hMat_time;

    // std::printf("Finish converting h_map\n");
    // std::fflush(stdout);

    //---------------------------------- finish doing a parallel data conversion by using pthread----------------------------------
    //clean up and re-generate a hMat hashmap
    if (par->_hMat_map != NULL)
        par->_hMat_map->~ConcurrentModelMap();

    par->_hMat_map = new ConcurrentModelMap(hMat_rowNum);
    assert(par->_hMat_map != NULL);

    if (thread_num == 0)
        thread_num = omp_get_max_threads();

    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(int k=0;k<hMat_rowNum;k++)
    {
        ConcurrentModelMap::accessor pos; 
        int col_id = (int)((hMat_native_mem[k])[0]);
        col_ids[k] = col_id;

        if(par->_hMat_map->insert(pos, col_id))
        {
            pos->second = k;
        }

        pos.release();

    }

    // std::printf("Finish constructing h_map\n");
    // std::fflush(stdout);

}/*}}}*/

/**
 * @brief release H matrix model data to JavaNumericTable 
 *
 * @tparam interm
 * @tparam cpu
 * @param r[]
 * @param par
 * @param dim_r
 * @param thread_num
 * @param hMat_blk_array
 * @param copylist
 */
template<typename interm, CpuType cpu>
void hMat_release(NumericTable *r[], mf_sgd::Parameter* &par, size_t dim_r, int thread_num, BlockDescriptor<interm>** &hMat_blk_array, internal::SOADataCopy<interm>** &copylist)
{/*{{{*/

    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double hMat_time = 0;

    //a parallel verison
    if (hMat_blk_array != NULL)
    {

        clock_gettime(CLOCK_MONOTONIC, &ts1);

        #pragma omp parallel for schedule(static) num_threads(thread_num) 
        for(int k=0;k<thread_num;k++)
        {
            internal::SOAReleaseBulkData<interm>(copylist[k]);
        }

        //free up memory space of native column data of hMat
        int hMat_rows_size = par->_Dim_h;
        for(int k=0;k<hMat_rows_size;k++)
        {
            hMat_blk_array[k]->~BlockDescriptor();
            delete hMat_blk_array[k];

        }

        //free up memory space of pthread copy args
        for(int k=0;k<thread_num;k++)
            delete copylist[k];

        delete[] copylist;

        clock_gettime(CLOCK_MONOTONIC, &ts2);
        diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
        hMat_time = (double)(diff)/1000000L;

        par->_jniDataConvertTime += (size_t)hMat_time;

    }


}/*}}}*/

} // namespace daal::internal
}
}
} // namespace daal

#endif

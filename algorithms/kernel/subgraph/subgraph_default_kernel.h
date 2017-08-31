/* file: subgraph_dense_default_kernel.h */
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
//  Declaration of template function that calculate subgraphs.
//--
*/

#ifndef __SUBGRAPH_FPK_H__
#define __SUBGRAPH_FPK_H__

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

#include "subgraph_batch.h"
#include "tbb/tick_count.h"

using namespace tbb;
using namespace daal::data_management;

typedef queuing_mutex currentMutex_t;

namespace daal
{
namespace algorithms
{
namespace subgraph
{

// typedef tbb::concurrent_hash_map<int, int> ConcurrentModelMap;
// typedef tbb::concurrent_hash_map<int, std::vector<int> > ConcurrentDataMap;

namespace internal
{

/**
 * @brief computation kernel for subgraph batch mode 
 *
 * @tparam interm
 * @tparam method
 * @tparam cpu
 */
template<typename interm, daal::algorithms::subgraph::Method method, CpuType cpu>
class subgraphBatchKernel : public Kernel
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
    daal::services::interface1::Status compute(const NumericTable** TrainSet, const NumericTable** TestSet,
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

private:

    services::SharedPtr<std::vector<int> > _trainWQueue;
    services::SharedPtr<std::vector<int> > _trainLQueue;
    services::SharedPtr<std::vector<int*> > _trainHQueue;
    services::SharedPtr<std::vector<interm*> > _trainVQueue;

};

/**
 * @brief computation kernel for subgraph distributed mode
 *
 * @tparam interm
 * @tparam method
 * @tparam cpu
 */
template<typename interm, daal::algorithms::subgraph::Method method, CpuType cpu>
class subgraphDistriKernel : public Kernel
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
    daal::services::interface1::Status compute(NumericTable** WPos, NumericTable** HPos, NumericTable** Val, NumericTable** WPosTest, NumericTable** HPosTest, NumericTable** ValTest, 
            NumericTable *r[], Parameter* &par, int* &col_ids, interm** &hMat_native_mem);

};

/**
 * @brief A TBB kernel for computing MF-SGD
 *
 * @tparam interm
 * @tparam cpu
 */
// template<typename interm, CpuType cpu>
// struct MFSGDTBB
// {
//     // default constructor  
//     MFSGDTBB(
//             interm* mtWDataTable,
//             interm* mtHDataTable,
//             int* workWPos,
//             int* workHPos,
//             interm *workV,
//             const long Dim,
//             const interm learningRate,
//             const interm lambda,
//             currentMutex_t* mutex_w,
//             currentMutex_t* mutex_h,
//             const int Avx_explicit,
//             const int step,
//             const int dim_train
//             );
//
// 	/**
// 	 * @brief operator used by parallel_for template
// 	 *
// 	 * @param[in] range range of parallel block to execute by a thread
// 	 */
//     void operator()( const blocked_range<int>& range ) const; 
// 	
// 	/**
// 	 * @brief set up the id of iteration
// 	 * used in distributed mode
// 	 *
// 	 * @param itr
// 	 */
//     void setItr(int itr) { _itr = itr;}
//     
//     void setTimeStart(tbb::tick_count timeStart) {_timeStart = timeStart;}
//
//     void setTimeOut(double timeOut) {_timeOut = timeOut;}
//
//     void setOrder(int* order) {_order = order;}
//
//     // model W 
//     interm* _mtWDataTable;  
//
//     // model H 
//     interm* _mtHDataTable;  
//
//     // row id of point in W 
//     int* _workWPos;         
//
//     // col id of point in H 
//     int* _workHPos;		    
//
//     // value of point 
//     interm* _workV;         
//
//     // dimension of vector in model W and H 
//     long _Dim;              
//     interm _learningRate;
//     interm _lambda;
//
//     // 1 if use explicit avx intrincis 0 if use compiler vectorization 
//     int _Avx_explicit;   
//
//     // stride of tasks if only part of tasks are executed 
//     int _step;              
//     // total number of tasks 
//     int _dim_train;         
//     // iteration id  
//     int _itr;               
//     tbb::tick_count _timeStart = tbb::tick_count::now();
//
//     double _timeOut = 0;
//
//     currentMutex_t* _mutex_w;
//     currentMutex_t* _mutex_h;
//
//     int* _order = NULL;
//
// };


/**
 * @brief A function to copy data between JavaNumericTable 
 * and native memory space in parallel
 *
 * @tparam interm
 */
// template <typename interm>
// struct SOADataCopy
// {
//     SOADataCopy(
//             NumericTable* SOA_Table,
//             int start_pos,
//             int len,
//             int nDim,
//             BlockDescriptor<interm>** Descriptor,
//             interm** nativeMem
//             ): _SOA_Table(SOA_Table), _start_pos(start_pos), _len(len), _nDim(nDim), _Descriptor(Descriptor), _nativeMem(nativeMem) {}
//
//     NumericTable* _SOA_Table;
//     int _start_pos;
//     int _len;
//     int _nDim;
//     BlockDescriptor<interm>** _Descriptor;
//     interm** _nativeMem;
//
// };

/**
 * @brief Function to retrieve data from JavaNumericTable
 *
 * @tparam interm
 * @param arg
 *
 * @return 
 */
// template <typename interm>
// void* SOACopyBulkData(void* arg)
// {/*{{{*/
//     internal::SOADataCopy<interm>* copyElem = static_cast<internal::SOADataCopy<interm>* >(arg);
//     (copyElem->_SOA_Table)->getBlockOfColumnValuesBM(copyElem->_start_pos, copyElem->_len, 0, copyElem->_nDim, writeOnly, copyElem->_Descriptor);
//
//     //assign ptr of blockDescriptors to nativeMem
//     for(int k=0;k<copyElem->_len;k++)
//     {
//         int feature_idx = copyElem->_start_pos + k;
//         (copyElem->_nativeMem)[feature_idx] = (copyElem->_Descriptor)[feature_idx]->getBlockPtr();
//
//     }
//
//     return NULL;
// }/*}}}*/


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
// template<typename interm, CpuType cpu>
// void hMat_generate(NumericTable *r[], subgraph::Parameter* &par, size_t dim_r, int thread_num, int* &col_ids,
//     interm** &hMat_native_mem, BlockDescriptor<interm>** &hMat_blk_array, internal::SOADataCopy<interm>** &copylist)
// {/*{{{*/
//
//     struct timespec ts1;
// 	struct timespec ts2;
//     int64_t diff = 0;
//     double hMat_time = 0;
//
//     // std::printf("Start constructing h_map\n");
//     // std::fflush(stdout);
//
//     int hMat_rowNum = par->_Dim_h;
//     assert(hMat_rowNum <= (r[1]->getNumberOfColumns()));
//
//     // should be dim_r + 1, there is a sentinel to record the col id 
//     int hMat_colNum = r[1]->getNumberOfRows(); 
//     assert(hMat_colNum == dim_r + 1);
//
//     col_ids = (int*)calloc(hMat_rowNum, sizeof(int));
//     assert(col_ids != NULL);
//
//     hMat_native_mem = new interm *[hMat_rowNum];
//     assert(hMat_native_mem != NULL);
//
//     hMat_blk_array = new BlockDescriptor<interm> *[hMat_rowNum];
//     assert(hMat_blk_array != NULL);
//
//     copylist = new internal::SOADataCopy<interm> *[thread_num];
//     assert(copylist != NULL);
//
//     for(int k=0;k<hMat_rowNum;k++)
//         hMat_blk_array[k] = new BlockDescriptor<interm>();
//
//     int res = hMat_rowNum%thread_num;
//     int cpy_len = (int)((hMat_rowNum - res)/thread_num);
//     int last_cpy_len = cpy_len + res;
//
//     for(int k=0;k<thread_num-1;k++)
//         copylist[k] = new internal::SOADataCopy<interm>(r[1], k*cpy_len, cpy_len, hMat_colNum, hMat_blk_array, hMat_native_mem);
//
//     copylist[thread_num-1] = new internal::SOADataCopy<interm>(r[1], (thread_num-1)*cpy_len, last_cpy_len, hMat_colNum, hMat_blk_array, hMat_native_mem);
//
//     // std::printf("Start converting h_map\n");
//     // std::fflush(stdout);
//
//     //---------------------------------- start doing a parallel data conversion by using openmp----------------------------------
//     clock_gettime(CLOCK_MONOTONIC, &ts1);
//
//     #pragma omp parallel for schedule(static) num_threads(thread_num) 
//     for(int k=0;k<thread_num;k++)
//     {
//         internal::SOACopyBulkData<interm>(copylist[k]);
//     }
//
//     clock_gettime(CLOCK_MONOTONIC, &ts2);
//     diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
//     hMat_time = (double)(diff)/1000000L;
//
//     // std::printf("Loading hMat time: %f\n", hMat_time);
//     // std::fflush(stdout);
//     par->_jniDataConvertTime += (size_t)hMat_time;
//
//     // std::printf("Finish converting h_map\n");
//     // std::fflush(stdout);
//     // debug: check the correctness of parallel copy
//     // int feature_select = 1; 
//     // BlockDescriptor<interm> check_col;
//     // r[1]->getBlockOfColumnValues((size_t)feature_select, 0, hMat_colNum, readOnly, check_col);
//     // interm* check_val = check_col.getBlockPtr();
//     // interm error_val = 0;
//     // for(int p=0;p<hMat_colNum;p++)
//     // {
//     //    error_val += (check_val[p] - hMat_native_mem[feature_select][p]); 
//     //    if (p<10)
//     //    {
//     //        std::printf("col val after: %f\n",hMat_native_mem[feature_select][p] );
//     //        std::fflush(stdout);
//     //    }
//     //        
//     // }
//     // std::printf("copy error for row: %f\n", error_val);
//     // std::fflush(stdout);
//
//
//     //---------------------------------- finish doing a parallel data conversion by using openmp----------------------------------
//     //clean up and re-generate a hMat hashmap
//     if (par->_hMat_map != NULL)
//         par->_hMat_map->~ConcurrentModelMap();
//
//     par->_hMat_map = new ConcurrentModelMap(hMat_rowNum);
//     assert(par->_hMat_map != NULL);
//
//     if (thread_num == 0)
//         thread_num = omp_get_max_threads();
//
//     #pragma omp parallel for schedule(guided) num_threads(thread_num) 
//     for(int k=0;k<hMat_rowNum;k++)
//     {
//         ConcurrentModelMap::accessor pos; 
//         int col_id = (int)((hMat_native_mem[k])[0]);
//         col_ids[k] = col_id;
//
//         if(par->_hMat_map->insert(pos, col_id))
//         {
//             pos->second = k;
//         }
//
//         pos.release();
//
//     }
//
//     // std::printf("Finish constructing h_map\n");
//     // std::fflush(stdout);
//
// }/*}}}*/

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
// template<typename interm, CpuType cpu>
// void hMat_release(NumericTable *r[], subgraph::Parameter* &par, size_t dim_r, int thread_num, BlockDescriptor<interm>** &hMat_blk_array, internal::SOADataCopy<interm>** &copylist)
// {/*{{{*/
//
//     struct timespec ts1;
// 	struct timespec ts2;
//     int64_t diff = 0;
//     double hMat_time = 0;
//
//     //a parallel verison
//     if (hMat_blk_array != NULL)
//     {
//
//         clock_gettime(CLOCK_MONOTONIC, &ts1);
//
//         //debug
//         // std::printf("entering parallel release\n");
//         // std::fflush(stdout);
//
//         #pragma omp parallel for schedule(static) num_threads(thread_num) 
//         for(int k=0;k<thread_num;k++)
//         {
//             internal::SOAReleaseBulkData<interm>(copylist[k]);
//         }
//
//         //debug
//         // std::printf("Finishing parallel release\n");
//         // std::fflush(stdout);
//
//         clock_gettime(CLOCK_MONOTONIC, &ts2);
//         diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
//         hMat_time = (double)(diff)/1000000L;
//
//         par->_jniDataConvertTime += (size_t)hMat_time;
//
//         //free up memory space of native column data of hMat
//         int hMat_rows_size = par->_Dim_h;
//         for(int k=0;k<hMat_rows_size;k++)
//         {
//             // hMat_blk_array[k]->~BlockDescriptor();
//             delete hMat_blk_array[k];
//
//         }
//
//         //debug
//         // std::printf("Finishing parallel free blockptr\n");
//         // std::fflush(stdout);
//
//         //free up memory space of pthread copy args
//         for(int k=0;k<thread_num;k++)
//             delete copylist[k];
//
//         delete[] copylist;
//
//         
//     }
//
//
// }/*}}}*/

} // namespace daal::internal
}
}
} // namespace daal

#endif

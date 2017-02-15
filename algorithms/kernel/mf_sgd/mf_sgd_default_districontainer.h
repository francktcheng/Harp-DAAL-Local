/* file: mf_sgd_dense_districontainer.h */
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
//  Implementation of mf_sgd calculation algorithm container.
//--
*/
#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <cstdio>
#include <math.h>       
#include <random>
#include <vector>
#include "numeric_table.h"
#include "service_rng.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include <omp.h>
#include "mf_sgd_types.h"
#include "mf_sgd_distri.h"
#include "mf_sgd_default_kernel.h"

using namespace tbb;

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
    

typedef tbb::concurrent_hash_map<int, int> ConcurrentModelMap;
typedef tbb::concurrent_hash_map<int, std::vector<int> > ConcurrentDataMap;

/**
 *  @brief Initialize list of mf_sgd with implementations for supported architectures
 */
template<ComputeStep step, typename interm, Method method, CpuType cpu>
DistriContainer<step, interm, method, cpu>::DistriContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MF_SGDDistriKernel, interm, method);
}

template<ComputeStep step, typename interm, Method method, CpuType cpu>
DistriContainer<step, interm, method, cpu>::~DistriContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<ComputeStep step, typename interm, Method method, CpuType cpu>
void DistriContainer<step, interm, method, cpu>::compute()
{
    // prepare the computation
    Input *input = static_cast<Input *>(_in);
    DistributedPartialResult *result = static_cast<DistributedPartialResult *>(_pres);
    Parameter *par = static_cast<Parameter*>(_par);
    int* col_ids = NULL;

    /* retrieve the training and test datasets */
    NumericTable *a0 = static_cast<NumericTable *>(input->get(wPos).get());
    const NumericTable *a1 = static_cast<const NumericTable *>(input->get(hPos).get());
    const NumericTable *a2 = static_cast<const NumericTable *>(input->get(val).get());
    
    NumericTable *a3 = NULL;
    NumericTable *a4 = NULL;
    NumericTable *a5 = NULL;

    if (par->_sgd2 == 1)
    {
        a3 = static_cast<NumericTable *>(input->get(wPosTest).get());
        a4 = static_cast<NumericTable *>(input->get(hPosTest).get());
        a5 = static_cast<NumericTable *>(input->get(valTest).get());
    }

    NumericTable **WPos = &a0;
    const NumericTable **HPos = &a1;
    const NumericTable **Val = &a2;

    NumericTable **WPosTest = &a3;
    NumericTable **HPosTest = &a4;
    NumericTable **ValTest = &a5;

    int dim_r = par->_Dim_r;

    NumericTable *r[4];
    r[0] = static_cast<NumericTable *>(result->get(presWMat).get());
    r[1] = static_cast<NumericTable *>(result->get(presHMat).get());

    
    //initialize wMat_hashtable for the first iteration
    //store the wMat_hashtable within par of mf_sgd
    //regenerate the wMat numericTablj 
    if (par->_wMat_map == NULL && par->_sgd2 == 1)
    {
        
        //construct the wMat_hashtable
        int wMat_size = r[0]->getNumberOfRows();

        interm* wMat_body = (interm*)calloc(dim_r*wMat_size, sizeof(interm));

        par->_wMat_map = new ConcurrentModelMap(wMat_size);

        interm scale = 1.0/sqrt(static_cast<interm>(dim_r));

        daal::internal::FeatureMicroTable<int, readWrite, cpu> wMat_index(r[0]);
        int* wMat_index_ptr = 0;
        wMat_index.getBlockOfColumnValues(0, 0, wMat_size, &wMat_index_ptr);

#ifdef _OPENMP

        int thread_num = omp_get_max_threads();
        #pragma omp parallel for schedule(guided) num_threads(thread_num) 
        for(int k=0;k<wMat_size;k++)
        {
            ConcurrentModelMap::accessor pos; 
            if(par->_wMat_map->insert(pos, wMat_index_ptr[k]))
            {
                pos->second = k;
            }

            pos.release();

            daal::internal::UniformRng<interm, daal::sse2> rng1(time(0));
            rng1.uniform(dim_r, 0.0, scale, &wMat_body[k*dim_r]);
        }

#else

        /* a serial version */
        daal::internal::UniformRng<interm, daal::sse2> rng1(time(0));

        for(int k=0;k<wMat_size;k++)
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

        result->set(presWData, data_management::NumericTablePtr(new HomogenNumericTable<interm>(wMat_body, dim_r, wMat_size)));

    }

    /* construct the hashmap to hold training point position indexed by col id */
    if (par->_train_map == NULL && par->_sgd2 == 1)
    {
        par->_train_map = new ConcurrentDataMap();

        int train_size = a0->getNumberOfRows();
        daal::internal::FeatureMicroTable<int, readWrite, cpu> train_wPos(a0);
        daal::internal::FeatureMicroTable<int, readWrite, cpu> train_hPos(a1);

        int* train_wPos_ptr = 0;
        train_wPos.getBlockOfColumnValues(0, 0, train_size, &train_wPos_ptr);
        int* train_hPos_ptr = 0;
        train_hPos.getBlockOfColumnValues(0, 0, train_size, &train_hPos_ptr);

#ifdef _OPENMP

        int thread_num = omp_get_max_threads();
        #pragma omp parallel for schedule(guided) num_threads(thread_num) 
        for(int k=0;k<train_size;k++)
        {
            ConcurrentModelMap::accessor pos_w;
            ConcurrentDataMap::accessor pos_train;

            /* replace row id by row position */
            int row_id = train_wPos_ptr[k];
            if (par->_wMat_map->find(pos_w, row_id))
            {
                train_wPos_ptr[k] = pos_w->second;
            }
            else
                train_wPos_ptr[k] = -1;

            pos_w.release();

            /* construct the training data queue indexed by col id */
            int col_id = train_hPos_ptr[k];
            par->_train_map->insert(pos_train, col_id);
            pos_train->second.push_back(k);

            pos_train.release();

        }

#else

        for(int k=0;k<train_size;k++)
        {
            ConcurrentModelMap::accessor pos_w;
            ConcurrentModelMap::accessor pos_train;

            /* replace row id by row position */
            int row_id = train_wPos_ptr[k];
            if (par->_wMat_map->find(pos_w, row_id))
            {
                train_wPos_ptr[k] = pos_w->second;
            }
            else
                train_wPos_ptr[k] = -1;

            pos_w.release();

            /* construct the training data queue indexed by col id */
            int col_id = train_hPos_ptr[k];
            par->_train_map->insert(pos_train, col_id);
            pos_train->second.push_back(k);

            pos_train.release();

        }

#endif


    }

    if (par->_test_map == NULL && par->_sgd2 == 1 )
    {
        par->_test_map = new ConcurrentDataMap();

        int test_size = a3->getNumberOfRows();
        daal::internal::FeatureMicroTable<int, readWrite, cpu> test_wPos(a3);
        daal::internal::FeatureMicroTable<int, readWrite, cpu> test_hPos(a4);

        int* test_wPos_ptr = 0;
        test_wPos.getBlockOfColumnValues(0, 0, test_size, &test_wPos_ptr);
        int* test_hPos_ptr = 0;
        test_hPos.getBlockOfColumnValues(0, 0, test_size, &test_hPos_ptr);

#ifdef _OPENMP

        int thread_num = omp_get_max_threads();
        #pragma omp parallel for schedule(guided) num_threads(thread_num) 
        for(int k=0;k<test_size;k++)
        {
            ConcurrentModelMap::accessor pos_w;
            ConcurrentDataMap::accessor pos_test;

            /* replace row id by row position */
            int row_id = test_wPos_ptr[k];
            if (par->_wMat_map->find(pos_w, row_id))
            {
                test_wPos_ptr[k] = pos_w->second;
            }
            else
                test_wPos_ptr[k] = -1;

            pos_w.release();

            /* construct the test data queue indexed by col id */
            int col_id = test_hPos_ptr[k];
            par->_test_map->insert(pos_test, col_id);
            pos_test->second.push_back(k);

            pos_test.release();

        }

#else

        for(int k=0;k<test_size;k++)
        {
            ConcurrentModelMap::accessor pos_w;
            ConcurrentDataMap::accessor pos_test;

            /* replace row id by row position */
            int row_id = test_wPos_ptr[k];
            if (par->_wMat_map->find(pos_w, row_id))
            {
                test_wPos_ptr[k] = pos_w->second;
            }
            else
                test_wPos_ptr[k] = -1;

            pos_w.release();

            /* construct the test data queue indexed by col id */
            int col_id = test_hPos_ptr[k];
            par->_test_map->insert(pos_test, col_id);
            pos_test->second.push_back(k);

            pos_test.release();

        }
        

#endif


    }

    r[3] = static_cast<NumericTable *>(result->get(presWData).get());

    //debug
    // std::printf("Created W Matrix Row: %d, col: %d\n", (int)(r[2]->getNumberOfRows()), (int)(r[2]->getNumberOfColumns()));
    // std::fflush(stdout);

    if (par->_sgd2 == 1)
    {
        //initialize the hMat_hashtable for every iteration
        //store the hMat_hashtable within par of mf_sgd
        int hMat_colNum = r[1]->getNumberOfColumns(); /* should be dim_r + 1, there is a sentinel to record the col id */
        int hMat_rowNum = r[1]->getNumberOfRows();

        col_ids = (int*)calloc(hMat_rowNum, sizeof(int));

        daal::internal::BlockMicroTable<interm, readWrite, cpu> hMat_block(r[1]);
        interm* hMat_block_ptr = 0;
        hMat_block.getBlockOfRows(0, hMat_rowNum, &hMat_block_ptr);

        if (par->_hMat_map != NULL)
            par->_hMat_map->~ConcurrentModelMap();
            
        par->_hMat_map = new ConcurrentModelMap(hMat_rowNum);

#ifdef _OPENMP

        int thread_num = omp_get_max_threads();
        #pragma omp parallel for schedule(guided) num_threads(thread_num) 
        for(int k=0;k<hMat_rowNum;k++)
        {
            ConcurrentModelMap::accessor pos; 
            int col_id = (int)(hMat_block_ptr[k*hMat_colNum]);
            col_ids[k] = col_id;

            if(par->_hMat_map->insert(pos, col_id))
            {
                pos->second = k;
            }

            pos.release();

        }

#else

        /* a serial version */
        for(int k=0;k<hMat_rowNum;k++)
        {
            ConcurrentModelMap::accessor pos; 
            int col_id = (int)(hMat_block_ptr[k*hMat_colNum]);

            if(par->_hMat_map->insert(pos, col_id))
            {
                pos->second = k;
            }

            pos.release();
        }

#endif

    }

    if ((static_cast<Parameter*>(_par))->_isTrain)
        r[2] = NULL;
    else
        r[2] = static_cast<NumericTable *>(result->get(presRMSE).get());

    daal::services::Environment::env &env = *_env;

    /* invoke the MF_SGDBatchKernel */
    __DAAL_CALL_KERNEL(env, internal::MF_SGDDistriKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), compute, WPos, HPos, Val, WPosTest, HPosTest, ValTest, r, par, col_ids);

    //clean up the memory space per iteration
    if (col_ids != NULL)
        free(col_ids);

    if (par->_hMat_map != NULL)
    {
        delete par->_hMat_map;
        par->_hMat_map = NULL;

    }
   
}

template<ComputeStep step, typename interm, Method method, CpuType cpu>
void DistriContainer<step, interm, method, cpu>::finalizeCompute() {}

}
}
} // namespace daal

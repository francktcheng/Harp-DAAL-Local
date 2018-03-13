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

//mem info extraction
#include <stdio.h>
#include <string.h>

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
class subgraphDistriKernel: public Kernel
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
    daal::services::interface1::Status compute(Parameter* &par, Input* &input);

    void computeBottom(Parameter* &par, Input* &input);
    void computeBottomTBB(Parameter* &par, Input* &input);

    void computeNonBottom(Parameter* &par, Input* &input);

    void computeNonBottomNbrSplit(Parameter* &par, Input* &input);
    void computeNonBottomNbrSplitTBB(Parameter* &par, Input* &input);

    void updateRemoteCountsNbrSplit(Parameter* &par, Input* &input);
    void updateRemoteCountsNbrSplitTBB(Parameter* &par, Input* &input);

    void updateRemoteCountsPipNbrSplit(Parameter* &par, Input* &input);
    void updateRemoteCountsPipNbrSplitTBB(Parameter* &par, Input* &input);

    void process_mem_usage(double& resident_set)
    {
        resident_set = 0.0;

        FILE *fp;
        long vmrss;
        int BUFFERSIZE=80;
        char *buf= new char[85];
        if((fp = fopen("/proc/self/status","r")))
        {
            while(fgets(buf, BUFFERSIZE, fp) != NULL)
            {
                if(strstr(buf, "VmRSS") != NULL)
                {
                    if (sscanf(buf, "%*s %ld", &vmrss) == 1){
                        // printf("VmSize is %dKB\n", vmrss);
                        resident_set = (double)vmrss;
                    }
                }
            }
        }

        fclose(fp);
        delete[] buf;
    }

};



} // namespace daal::internal
}
}
} // namespace daal

#endif

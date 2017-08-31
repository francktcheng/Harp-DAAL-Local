/* file: subgraph_batch.h */
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
//  Implementation of the interface for the subgraph algorithm in the
//  batch processing mode
//--
*/

#ifndef __SUBGRAPH_BATCH_H__
#define __SUBGRAPH_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/subgraph/subgraph_types.h"

namespace daal
{
namespace algorithms
{
namespace subgraph
{

namespace interface1
{
/** @defgroup subgraph_batch Batch
* @{
*/
/**
 * <a name="DAAL-CLASS-ALGORITHMS__subgraph__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the subgraph algorithm in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the subgraph algorithm, double or float
 * \tparam method           Computation method of the subgraph algorithm, \ref daal::algorithms::subgraph::Method
 *
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch> 
{
public:
    /**
     * Constructs a container for the color-coding subgraph algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the subgraph algorithm in the batch processing mode
     */
    virtual daal::services::interface1::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__subgraph__BATCH"></a>
 * \brief Computes the results of the subgraph algorithm in the batch processing mode.
 * \n<a href="DAAL-REF-subgraph-ALGORITHM">subgraph algorithm description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the subgraph algorithm, double or float
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::subgraph::Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the subgraph algorithm
 */
template<typename algorithmFPType = double, Method method = defaultSC>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    Input input;            /*!< Input object */
    Parameter parameter;    /*!< subgraph parameters */

    Batch()
    {
        initialize();
    }

    /**
     * Constructs a subgraph algorithm by copying input objects and parameters
     * of another subgraph algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other)
    {
        initialize();
        // input.set(dataTrain, other.input.get(dataTrain));		/* !< copy input training dataset */
        // input.set(dataTest, other.input.get(dataTest));			/* !< copy input test dataset */
        parameter = other.parameter;
    }

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns the structure that contains the results of the subgraph algorithm
     * \return Structure that contains the results of the subgraph algorithm
     */
    services::SharedPtr<Result> getResult() { return _result; }

    /**
     * Register user-allocated memory to store the results of the subgraph algorithm
     * \return Structure to store the results of the subgraph algorithm
     */
    daal::services::interface1::Status setResult(const services::SharedPtr<Result>& res)
    {
        daal::services::interface1::Status status;
        DAAL_CHECK(res, daal::services::ErrorNullResult)
        _result = res;
        _res = _result.get();
        return status;
    }

    /**
     * Returns a pointer to the newly allocated subgraph algorithm
     * with a copy of input objects and parameters of this subgraph algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual daal::services::interface1::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status status;
         /*  the function to allocate the result */
        _result = services::SharedPtr<Result>(new Result());
        _result->allocate<algorithmFPType>(&input, _par, 0);
        _res = _result.get();
        return status;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in   = &input;
        _par  = &parameter;
    }

private:
    services::SharedPtr<Result> _result;
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace daal::algorithms::subgraph
}
} // namespace daal
#endif

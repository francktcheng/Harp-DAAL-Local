/* file: kmeans_parameter_types.cpp */
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
//  Implementation of kmeans classes.
//--
*/

#include "algorithms/kmeans/kmeans_types.h"
#include "daal_defines.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{

namespace interface1
{

/**
 *  Constructs parameters of the K-Means algorithm
 *  \param[in] _nClusters   Number of clusters
 *  \param[in] _maxIterations Number of iterations
 */
Parameter::Parameter(size_t _nClusters, size_t _maxIterations) :
    nClusters(_nClusters), maxIterations(_maxIterations), accuracyThreshold(0.0), gamma(1.0),
    distanceType(euclidean), assignFlag(true), nThreads(0) {}

/**
 *  Constructs parameters of the K-Means algorithm by copying another parameters of the K-Means algorithm
 *  \param[in] other    Parameters of the K-Means algorithm
 */
Parameter::Parameter(const Parameter &other) :
    nClusters(other.nClusters), maxIterations(other.maxIterations),
    accuracyThreshold(other.accuracyThreshold), gamma(other.gamma),
    distanceType(other.distanceType), assignFlag(other.assignFlag), nThreads(other.nThreads)
{}

void Parameter::setNThreads(size_t num)
{
    nThreads = num;
}

void Parameter::check() const
{
    DAAL_CHECK_EX(nClusters > 0, ErrorIncorrectParameter, ParameterName, nClustersStr());
    DAAL_CHECK_EX(accuracyThreshold >= 0, ErrorIncorrectParameter, ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(gamma >= 0, ErrorIncorrectParameter, ParameterName, gammaStr());
}

} // namespace interface1
} // namespace kmeans
} // namespace algorithm
} // namespace daal

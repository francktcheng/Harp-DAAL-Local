/* file: decision_tree_classification_predict_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Common functions for Decision tree predictions calculation
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __DECISION_TREE_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "threading.h"
#include "daal_defines.h"
#include "algorithm.h"
#include "service_utils.h"
#include "service_data_utils.h"
#include "numeric_table.h"
#include "decision_tree_classification_predict_dense_default_batch.h"
#include "decision_tree_classification_model_impl.h"
#include "decision_tree_impl.i"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace prediction
{
namespace internal
{

using namespace daal::data_management;
using namespace daal::services::internal;
using namespace decision_tree::internal;

template<typename algorithmFPType, CpuType cpu>
services::Status DecisionTreePredictKernel<algorithmFPType, defaultDense, cpu>::
    compute(const NumericTable * x, const classifier::Model * m, NumericTable * y, const daal::algorithms::Parameter * par)
{
    typedef daal::data_feature_utils::internal::SignBit<algorithmFPType, cpu> SignBitType;

    DAAL_ASSERT(x);
    DAAL_ASSERT(y);

    const decision_tree::classification::Parameter * const parameter = static_cast<const decision_tree::classification::Parameter *>(par);
    const decision_tree::classification::Model * const model = static_cast<const decision_tree::classification::Model *>(m);

    DAAL_ASSERT(parameter);
    DAAL_ASSERT(model);

    FeatureTypesCache featureTypesCache(*x);
    const DecisionTreeTable & treeTable = *(model->impl()->getTreeTable());
    const DecisionTreeNode * const nodes = static_cast<const DecisionTreeNode *>(treeTable.getArray());
    DAAL_ASSERT(treeTable.getNumberOfRows());

    const size_t xRowCount = x->getNumberOfRows();
    const size_t xColumnCount = x->getNumberOfColumns();
    const size_t yColumnCount = y->getNumberOfColumns();
    DAAL_ASSERT(xRowCount == y->getNumberOfRows());

    const auto rowsPerBlock = 512;
    const auto blockCount = (xRowCount + rowsPerBlock - 1) / rowsPerBlock;
    daal::threader_for(blockCount, blockCount, [=, &featureTypesCache, &treeTable](int iBlock)
    {
        const size_t first = iBlock * rowsPerBlock;
        const size_t last = min<cpu>(static_cast<decltype(xRowCount)>(first + rowsPerBlock), xRowCount);

        BlockDescriptor<algorithmFPType> xBD;
        const_cast<NumericTable &>(*x).getBlockOfRows(first, last - first, readOnly, xBD);
        const algorithmFPType * const dx = xBD.getBlockPtr();
        BlockDescriptor<algorithmFPType> yBD;
        y->getBlockOfRows(first, last - first, writeOnly, yBD);
        auto * const dy = yBD.getBlockPtr();
        for (size_t i = 0; i < last - first; ++i)
        {
            const algorithmFPType * const xRow = &dx[i * xColumnCount];
            size_t nodeIndex = 0;
            DAAL_ASSERT(nodeIndex < treeTable.getNumberOfRows());
            while (nodes[nodeIndex].dimension != static_cast<size_t>(-1))
            {
                switch (featureTypesCache[nodes[nodeIndex].dimension])
                {
                case data_management::data_feature_utils::DAAL_CATEGORICAL:
                    nodeIndex = nodes[nodeIndex].leftIndexOrClass + ((nodes[nodeIndex].cutPoint == xRow[nodes[nodeIndex].dimension]) ? 0 : 1);
                    break;
                case data_management::data_feature_utils::DAAL_ORDINAL:
                case data_management::data_feature_utils::DAAL_CONTINUOUS:
                    nodeIndex = nodes[nodeIndex].leftIndexOrClass + SignBitType::get(nodes[nodeIndex].cutPoint - xRow[nodes[nodeIndex].dimension]);
                    break;
                default:
                    DAAL_ASSERT(false);
                    break;
                }
                DAAL_ASSERT(nodeIndex < treeTable.getNumberOfRows());
            }
            DAAL_ASSERT(nodeIndex < treeTable.getNumberOfRows());
            dy[i * yColumnCount] = nodes[nodeIndex].leftIndexOrClass;
        }
        y->releaseBlockOfRows(yBD);
        const_cast<NumericTable &>(*x).releaseBlockOfRows(xBD);
    } );
    DAAL_RETURN_STATUS();
}

} // namespace internal
} // namespace prediction
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
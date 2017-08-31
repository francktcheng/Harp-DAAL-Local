/* file: parameter.cpp */
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

#include <jni.h>
#include "subgraph_types.i"

#include "JComputeMode.h"
#include "subgraph/JParameter.h"
#include "subgraph/JMethod.h"

#include "common_helpers.h"


USING_COMMON_NAMESPACES()
using namespace daal::algorithms::subgraph;

/*
 * Class:     com_intel_daal_algorithms_subgraph_Parameter
 * Method:    cSetInputTable
 * cSetParameters(this.cObject,learningRate, lambda, Dim_r, Dim_w,  Dim_h, iteration, thread_num, tbb_grainsize, Avx_explicit );
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Parameter_cSetParameters
(JNIEnv *env, jobject thisObj, jlong parAddr, jint iteration, jint thread_num)
{
	((subgraph::Parameter*)parAddr)->_iteration = iteration;
	((subgraph::Parameter*)parAddr)->_thread_num = thread_num;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Parameter_cSetIteration
(JNIEnv *env, jobject thisObj, jlong parAddr, jint itr)
{
	((subgraph::Parameter*)parAddr)->_iteration = itr;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Parameter_cFreeData
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
	((subgraph::Parameter*)parAddr)->freeData();
}

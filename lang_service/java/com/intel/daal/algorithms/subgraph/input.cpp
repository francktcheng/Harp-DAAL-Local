/* file: input.cpp */
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
#include "subgraph/JInput.h"
#include "subgraph/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::subgraph;

/*
 * Class:     com_intel_daal_algorithms_subgraph_Input
 * Method:    cSetInputTable
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cSetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<subgraph::Input>::set<subgraph::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/**
 * @brief read in graph data from HDFS 
 *
 * @param env
 * @param thisObj
 * @param inputAddr
 *
 * @return 
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cReadGraph
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	((subgraph::Input*)inputAddr)->readGraph();
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cInitGraph
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	((subgraph::Input*)inputAddr)->init_Graph();
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cReadTemplate
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	((subgraph::Input*)inputAddr)->readTemplate();
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cInitTemplate
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	((subgraph::Input*)inputAddr)->init_Template();
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cInitNumTable
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	((subgraph::Input*)inputAddr)->create_tables();
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cInitDTTable
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	((subgraph::Input*)inputAddr)->init_DTTable();
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cInitDTSub
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint s)
{
	((subgraph::Input*)inputAddr)->initDtSub(s);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cClearDTSub
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint s)
{
	((subgraph::Input*)inputAddr)->clearDtSub(s);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cSetToTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint src, jint dst)
{
	((subgraph::Input*)inputAddr)->setToTable(src, dst);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cSampleColors
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	((subgraph::Input*)inputAddr)->sampleGraph();
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cInitPartitioner
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	((subgraph::Input*)inputAddr)->init_Partitioner();
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cFreeInput
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	((subgraph::Input*)inputAddr)->free_input();
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cGetReadInThd
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	return (jint)(((subgraph::Input*)inputAddr)->getReadInThd());
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cGetLocalVNum
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	return (jint)(((subgraph::Input*)inputAddr)->getLocalVNum());
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cGetSubVertN
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint sub_itr)
{
	return (jint)(((subgraph::Input*)inputAddr)->getSubVertN(sub_itr));
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cGetSubtemplateCount
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	return (jint)(((subgraph::Input*)inputAddr)->getSubtemplateCount());
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cGetMorphism
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	return (jint)(((subgraph::Input*)inputAddr)->computeMorphism());
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cGetTVNum
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	return (jint)(((subgraph::Input*)inputAddr)->getTVNum());
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cGetTENum
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	return (jint)(((subgraph::Input*)inputAddr)->getTENum());
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cGetLocalMaxV
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	return (jint)(((subgraph::Input*)inputAddr)->getLocalMaxV());
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cSetGlobalMaxV
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
	((subgraph::Input*)inputAddr)->setGlobalMaxV(id);
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cGetLocalADJLen
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
	return (jint)(((subgraph::Input*)inputAddr)->getLocalADJLen());
}
/*
 * Class:     com_intel_daal_algorithms_subgraph_Input
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_subgraph_Input_cGetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<subgraph::Input>::get<subgraph::InputId, NumericTable>(inputAddr, id);
}


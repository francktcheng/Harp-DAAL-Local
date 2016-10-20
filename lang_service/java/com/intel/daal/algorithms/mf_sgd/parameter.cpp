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
#include "mf_sgd_types.i"

#include "JComputeMode.h"
#include "mf_sgd/JParameter.h"
#include "mf_sgd/JMethod.h"

#include "common_helpers.h"


USING_COMMON_NAMESPACES()
using namespace daal::algorithms::mf_sgd;

/*
 * Class:     com_intel_daal_algorithms_mf_sgd_Parameter
 * Method:    cSetInputTable
 * cSetParameters(this.cObject,learningRate, lambda, Dim_r, Dim_w,  Dim_h, iteration, thread_num, tbb_grainsize, Avx512_explicit );
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_mf_1sgd_Parameter_cSetParameters
(JNIEnv *env, jobject thisObj, jlong parAddr, jdouble rate, jdouble lambda, jlong dim_r, jlong dim_w, jlong dim_h, jint iteration, jint thread_num, 
 jint tbb_grainsize, jint avx512_explicit)
{
	((mf_sgd::Parameter*)parAddr)->_learningRate = rate;
	((mf_sgd::Parameter*)parAddr)->_lambda = lambda;
	((mf_sgd::Parameter*)parAddr)->_Dim_r = dim_r;
	((mf_sgd::Parameter*)parAddr)->_Dim_w = dim_w;
	((mf_sgd::Parameter*)parAddr)->_Dim_h = dim_h;
	((mf_sgd::Parameter*)parAddr)->_iteration = iteration;
	((mf_sgd::Parameter*)parAddr)->_thread_num = thread_num;
	((mf_sgd::Parameter*)parAddr)->_tbb_grainsize = tbb_grainsize;
	((mf_sgd::Parameter*)parAddr)->_Avx512_explicit = avx512_explicit;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_mf_1sgd_Parameter_cSetRatio
(JNIEnv *env, jobject thisObj, jlong parAddr, jdouble ratio)
{
	((mf_sgd::Parameter*)parAddr)->_ratio = ratio;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_mf_1sgd_Parameter_cSetIteration
(JNIEnv *env, jobject thisObj, jlong parAddr, jint itr)
{
	((mf_sgd::Parameter*)parAddr)->_itr = itr;
}

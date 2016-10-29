/* file: Parameter.java */
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

/**
 * @brief Contains classes for computing mf_sgd algorithm
 */
package com.intel.daal.algorithms.mf_sgd;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MF_SGD__PARAMETER"></a>
 * @brief Parameter of the mf_sgd_batch algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the parameter for mf_sgd_batch algorithm
     * @param context       Context to manage mf_sgd_batch algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the parameter for mf_sgd_batch algorithm
     * @param context    Context to manage the mf_sgd_batch algorithm
     * @param cObject    Pointer to C++ implementation of the parameter
     */
    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

	
	/**
	 * @brief Set up the parameters 
	 *
	 * @param learningRate  the rate of learning by SGD  
	 * @param lambda  the lambda parameter in standard SGD
	 * @param Dim_r  the feature dimension of model W and H
	 * @param Dim_w  the row num of model W
	 * @param Dim_h  the column num of model H
	 * @param iteration  the iterations of SGD
	 * @param thread_num  specify the threads used by TBB
	 * @param tbb_grainsize  specify the grainsize for TBB parallel_for
	 * @param Avx_explicit  specify whether use explicit Avx instructions
	 *
	 * @return 
	 */
    public void set(double learningRate, double lambda, long Dim_r, long Dim_w, long Dim_h, int iteration, int thread_num, int tbb_grainsize, int Avx_explicit) {
        cSetParameters(this.cObject,learningRate, lambda, Dim_r, Dim_w,  Dim_h, iteration, thread_num, tbb_grainsize, Avx_explicit );
    }

    /**
     * @brief control the percentage of tasks to execute
     *
     * @param ratio
     *
     * @return 
     */
    public void setRatio(double ratio) {
        cSetRatio(this.cObject, ratio);
    }

    /**
     * @brief set the id of training iteration, used in distributed mode
     *
     * @param itr
     *
     * @return 
     */
    public void setIteration(int itr) {
        cSetIteration(this.cObject, itr);
    }

    /**
     * @brief set the id of inner training iteration, used in distributed mode, e.g., model rotation
     *
     * @param innerItr
     *
     * @return 
     */
    public void setInnerItr(int innerItr) {
        cSetInnerItr(this.cObject, innerItr);
    }

    /**
     * @brief total num of inner training iteration, used in distributed mode, e.g., model rotation
     *
     * @param innerNum
     *
     * @return 
     */
    public void setInnerNum(int innerNum) {
        cSetInnerNum(this.cObject, innerNum);
    }

    private native void cSetParameters(long parAddr, double learningRate, double lambda, long Dim_r, long Dim_w, long Dim_h, int iteration, int thread_num, int tbb_grainsize, 
			int Avx_explicit );

    private native void cSetRatio(long parAddr, double ratio);

    private native void cSetIteration(long parAddr, int itr);

    private native void cSetInnerItr(long parAddr, int innerItr);

    private native void cSetInnerNum(long parAddr, int innerNum);

}

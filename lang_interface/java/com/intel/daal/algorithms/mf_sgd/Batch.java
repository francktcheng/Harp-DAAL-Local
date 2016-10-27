/* file: Batch.java */
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
 * @brief Contains classes for computing the MF-SGD-Batch 
 */
package com.intel.daal.algorithms.mf_sgd;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MF_SGD__BATCH"></a>
 * @brief Computes the results of the mf_sgd algorithm in the batch processing mode
 * \n<a href="DAAL-REF-MF-SGD-ALGORITHM">mf_sgd algorithm description and usage models</a>
 *
 * @par References
 *      - Method class.  Computation methods for the mf_sgd algorithm
 *      - InputId class. Identifiers of input objects for the mf_sgd algorithm
 *      - ResultId class. Identifiers of the results of the algorithm
 *      - Input class
 *      - Result class
 */
public class Batch extends AnalysisBatch {

    public Input					  input; /*!< %Input data */
    public Method					  method; /*!< Computation method for the algorithm */
	public Parameter				  parameter;
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the mf_sgd algorithm by copying input objects and parameters
     * of another mf_sgd algorithm
     * @param context   Context to manage the mf_sgd algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
		parameter = new Parameter(getContext(), cGetParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the mf_sgd algorithm
     * @param context   Context to manage the mf_sgd algorithm
     * @param cls       Data type to use in intermediate computations for the mf_sgd algorithm,
     *                  Double.class or Float.class
     * @param method    Computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;

        if (method != Method.defaultSGD) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
		parameter = new Parameter(getContext(), cGetParameter(cObject, prec.getValue(), method.getValue()));

    }

    /**
     * Runs the mf_sgd algorithm
     * @return  Results of the mf_sgd algorithm
     */
    @Override
    public Result compute() {
        super.compute();
		return null;
    }

    /**
     * Registers user-allocated memory for storing the results of the mf_sgd algorithm
     * @param result Structure for storing the results of the mf_sgd algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated mf_sgd algorithm
     * with a copy of input objects and parameters of this mf_sgd algorithm
     * @param context   Context to manage the mf_sgd algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetParameter(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cResult);

    private native long cClone(long algAddr, int prec, int method);
}

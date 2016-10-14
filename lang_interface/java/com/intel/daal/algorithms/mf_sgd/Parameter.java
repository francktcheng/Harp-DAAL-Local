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
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Batch;
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


    // private Batch _function;
    //
    // private native void cSetFunction(long parAddr, long function);
    //
    // private native void cSetNIterations(long parAddr, long nIterations);
    // private native long cGetNIterations(long parAddr);
    //
    // private native void cSetAccuracyThreshold(long parAddr, double accuracyThreshold);
    // private native double cGetAccuracyThreshold(long parAddr);
    //
    // private native void cSetOptionalResultRequired(long parAddr, boolean flag);
    // private native boolean cGetOptionalResultRequired(long parAddr);

}

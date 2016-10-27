/* file: PartialResultId.java */
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

package com.intel.daal.algorithms.mf_sgd;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MF_SGD__PARTIALRESULTID"></a>
 * @brief Available types of the results of the mf_sgd algorithm
 */
public final class PartialResultId {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    public PartialResultId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int presWMatId = 0;
    private static final int presHMatId = 1;

    public static final PartialResultId presWMat = new PartialResultId(presWMatId); /*!< model matrix W */
    public static final PartialResultId presHMat = new PartialResultId(presHMatId); /*!< model matrix H */
}

/* file: InputId.java */
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

package com.intel.daal.algorithms.subgraph;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__subgraph__INPUTID"></a>
 * @brief Available identifiers of input objects for the subgraph algorithm
 */
public final class InputId {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    public InputId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int filenamesId = 0;   
    private static final int fileoffsetId = 1;   
    private static final int localVId = 2;   
    private static final int tfilenamesId = 3;   
    private static final int tfileoffsetId = 4;   

    /** %Input data table */
    public static final InputId filenames = new InputId(filenamesId);
    public static final InputId fileoffset = new InputId(fileoffsetId);
    public static final InputId tfilenames = new InputId(tfilenamesId);
    public static final InputId tfileoffset = new InputId(tfileoffsetId);
    public static final InputId localV = new InputId(localVId);

}

/* file: Input.java */
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

import java.lang.System.*;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;
import java.util.Iterator;
import java.util.Set;
import java.lang.Long;
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__subgraph__INPUT"></a>
 * @brief Input objects for the subgraph algorithm in the batch and distributed mode and for the  
 */
public final class Input extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Input(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets input object for the subgraph algorithm
     * @param id    Identifier of the %input object for the subgraph algorithm
     * @param val   Value of the input object
     */
    public void set(InputId id, NumericTable val) {
        cSetInputTable(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns input object for the subgraph algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(InputId id) {
        return new HomogenNumericTable(getContext(), cGetInputTable(cObject, id.getValue()));
    }

    public void readGraph() {
        cReadGraph(cObject);
    }

    public void readTemplate() {
        cReadTemplate(cObject);
    }

    public void initGraph() {
        cInitGraph(cObject);
    }

    public void initTemplate() {
        cInitTemplate(cObject);
    }

    public void initPartitioner() {
        cInitPartitioner(cObject);
    }

    public void initNumTable() {
        cInitNumTable(cObject);
    }

    public void initDTTable() {
        cInitDTTable(cObject);
    }

    public int getReadInThd() {
        return cGetReadInThd(cObject);
    }

    public int getLocalVNum() {
        return cGetLocalVNum(cObject);
    }
    
    public int getTVNum() {
        return cGetTVNum(cObject);
    }

    public int getTENum() {
        return cGetTENum(cObject);
    }

    public int getLocalMaxV() {
        return cGetLocalMaxV(cObject);
    }

    public int getLocalADJLen() {
        return cGetLocalADJLen(cObject);
    }

    public void setGlobalMaxV(int id) {
        cSetGlobalMaxV(cObject, id);
    }

    public int getSubtemplateCount(){
        return cGetSubtemplateCount(cObject);
    }

    public void freeInput() {
        cFreeInput(cObject);
    }

    private native void cSetInputTable(long cInput, int id, long ntAddr);
    private native long cGetInputTable(long cInput, int id);
    private native void cReadGraph(long cInput);
    private native void cInitGraph(long cInput);
    private native void cReadTemplate(long cInput);
    private native void cInitTemplate(long cInput);
    private native void cInitPartitioner(long cInput);
    private native void cInitNumTable(long cInput);
    private native void cInitDTTable(long cInput);

    private native void cFreeInput(long cInput);
    private native int cGetReadInThd(long cInput);
    private native int cGetLocalVNum(long cInput);
    private native int cGetTVNum(long cInput);
    private native int cGetTENum(long cInput);
    private native int cGetLocalMaxV(long cInput);
    private native int cGetLocalADJLen(long cInput);
    private native int cGetSubtemplateCount(long cInput);
    private native void cSetGlobalMaxV(long cInput, int id);

}

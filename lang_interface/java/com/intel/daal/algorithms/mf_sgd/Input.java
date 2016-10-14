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

package com.intel.daal.algorithms.mf_sgd;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

import java.util.Random;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MF_SGD__INPUT"></a>
 * @brief Input objects for the mf_sgd algorithm in the batch and distributed mode and for the  
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
     * Sets input object for the mf_sgd algorithm
     * @param id    Identifier of the %input object for the mf_sgd algorithm
     * @param val   Value of the input object
     */
    public void set(InputId id, NumericTable val) {
        if (id == InputId.dataTrain || id == InputId.dataTest) {
            cSetInputTable(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns input object for the mf_sgd algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(InputId id) {
        if (id == InputId.dataTrain || id == InputId.dataTest) {
            return new HomogenNumericTable(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    public void generate_points(VPoint[] points_Train, int num_Train, VPoint[] points_Test, int num_Test, int row_num_w, int col_num_h)
    {//{{{

        Random randomGenerator = new Random();

        int counts_train = 0;
        int counts_test = 0;
        float scale = 10;

        int i, j;

        for(i=0;i<row_num_w;i++)
        {
            for (j=0;j<col_num_h;j++) 
            {
                if (i == j)
                {
                    // put diagonal item into train dataset
                    if (counts_train < num_Train)
                    {
                        points_Train[counts_train]._wPos = i;
                        points_Train[counts_train]._hPos = j;

                        points_Train[counts_train]._val = scale*randomGenerator.nextFloat();
                        counts_train++;
                    }
                }
                else
                {
                    if ( randomGenerator.nextFloat() > 0.2)
                    {
                        // put item into train dataset
                        if (counts_train < num_Train)
                        {
                            points_Train[counts_train]._wPos = i;
                            points_Train[counts_train]._hPos = j;

                            points_Train[counts_train]._val = scale*randomGenerator.nextFloat();
                            counts_train++;
                        }
                        else if (counts_test < num_Test)
                        {
                            points_Test[counts_test]._wPos = i;
                            points_Test[counts_test]._hPos = j;

                            points_Test[counts_test]._val = scale*randomGenerator.nextFloat();
                            counts_test++;
                        }
                    }
                    else
                    {
                        // put item into test dataset
                        if (counts_test < num_Test)
                        {
                            points_Test[counts_test]._wPos = i;
                            points_Test[counts_test]._hPos = j;

                            points_Test[counts_test]._val = scale*randomGenerator.nextFloat();
                            counts_test++;
                        }
                        else if (counts_train < num_Train)
                        {
                            points_Train[counts_train]._wPos = i;
                            points_Train[counts_train]._hPos = j;

                            points_Train[counts_train]._val = scale*randomGenerator.nextFloat();
                            counts_train++;
                        }
                    }
                }
            }
        }


    }//}}}

    private native void cSetInputTable(long cInput, int id, long ntAddr);

    private native long cGetInputTable(long cInput, int id);
}

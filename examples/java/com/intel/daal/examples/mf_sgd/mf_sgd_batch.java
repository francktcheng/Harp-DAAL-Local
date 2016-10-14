/* file: mf_sgd_batch.java */
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

/*
 //  Content:
 //     Java example of computing QR decomposition in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-MF_SGD">
 * @example mf_sgd_batch.java
 */

package com.intel.daal.examples.mf_sgd;

import com.intel.daal.algorithms.mf_sgd.*;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.AOSNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

import java.lang.System.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Iterator;
import java.util.Set;
import java.lang.Long;
import java.util.ArrayList;

class mf_sgd_batch{

    /* Input data set parameters */
    private static final String trainDataFile = "../data/batch/movielens-train.mm";
    private static final String testDataFile = "../data/batch/movielens-train.mm";

    private static double learningRate = 0.005;
    private static double lambda = 0.002;
    private static int iteration = 10;		    //num of iterations in SGD training

    private static int threads = 0;			    // automatic threads num by TBB
    private static int tbb_grainsize = 0;         //auto partitioner by TBB  
    private static int Avx512_explicit = 0;       //use compiler generated AVX512 vectorization

    // dimension of model W and model H
    private static int r_dim = 128;

    private static int row_num_w = 1000;
    private static int col_num_w = r_dim;

    private static int row_num_h = r_dim;
    private static int col_num_h = 1000;

    // dimension of training and testing matrices
    private static int num_Train;
    private static int num_Test;
    private static final int field_v = 3;

    // private static final int    nVectors = 100;
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        //command line args 
        if (args.length > 0)
            learningRate = Double.parseDouble(args[0]);
            
        if (args.length > 1)
            lambda = Double.parseDouble(args[1]);
        
        if (args.length > 2)
            iteration = Integer.parseInt(args[2]); 

        if (args.length > 3)
            threads = Integer.parseInt(args[3]); 

        if (args.length > 4)
            tbb_grainsize = Integer.parseInt(args[4]); 

        if (args.length > 5)
            r_dim = Integer.parseInt(args[5]);

        if (args.length > 6)
            Avx512_explicit = Integer.parseInt(args[6]); 

        if (args.length > 7)
            row_num_w = Integer.parseInt(args[7]);

        if (args.length > 8)
            col_num_h = Integer.parseInt(args[8]);

        col_num_w = r_dim;
        row_num_h = r_dim;

        VPoint[] points_Train;
        VPoint[] points_Test;

        /* Create an algorithm to compute mf_sgd_batch  */
        Batch sgdAlgorithm = new Batch(context, Float.class, Method.defaultSGD);

        // if (args.length > 7)
        // {
        //     // generate the dataset
        //     // size of training dataset and test dataset
        //     num_Train = row_num_w + (int)(0.6*(row_num_w*col_num_h - row_num_w));
        //     num_Test = (int)(0.002*(row_num_w*col_num_h- row_num_w));
        //
        //     // generate the Train and Test datasets
        //     points_Train = new VPoint[num_Train];
        //     points_Test = new VPoint[num_Test];
        //
        //     for (int j=0; j<num_Train; j++) 
        //         points_Train[j] = new VPoint(0,0,0);
        //
        //     for (int j=0; j<num_Test; j++) 
        //         points_Test[j] = new VPoint(0,0,0);
        //
        //     System.out.println("Train set num of Points: " + num_Train);
        //     System.out.println("Test set num of Points: " + num_Test);
        //
        //                   
        //     sgdAlgorithm.input.generate_points(points_Train, num_Train, points_Test, num_Test, row_num_w, col_num_h);
        //
        //     System.out.println("Model W Rows: " + row_num_w);
        //     System.out.println("Model H Columns: " + col_num_h);
        //     System.out.println("Model Dimension: " + r_dim);
        //
        // }
        // else
        // {
			//load in the dataset from disk
			HashMap<Long, ArrayList<VPoint> > map_train = new HashMap<Long, ArrayList<VPoint> >();
			HashMap<Long, ArrayList<VPoint> > map_test = new HashMap<Long, ArrayList<VPoint> >();

			num_Train = sgdAlgorithm.input.loadData(trainDataFile, map_train);
			num_Test = sgdAlgorithm.input.loadData(testDataFile, map_test);

			points_Train = new VPoint[num_Train];
			points_Test = new VPoint[num_Test];

			for (int j=0; j<num_Train; j++) 
				points_Train[j] = new VPoint(0,0,0);

			for (int j=0; j<num_Test; j++) 
				points_Test[j] = new VPoint(0,0,0);

			System.out.println("Training points num: " + num_Train);

			int[] row_col_num = sgdAlgorithm.input.convert_format(points_Train, num_Train, points_Test, num_Test, map_train, map_test);

			row_num_w = row_col_num[0];
			col_num_h = row_col_num[1];

			System.out.println("Row num w: " + row_num_w);
			System.out.println("Col num h: " + col_num_h);

        // }
        

        AOSNumericTable dataTable_Train = new AOSNumericTable(context, points_Train);
        AOSNumericTable dataTable_Test = new AOSNumericTable(context, points_Test);
        //
        sgdAlgorithm.input.set(InputId.dataTrain, dataTable_Train);
        sgdAlgorithm.input.set(InputId.dataTest, dataTable_Test);
        //
        sgdAlgorithm.parameter.set(learningRate,lambda, r_dim, row_num_w, col_num_h, iteration, threads, tbb_grainsize, Avx512_explicit);
		//
		Result res = sgdAlgorithm.compute();

        context.dispose();
    }
}

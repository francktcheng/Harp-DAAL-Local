/* file: mf_sgd_dense_default_container.h */
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
//++
//  Implementation of mf_sgd calculation algorithm container.
//--
*/

//#include "mf_sgd.h"
#include "mf_sgd_types.h"
#include "mf_sgd_batch.h"
#include "mf_sgd_default_kernel.h"
#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <math.h>       
#include <random>

#include "numeric_table.h"
#include "service_rng.h"

namespace daal
{
namespace algorithms
{
namespace mf_sgd
{
    
// typedef std::mt19937 CppRNG;

/**
 *  \brief Initialize list of cholesky kernels with implementations for supported architectures
 */
template<typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MF_SGDBatchKernel, interm, method);
}

template<typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
void BatchContainer<interm, method, cpu>::compute()
{
    // prepare the computation

    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);


    NumericTable *a0 = static_cast<NumericTable *>(input->get(dataTrain).get());
    NumericTable *a1 = static_cast<NumericTable *>(input->get(dataTest).get());

    NumericTable **TrainSet = &a0;
    NumericTable **TestSet = &a1;

    NumericTable *r[2];

    r[0] = static_cast<NumericTable *>(result->get(resWMat).get());
    r[1] = static_cast<NumericTable *>(result->get(resHMat).get());

    // initialize mode W and H with random values
    // To to implemented
    size_t w_row = r[0]->getNumberOfRows();
    size_t w_col = r[0]->getNumberOfColumns();

    size_t h_row = r[1]->getNumberOfRows();
    size_t h_col = r[1]->getNumberOfColumns();


    BlockDescriptor<interm> W_Block;
    BlockDescriptor<interm> H_Block;

    r[0]->getBlockOfRows(0, w_row, writeOnly, W_Block); 
    r[1]->getBlockOfRows(0, h_row, writeOnly, H_Block); 

    interm *W_Ptr = W_Block.getBlockPtr();
    interm *H_Ptr = H_Block.getBlockPtr();

    int q;
    interm scale = 1.0/sqrt((interm)w_col);

    /*std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<interm> dis(0, scale);

    for (q = 0; q < w_row*w_col; q++) {
        W_Ptr[q] = dis(gen); 
    }

    for (q = 0; q < h_row*h_col; q++) {
        H_Ptr[q] = dis(gen);
    }*/

	daal::internal::UniformRng<interm, daal::sse2> rng1(time(0));

    rng1.uniform(w_row*w_col, 0.0, scale, W_Ptr);


	daal::internal::UniformRng<interm, daal::sse2> rng2(time(0));
    rng2.uniform(h_row*h_col, 0.0, scale, H_Ptr);

    // debug check the norm of W_Ptr and H_Ptr
    /*interm addres = 0;
    for (q = 0; q < w_row*w_col; q++) {
        addres += (W_Ptr[q]*W_Ptr[q]);
    }

    std::cout<<"W row: "<<w_row<<" W_Ptr norm: "<<addres<<std::endl;*/


    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    /* invoke the MF_SGDBatchKernel */
    __DAAL_CALL_KERNEL(env, internal::MF_SGDBatchKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), compute, TrainSet, TestSet, r, par);

}


}
}
} // namespace daal

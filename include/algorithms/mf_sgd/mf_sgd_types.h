/* file: mf_sgd_types.h */
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
//  Definition of mf_sgd common types.
//--
*/


#ifndef __MF_SGD_TYPES_H__
#define __MF_SGD_TYPES_H__

#include <string>
#include <unordered_map>
#include <vector>
#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{

const int SERIALIZATION_MF_SGD_RESULT_ID = 106000; 

namespace algorithms
{

/**
* @defgroup Matrix factorization Recommender System by using Standard SGD 
* \copydoc daal::algorithms::mf_sgd
* @ingroup mf_sgd
* @{
*/
/** \brief Contains classes for computing the results of the mf_sgd algorithm */
namespace mf_sgd
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__mf_sgd__METHOD"></a>
 * Available methods for computing the mf_sgd algorithm
 */
enum Method
{
    defaultSGD    = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__mf_sgd__INPUTID"></a>
 * Available types of input objects for the mf_sgd decomposition algorithm
 */
enum InputId
{
    dataTrain = 0,      /*!< Input model W */
	dataTest = 1	   /*!< Input model H */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__mf_sgd__RESULTID"></a>
 * Available types of results of the mf_sgd algorithm
 */
enum ResultId
{
    resWMat = 0,   /*!< Output Model W */
    resHMat = 1    /*!< Output Model H */
};

/**
 * @brief A struct for storing sparse matrix data from CSV file 
 */
template<typename interm>
struct VPoint 
{
    long wPos; //abs row id in Model W
    long hPos; //abs column id in Model H
    interm val;
};

struct VPoint_bin 
{
    int wPos;
    int hPos;
    float val;
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__mf_sgd__INPUT"></a>
 * \brief Input objects for the mf_sgd algorithm in the batch and distributed modes 
 * algorithm.
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input();
    /** Default destructor */
    virtual ~Input() {}

    /**
     * Returns input object of the mf_sgd algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets input object for the mf_sgd algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr &value);

    size_t getNumberOfColumns(InputId id) const;

    size_t getNumberOfRows(InputId id) const;


    /**
     * @brief generate an input dataset for mf_sgd 
     *
     * @tparam algorithmFPType
     * @param points_Train
     * @param num_Train
     * @param points_Test
     * @param num_Test
     * @param row_num_w
     * @param col_num_h
     */
    template <typename algorithmFPType>
    void generate_points(daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Train, long num_Train, 
            daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Test, long num_Test,  long row_num_w, long col_num_h);

    /**
     * @brief Convert NumericTablePtr to array of VPoints 
     *
     * @tparam algorithmFPType
     * @param points_Train
     * @param num_Train
     * @param points_Test
     * @param num_Test
     * @param trainTable
     * @param testTable
     */
    template <typename algorithmFPType>
    void convert_format(daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Test, 
            long num_Test, data_management::NumericTablePtr trainTable, data_management::NumericTablePtr testTable, long &row_num_w, long &col_num_h);

    template <typename algorithmFPType>
    void convert_format_binary(daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Test, 
            long num_Test, daal::algorithms::mf_sgd::VPoint_bin* bin_Train, daal::algorithms::mf_sgd::VPoint_bin* bin_Test, long &row_num_w, long &col_num_h);

	template <typename algorithmFPType>
    void convert_format(daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Train, long num_Train, daal::algorithms::mf_sgd::VPoint<algorithmFPType>* points_Test, 
            long num_Test, std::unordered_map<long, std::vector<mf_sgd::VPoint<algorithmFPType>*>*> &map_train, 
			std::unordered_map<long, std::vector<mf_sgd::VPoint<algorithmFPType>*>*> &map_test, long &row_num_w, long &col_num_h);


	/**
	 * @brief load dataset of CSV file 
	 *
	 * @tparam algorithmFPType
	 * @param filename
	 * @param map
	 * @param lineContainer
	 */
    template <typename algorithmFPType>
	void loadData(std::string filename, std::unordered_map<long, std::vector<mf_sgd::VPoint<algorithmFPType>*>*> &map, long &num_points, std::vector<long>* lineContainer);


	/**
	 * @brief free up the memory allocated in the map 
	 *
	 * @tparam algorithmFPType
	 * @param map
	 */
	template <typename algorithmFPType>
	void freeData(std::unordered_map<long, std::vector<mf_sgd::VPoint<algorithmFPType>*>*> &map);


    /**
     * Checks parameters of the algorithm
     * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;


};



/**
 * <a name="DAAL-CLASS-ALGORITHMS__mf_sgd__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the mf_sgd decomposition algorithm
 *        in the batch processing mode or finalizeCompute() method of algorithm in the online processing mode
 *        or on the second and third steps of the algorithm in the distributed processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result();
    /** Default destructor */
    virtual ~Result() {}

    /**
     * Returns the result of the mf_sgd decomposition algorithm
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Allocates memory for storing final results of the mf_sgd decomposition algorithm
     * implemented in mf_sgd_default_batch.h
     * \param[in] input     Pointer to input object
     * \param[in] parameter Pointer to parameter
     * \param[in] method    Algorithm method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);


    /**
    * Sets an input object for the mf_sgd decomposition algorithm
    * \param[in] id    Identifier of the result
    * \param[in] value Pointer to the result
    */
    void set(ResultId id, const data_management::NumericTablePtr &value);

    /**
       * Checks final results of the algorithm
      * \param[in] input  Pointer to input objects
      * \param[in] par    Pointer to parameters
      * \param[in] method Computation method
      */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Allocates memory for storing final results of the mf_sgd decomposition algorithm
     * \tparam     algorithmFPType  Data type to be used for storage in resulting HomogenNumericTable
     * \param[in]  r  dimension of feature vector, num col of model W and num row of model H 
     * \param[in]  w  Number of rows in the model W 
     * \param[in]  h  Number of cols in the model H 
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocateImpl(size_t r, size_t w, size_t h);

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_MF_SGD_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};


/**
 * <a name="DAAL-STRUCT-ALGORITHMS__mf_sgd__PARAMETER"></a>
 * \brief Parameters for the mf_sgd decomposition compute method
 */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{

    Parameter() 
    {
        _learningRate = 0.0;
        _lambda = 0.0;
        _Dim_r = 10;
        _Dim_w = 10;
        _Dim_h = 10;
        _iteration = 0;
        _thread_num = 10;
        _tbb_grainsize = 1000;
        _isAvx512 = 0;
    }

    virtual ~Parameter() {}

    /**
     * Checks the correctness of the parameter
     */
    // virtual void check() const;
    //
    void setParameter(double learningRate, double lambda, long Dim_r, long Dim_w, long Dim_h, int iteration, int thread_num, int tbb_grainsize, int isAvx512)
    {

        _learningRate = learningRate;
        _lambda = lambda;
        _Dim_r = Dim_r;
        _Dim_w = Dim_w;
        _Dim_h = Dim_h;
        _iteration = iteration;
        _thread_num = thread_num;
        _tbb_grainsize = tbb_grainsize;
        _isAvx512 = isAvx512;
    }

    double _learningRate;                     // the rate of learning by SGD 
    double _lambda;                           // the lambda parameter in standard SGD
    long    _Dim_r;                           //the feature dimension of model W and H
    long    _Dim_w;                           //the row num of model W
    long    _Dim_h;                           //the column num of model H
    int    _iteration;                       //the iterations of SGD
    int     _thread_num;                      //specify the threads used by TBB
    int     _tbb_grainsize;                   //specify the grainsize for TBB parallel_for
    int     _isAvx512;                       //specify whether enable Avx512 of mic

};
/** @} */
/** @} */
} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::Parameter;

} // namespace daal::algorithms::mf_sgd
} // namespace daal::algorithms
} // namespace daal

#endif

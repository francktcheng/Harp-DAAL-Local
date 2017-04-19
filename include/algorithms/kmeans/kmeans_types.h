/* file: kmeans_types.h */
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
//  Implementation of the K-Means algorithm interface.
//--
*/

#ifndef __KMEANS_TYPES_H__
#define __KMEANS_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup kmeans_compute Computation
 * \copydoc daal::algorithms::kmeans
 * @ingroup kmeans
 * @{
 */
/** \brief Contains classes of the K-Means algorithm */
namespace kmeans
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__METHOD"></a>
 * Available methods of the K-Means algorithm
 */
enum Method
{
    lloydDense = 0,     /*!< Default: performance-oriented method, synonym of defaultDense */
    defaultDense = 0,   /*!< Default: performance-oriented method, synonym of lloydDense */
    lloydCSR = 1        /*!< Implementation of the Lloyd algorithm for CSR numeric tables */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__DISTANCETYPE"></a>
 * Supported distance types
 */
enum DistanceType
{
    euclidean /*!< Euclidean distance */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INPUTID"></a>
 * \brief Available identifiers of input objects for the K-Means algorithm
 */
enum InputId
{
    data = 0,            /*!< %Input data table */
    inputCentroids = 1 /*!< Initial centroids for the algorithm */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__MASTERINPUTID"></a>
 * \brief Available identifiers of input objects for the K-Means algorithm in the distributed processing mode
 */
enum MasterInputId
{
    partialResults = 0   /*!< Collection of partial results computed on local nodes  */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__PARTIALRESULTID"></a>
 * \brief Available identifiers of partial results of the K-Means algorithm in the distributed processing mode
 */
enum PartialResultId
{
    nObservations       = 0,  /*!< Table containing the number of observations assigned to centroids */
    partialSums         = 1,  /*!< Table containing the sum of observations assigned to centroids */
    partialGoalFunction = 2,  /*!< Table containing a goal function value */
    partialAssignments  = 3   /*!< Table containing assignments of observations to particular clusters */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__RESULTID"></a>
 * \brief Available identifiers of results of the K-Means algorithm
 */
enum ResultId
{
    centroids    = 0, /*!< Table containing cluster centroids */
    assignments  = 1, /*!< Table containing assignments of observations to particular clusters */
    goalFunction = 2, /*!< Table containing a goal function value */
    nIterations  = 3  /*!< Table containing the number of executed iterations */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__KMEANS__PARAMETER"></a>
 * \brief Parameters for the K-Means algorithm
 * \par Enumerations
 *      - \ref DistanceType Methods for distance computation
 *
 * \snippet kmeans/kmeans_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Constructs parameters of the K-Means algorithm
     *  \param[in] _nClusters   Number of clusters
     *  \param[in] _maxIterations Number of iterations
     */
    Parameter(size_t _nClusters, size_t _maxIterations);

    /**
     *  Constructs parameters of the K-Means algorithm by copying another parameters of the K-Means algorithm
     *  \param[in] other    Parameters of the K-Means algorithm
     */
    Parameter(const Parameter &other);

    size_t nClusters;                                      /*!< Number of clusters */
    size_t maxIterations;                                  /*!< Number of iterations */
    double accuracyThreshold;                              /*!< Threshold for the termination of the algorithm */
    double gamma;                                          /*!< Weight used in distance computation for categorical features */
    DistanceType distanceType;                             /*!< Distance used in the algorithm */
    bool assignFlag;                                       /*!< Do data points assignment */
    size_t nThreads;                                       /*!< specified number of tbb threads */

    void setNThreads(size_t num);
    void check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INPUTIFACE"></a>
 * \brief Interface for input objects for the the K-Means algorithm in the batch and distributed processing modes
 */
class DAAL_EXPORT InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {};

    virtual size_t getNumberOfFeatures() const = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INPUT"></a>
 * \brief %Input objects for the K-Means algorithm
 */
class DAAL_EXPORT Input : public InputIface
{
public:
    Input();
    virtual ~Input() {}

    /**
    * Returns an input object for the K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Sets an input object for the K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(InputId id, const data_management::NumericTablePtr &ptr);


    /**
    * Returns the number of features in the input object
    * \return Number of features in the input object
    */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Checks input objects for the K-Means algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__PARTIALRESULT"></a>
 * \brief Partial results obtained with the compute() method of the K-Means algorithm in the batch processing mode
 */
class DAAL_EXPORT PartialResult : public daal::algorithms::PartialResult
{
public:
    PartialResult();

    virtual ~PartialResult() {};

    /**
     * Allocates memory to store partial results of the K-Means algorithm
     * \param[in] input        Pointer to the structure of the input objects
     * \param[in] parameter    Pointer to the structure of the algorithm parameters
     * \param[in] method       Computation method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns a partial result of the K-Means algorithm
     * \param[in] id   Identifier of the partial result
     * \return         Partial result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(PartialResultId id) const;

    /**
     * Sets a partial result of the K-Means algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the object
     */
    void set(PartialResultId id, const data_management::NumericTablePtr &ptr);

    /**
    * Returns the number of features in the Input data table
    * \return Number of features in the Input data table
    */

    size_t getNumberOfFeatures() const;

    /**
    * Checks partial results of the K-Means algorithm
    * \param[in] input   %Input object of the algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks partial results of the K-Means algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
     /**
     * Returns the serialization tag of a partial result
     * \return         Serialization tag of the partial result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_KMEANS_PARTIAL_RESULT_ID; }

    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes an object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__RESULT"></a>
 * \brief Results obtained with the compute() method of the K-Means algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the K-Means algorithm
     * \param[in] input     Pointer to the structure of the input objects
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Allocates memory to store the results of the K-Means algorithm
     * \param[in] partialResult Pointer to the partial result structure
     * \param[in] parameter     Pointer to the structure of the algorithm parameters
     * \param[in] method        Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns the result of the K-Means algorithm
     * \param[in] id   Result identifier
     * \return         Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the K-Means algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the object
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
    * Checks the result of the K-Means algorithm
    * \param[in] input   %Input objects for the algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks the results of the K-Means algorithm
    * \param[in] pres    Partial results of the algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::PartialResult *pres, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

     /**
     * Returns the serialization tag of the result
     * \return         Serialization tag of the result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_KMEANS_RESULT_ID; }

    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes an object
    *  \param[in]  arch  Storage for a deserialized object or data structure
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
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * \brief %Input objects for the K-Means algorithm in the distributed processing mode
 */
class DAAL_EXPORT DistributedStep2MasterInput : public InputIface
{
public:
    DistributedStep2MasterInput();

    virtual ~DistributedStep2MasterInput() {}

    /**
    * Returns an input object for the K-Means algorithm in the second step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::DataCollectionPtr get(MasterInputId id) const;

    /**
    * Sets an input object for the K-Means algorithm in the second step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(MasterInputId id, const data_management::DataCollectionPtr &ptr);

    /**
     * Adds partial results computed on local nodes to the input for the K-Means algorithm
     * in the second step of the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the object
     */
    void add(MasterInputId id, const services::SharedPtr<PartialResult> &value);

    /**
    * Returns the number of features in the Input data table in the second step of the distributed processing mode
    * \return Number of features in the Input data table
    */

    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Checks an input object for the K-Means algorithm in the second step of the distributed processing mode
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::InputIface;
using interface1::Input;
using interface1::PartialResult;
using interface1::Result;
using interface1::DistributedStep2MasterInput;

} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
#endif

/* file: subgraph_types.h */
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
//  Definition of subgraph common types.
//--
*/


#ifndef __SUBGRAPH_TYPES_H__
#define __SUBGRAPH_TYPES_H__

#include <cstdio> 
#include <vector>
#include <string>
#include <cstring>
#include <set>

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "services/hdfs.h"


namespace daal
{

//further change the id values
const int SERIALIZATION_SUBGRAPH_RESULT_ID = 106001; 
const int SERIALIZATION_SUBGRAPH_DISTRI_PARTIAL_RESULT_ID = 106101; 

namespace algorithms
{

/**
* @defgroup color coding based subgraph counting 
* \copydoc daal::algorithms::subgraph
* @ingroup subgraph
* @{
*/
/** \brief Contains classes for computing the results of the subgraph algorithm */
namespace subgraph
{

    

/**
 * <a name="DAAL-ENUM-ALGORITHMS__subgraph__METHOD"></a>
 * Available methods for computing the subgraph algorithm
 */
enum Method
{
    defaultSC    = 0 /*!< Default Standard color coding subgraph counting */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__subgraph__INPUTID"></a>
 * Available types of input objects for the subgraph algorithm
 */
enum InputId
{
    filenames = 0,		  
	fileoffset = 1,
    localV = 2,
    tfilenames =3,
    tfileoffset = 4
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__subgraph__RESULTID"></a>
 * Available types of results of the subgraph algorithm
 */
enum ResultId
{
    resWMat = 0,   /*!< Model W */
    resHMat = 1    /*!< Model H */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__subgraph__DISTRIBUTED_RESULTID"></a>
 * Available types of partial results of the subgraph algorithm
 */
enum DistributedPartialResultId
{
    presWMat = 0,   /*!< Model W, used in distributed mode */
    presHMat = 1,   /*!< Model H, used in distributed mode*/
    presRMSE = 2,   /*!< RMSE computed from test dataset */
    presWData = 3
};


/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
    
    /**
     * @brief store abs v id and its adj list
     */
    struct v_adj_elem{

        v_adj_elem(int v_id)
        {
            _v_id = v_id;
        }

        ~v_adj_elem()
        {
            std::vector<int>().swap(_adjs);
        }

        int _v_id = 0;
        std::vector<int> _adjs;

    };

    struct Graph
    {

        Graph(){}
        void initTemplate(int ng, int mg, int* src, int* dst);
        void freeMem();
        //copy the graph (templates)
        Graph& operator= (const Graph& param);

        int* adjacent_vertices(int v){return &(adj_index_table[v]->_adjs)[0];}
        int* adjacent_vertices_abs(int v)
        {
            //from abs id to rel id
            int rel_id = vertex_local_ids[v];
            if (rel_id >= 0)
                return &(adj_index_table[rel_id]->_adjs)[0];
            else
                return NULL;
        }

        int get_relative_v_id(int v_abs){return vertex_local_ids[v_abs];}
        int out_degree(int v){ return (adj_index_table[v]->_adjs).size();}
        int out_degree_abs(int v)
        {
            //from abs id to rel id
            int rel_id = vertex_local_ids[v];
            if (rel_id >=0)
                return (adj_index_table[rel_id]->_adjs).size();
            else
                return -1;
        }

        int max_degree(){return max_deg;}
        int* get_abs_v_ids(){return vertex_ids;}
        
        int vert_num_count = 0;
        int max_v_id_local = 0;
        int max_v_id = 0; //global max_v_id
        int adj_len = 0;
        int num_edges = 0;
        int max_deg = 0;
        int* vertex_ids = NULL; // absolute v_id
        int* vertex_local_ids = NULL; // mapping from absolute v_id to relative v_id

        v_adj_elem** adj_index_table = NULL; //a table to index adj list for each local vert

        bool isTemplate = false;

    };
    
    class partitioner {

        public:
            partitioner(){}
            partitioner(Graph& t, bool label, int* label_map);
            void sort_subtemplates();
            void clear_temparrays();

        private:

            void init_arrays(){subtemplates_create = new Graph[create_size];}
            //Do a bubble sort based on each subtemplate's parents' index
            //This is a simple way to organize the partition tree for use
            // with dt.init_sub() and memory management of dynamic table
            bool sub_count_needed(int s){ return count_needed[s];}
            Graph* get_subtemplates(){ return subtemplates; }
            int get_subtemplate_count(){ return subtemplate_count;}

            int* get_labels(int s)
            {
                if (labeled)
                    return label_maps[s];
                else
                    return NULL;
            }

            int get_active_index(int a){ return active_children[a];}
            int get_passive_index(int p){return passive_children[p];}
            int get_num_verts_active(int s){return subtemplates[active_children[s]].vert_num_count;}
            int get_num_verts_passive(int s){return subtemplates[passive_children[s]].vert_num_count;}

            void partition_recursive(int s, int root);
            int* split(int s, int root);
            int split_sub(int s, int root, int other_root);
            void fin_arrays();
            void check_nums(int root, std::vector<int>& srcs, std::vector<int>& dsts, int* labels, int* labels_sub);

            void set_active_child(int s, int a)
            {
                while( active_children.size() <= s)
                    active_children.push_back(null_val);

                active_children[s] = a;
            }

            void set_passive_child(int s, int p)
            {
                while(passive_children.size() <= s)
                    passive_children.push_back(null_val);

                passive_children[s] = p;
            }

            void set_parent(int c, int p)
            {
                while(parents.size() <= c )
                    parents.push_back(null_val);

                parents[c]= p;
            }

            Graph* subtemplates_create;
            Graph* subtemplates;
            Graph subtemplate;

            std::vector<int> active_children;
            std::vector<int> passive_children;
            std::vector<int> parents;
            std::vector<int> cut_edge_labels;
            std::vector<int*> label_maps;

            int current_creation_index;
            int subtemplate_count;

            bool* count_needed;
            bool labeled;

            int null_val = 2147483647;  //integer max 
            int create_size = 100;

    };

    /**
     * @brief store sub-template chains for
     * dynamic programming
     */
    struct dynamic_table_array{

        dynamic_table_array()
        {

        }

    };

/**
 * <a name="DAAL-CLASS-ALGORITHMS__subgraph__INPUT"></a>
 * \brief Input objects for the subgraph algorithm in the batch and distributed modes 
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
     * Returns input object of the subgraph algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets input object for the subgraph algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr &value);

	/**
	 * @brief get the column num of NumericTable associated to an inputid
	 *
	 * @param[in] id of input table
	 * @return column num of input table 
	 */
    size_t getNumberOfColumns(InputId id) const;

	/**
	 * @brief get the column num of NumericTable associated to an inputid
	 *
	 * @param[in]  id of input table
	 *
	 * @return row num of input table 
	 */
    size_t getNumberOfRows(InputId id) const;

    daal::services::interface1::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    // input func for read in data from HDFS
    void readGraph();
    void readTemplate();
    void free_input();

    void init_Graph();
    void init_Template();

    size_t getReadInThd();
    size_t getLocalVNum();
    size_t getLocalMaxV();
    size_t getLocalADJLen();
    size_t getTVNum();
    size_t getTENum();
    void setGlobalMaxV(size_t id);

    

private:

    // thread in read in graph
    int thread_num = 0;
    std::vector<v_adj_elem*>* v_adj = NULL; //store data from reading data

    Graph g; // graph data
    Graph t; // template data
    
    // for template data
    size_t t_ng = 0;
    size_t t_mg = 0;
    std::vector<int> t_src;
    std::vector<int> t_dst;

    hdfsFS* fs = NULL;
    int* fileOffsetPtr = NULL;
    int* fileNamesPtr = NULL;

    // table for dynamic programming
    partitioner* part = NULL;
    // dynamic_table_array* dt = NULL;
   
};



/**
 * <a name="DAAL-CLASS-ALGORITHMS__subgraph__RESULT"></a>
 * \brief Provides methods to access results obtained with the compute() method of the subgraph algorithm
 *        in the batch processing mode 
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result();
    /** Default destructor */
    virtual ~Result() {}

    /**
     * Returns the result of the subgraph algorithm
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Allocates memory for storing final results of the subgraph algorithm
     * implemented in subgraph_default_batch.h
     * \param[in] input     Pointer to input object
     * \param[in] parameter Pointer to parameter
     * \param[in] method    Algorithm method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);


    template <typename algorithmFPType>
    DAAL_EXPORT void free_mem(size_t r, size_t w, size_t h);

    /**
    * Sets an input object for the subgraph algorithm
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
    daal::services::interface1::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Allocates memory for storing final results of the subgraph algorithm
     * \tparam     algorithmFPType float or double 
     * \param[in]  r  dimension of feature vector, num col of model W and num row of model H 
     * \param[in]  w  Number of rows in the model W 
     * \param[in]  h  Number of cols in the model H 
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocateImpl(size_t r, size_t w, size_t h);

    template <typename algorithmFPType>
    DAAL_EXPORT void freeImpl(size_t r, size_t w, size_t h);

    template <typename algorithmFPType>
    DAAL_EXPORT void allocateImpl_cache_aligned(size_t r, size_t w, size_t h);

    template <typename algorithmFPType>
    DAAL_EXPORT void freeImpl_cache_aligned(size_t r, size_t w, size_t h);

    template <typename algorithmFPType>
    DAAL_EXPORT void allocateImpl_hbw_mem(size_t r, size_t w, size_t h);

    template <typename algorithmFPType>
    DAAL_EXPORT void freeImpl_hbw_mem(size_t r, size_t w, size_t h);

	/**
	 * @brief get a serialization tag for result
	 *
	 * @return serilization code  
	 */
    int getSerializationTag() const DAAL_C11_OVERRIDE  { return SERIALIZATION_SUBGRAPH_RESULT_ID; }

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
 * <a name="DAAL-CLASS-ALGORITHMS__subgraph__RESULT"></a>
 * \brief Provides methods to access results obtained with the compute() method of the subgraph algorithm
 *        in the batch processing mode or finalizeCompute() method of algorithm in the online processing mode
 *        or on the second and third steps of the algorithm in the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResult : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    DistributedPartialResult();
    /** Default destructor */
    virtual ~DistributedPartialResult() {}

    /**
     * Returns the result of the subgraph algorithm 
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
	data_management::NumericTablePtr get(DistributedPartialResultId id) const;

    /**
     * Sets Result object to store the result of the subgraph algorithm
     * \param[in] id    Identifier of the result
     * \param[in] value Pointer to the Result object
     */
    void set(DistributedPartialResultId id, const data_management::NumericTablePtr &value);


	/**
	 * Checks partial results of the algorithm
	 * \param[in] parameter Pointer to parameters
	 * \param[in] method Computation method
	 */
    daal::services::interface1::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    /**
      * Checks final results of the algorithm
      * \param[in] input      Pointer to input objects
      * \param[in] parameter  Pointer to parameters
      * \param[in] method     Computation method
      */
    daal::services::interface1::Status check(const daal::algorithms::Input* input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

	/**
	 * @brief get serilization tag for partial result
	 *
	 * @return serilization code for partial result
	 */
    int getSerializationTag() const DAAL_C11_OVERRIDE  { return SERIALIZATION_SUBGRAPH_DISTRI_PARTIAL_RESULT_ID;}

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


    void init_model(int threads);

    // member for counts container for threads
    // must use double to avoid overflow of count num of large datasets/templates
    // len is thread number
    int thread_num = 0;
    double* cc_ato = NULL;
    double* count_local_root = NULL;
    double* count_comm_root = NULL;

    
protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__subgraph__PARAMETER"></a>
 * \brief Parameters for the subgraph compute method
 * used in both of batch mode and distributed mode
 */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{

	/* default constructor */
    Parameter() 
    {
        _iteration = 1;
        _thread_num =10;
    }

    virtual ~Parameter() {}

	/**
	 * @brief set the parameters in both of batch mode and distributed mode 
	 *
	 */
    void setParameter(size_t iteration, size_t thread_num)
    {
        _iteration = iteration;
        _thread_num = thread_num;
    }

	/**
	 * @brief set the iteration id, 
	 * used in distributed mode
	 *
	 * @param itr
	 */
    void setIteration(size_t itr)
    {
        _iteration = itr;
    }

    /**
     * @brief free up the user allocated memory
     */
    void freeData()
    {
    }

    size_t _iteration;                       /* the iterations of SGD */
    size_t _thread_num;                      /* specify the threads used by TBB */

    int max_abs_id = 0;
    int thread_num = 0;
    int core_num = 0;
    int tpc = 0;
    int affinity = 0;
    int do_graphlet_freq = 0;
    int do_vert_output = 0;
    int verbose = 0;

    int* labels_g = NULL;
    int labeled = 0;
    int num_verts_graph = 0;

};
/** @} */
/** @} */
} // namespace interface1

using interface1::Input;
using interface1::Result;
using interface1::DistributedPartialResult;
using interface1::Parameter;

} // namespace daal::algorithms::subgraph
} // namespace daal::algorithms
} // namespace daal

#endif

/* file: subgraph_default_distri_impl.i */
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
//  Implementation of distributed mode subgraph method
//--
*/

#ifndef __SUBGRAPH_KERNEL_DISTRI_IMPL_I__
#define __SUBGRAPH_KERNEL_DISTRI_IMPL_I__

#include <time.h>
#include <math.h>       
#include <algorithm>
#include <cstdlib> 
#include <vector>
#include <iostream>
#include <cstdio> 
#include <algorithm>
#include <ctime>        
#include <omp.h>
#include <immintrin.h>
#include <stdlib.h>

#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "threading.h"
#include "tbb/tick_count.h"
#include "task_scheduler_init.h"
#include "tbb/concurrent_hash_map.h"
#include "tbb/concurrent_vector.h"

#include "blocked_range.h"
#include "parallel_for.h"
#include "queuing_mutex.h"

#include "subgraph_default_impl.i"

using namespace tbb;
using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace subgraph
{
namespace internal
{
    

template <typename interm, daal::algorithms::subgraph::Method method, CpuType cpu>
daal::services::interface1::Status subgraphDistriKernel<interm, method, cpu>::compute(Parameter* &par, Input* &input)
{
    services::Status status;
    int stage = par->_stage;
    
    if (stage == 0)
        computeBottom(par, input);
    else if (stage == 1)
        computeNonBottom(par, input);
    else if (stage == 2)
        updateRemoteCounts(par, input);
    else if (stage == 3)
        updateRemoteCountsPip(par, input);
    else
        ;   

    return status;
}

template <typename interm, daal::algorithms::subgraph::Method method, CpuType cpu>
void subgraphDistriKernel<interm, method, cpu>::computeBottom(Parameter* &par, Input* &input)
{
    std::printf("Start Distrikernel compute bottom\n");
    std::fflush;

    int* colors = input->getColorsG();

    daal::algorithms::subgraph::interface1::dynamic_table_array* dt = input->getDTTable();
    daal::algorithms::subgraph::interface1::partitioner* part = input->getPartitioner();

    int** comb_idx_set = input->comb_num_indexes_set;
    int thread_num = par->_thread_num;
    int num_vert_g = input->getLocalVNum();
    int s = par->_sub_itr;

    //start omp function
    if (thread_num == 0)
        thread_num = omp_get_max_threads();

    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(int v=0;v<num_vert_g;v++)
    {
        int n = colors[v];
        dt->set(v, comb_idx_set[s][n], 1.0);
    }

    std::printf("Finish Distrikernel compute bottom\n");
    std::fflush;

}

template <typename interm, daal::algorithms::subgraph::Method method, CpuType cpu>
void subgraphDistriKernel<interm, method, cpu>::computeNonBottom(Parameter* &par, Input* &input)
{
    std::printf("Start Distrikernel compute nonlast\n");
    std::fflush;

    //setup omp affinity
    // setenv("KMP_AFFINITY","granularity=core,compact",1);
    int set_flag = setenv("KMP_AFFINITY","granularity=fine,compact",1);
    // int set_flag = setenv("KMP_AFFINITY","granularity=core,scatter",1);
    if (set_flag == 0)
    {
        std::printf("omp affinity bind successful\n");
        std::fflush;
    }
    
    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double compute_time = 0;


    clock_gettime(CLOCK_MONOTONIC, &ts1);

    daal::algorithms::subgraph::interface1::dynamic_table_array* dt = input->getDTTable();
    daal::algorithms::subgraph::interface1::partitioner* part = input->getPartitioner();

    int** comb_idx_set = input->comb_num_indexes_set;
    int**** comb_num_idx = input->comb_num_indexes;

    int thread_num = par->_thread_num;
    int num_vert_g = input->getLocalVNum();
    int color_num = input->getColorNum();
    int s = par->_sub_itr;
    int cur_sub_vert = part->get_num_verts_sub(s);
    int active_sub_vert = part->get_num_verts_active(s);
    int cur_a_comb_num = input->choose_table[cur_sub_vert][active_sub_vert];
    int cur_comb_num = input->choose_table[color_num][cur_sub_vert];
    int mapper_num = input->mapper_num;
    services::SharedPtr<int>* update_map = input->update_map;
    services::SharedPtr<int> update_map_size = input->update_map_size;

    daal::algorithms::subgraph::interface1::Graph* g = input->getGraphPtr();

    //start omp function
    if (thread_num == 0)
        thread_num = omp_get_max_threads();

    double total_count_cursub = 0.0;

    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for(int v=0;v<num_vert_g;v++)
    {
        // v is relative v_id from 0 to num_verts -1 
        // std::vector<int> valid_nbrs;
        if( dt->is_vertex_init_active(v))
        {
            int* valid_nbrs = new int[g->out_degree(v)];
            int valid_nbrs_count = 0;

            //adjs is absolute v_id
            int* adjs_abs = g->adjacent_vertices(v);
            int end = g->out_degree(v);

            //indexed by comb number
            //counts of v at active child
            float* counts_a = dt->get_active(v);

            //loop overall its neighbours
            int nbr_comm_itr = 0;
            for(int i = 0; i < end; ++i)
            {
                int adj_i = g->get_relative_v_id(adjs_abs[i]);
                //how to determine whether adj_i is in the current passive table
                if( adj_i >=0 && dt->is_vertex_init_passive(adj_i))
                {
                    // valid_nbrs.push_back(adj_i);
                    valid_nbrs[valid_nbrs_count++] = adj_i;
                }
                if (mapper_num > 1 && adj_i < 0)
                    update_map[v].get()[nbr_comm_itr++] = adjs_abs[i];
            }
            if (mapper_num > 1)
                update_map_size.get()[v] = nbr_comm_itr;

            // if(valid_nbrs.size() != 0)
            if(valid_nbrs_count != 0)
            {
                // for a specific vertex v initialized on active child
                // first loop on different color_combs of cur subtemplate
                for(int n = 0; n < cur_comb_num; ++n)
                {
                    double color_count = 0.0;
                    int* comb_indexes_a = comb_num_idx[0][s][n];
                    int* comb_indexes_p = comb_num_idx[1][s][n];

                    int p = cur_a_comb_num -1;

                    // second loop on different color_combs of active/passive children
                    // a+p == num_combinations_ato 
                    // (total colorscombs for both of active child and pasive child)
                    for(int a = 0; a < cur_a_comb_num; ++a, --p)
                    {
                        float count_a = counts_a[comb_indexes_a[a]];
                        if( count_a > 0)
                        {
                            //third loop on different valid nbrs
                            // for(int i = 0; i < valid_nbrs.size(); ++i)
                            for(int i = 0; i < valid_nbrs_count; ++i)
                            {
                                //validated nbrs already checked to be on passive child
                                color_count += (double)count_a * dt->get_passive(valid_nbrs[i], comb_indexes_p[p]);
                            }
                        }
                    
                    }

                    if( color_count > 0.0)
                    {
                        if(s != 0)
                        {
                            dt->set(v, comb_idx_set[s][n], (float)color_count);
                            #pragma omp atomic
                            total_count_cursub += color_count;
                        }
                        else
                        {
                            #pragma omp atomic
                            total_count_cursub += color_count;
                        }
                    }
                }
            }

            delete[] valid_nbrs;
        }
    }

    std::printf("Finish Distrikernel compute for s: %d\n", s);
    std::fflush;

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    diff = 1000000000L *(ts2.tv_sec - ts1.tv_sec) + ts2.tv_nsec - ts1.tv_nsec;
    compute_time = (double)(diff)/1000000L;

    par->_count_time += compute_time;
    par->_total_counts = total_count_cursub;

    if (s == 0)
    {
        std::printf("Finish Final compute with total count %e\n", total_count_cursub);
        std::printf("Omp total compute time: %f ms\n", par->_count_time);
        std::fflush;
    }

}

template <typename interm, daal::algorithms::subgraph::Method method, CpuType cpu>
void subgraphDistriKernel<interm, method, cpu>::updateRemoteCounts(Parameter* &par, Input* &input)
{
    std::printf("Distrikernel updateRemoteCounts\n");
    std::fflush;

    //setup omp affinity
    // setenv("KMP_AFFINITY","granularity=core,compact",1);
    int set_flag = setenv("KMP_AFFINITY","granularity=fine,compact",1);
    // int set_flag = setenv("KMP_AFFINITY","granularity=core,scatter",1);
    if (set_flag == 0)
    {
        std::printf("omp affinity bind successful\n");
        std::fflush;
    }
    
    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double compute_time = 0;

    daal::algorithms::subgraph::interface1::dynamic_table_array* dt = input->getDTTable();
    daal::algorithms::subgraph::interface1::partitioner* part = input->getPartitioner();

    int thread_num = par->_thread_num;
    int num_vert_g = input->getLocalVNum();
    int sub_id = par->_sub_itr;
    int num_colors = input->getColorNum();

    services::SharedPtr<int>* update_map = input->update_map;
    services::SharedPtr<int> update_map_size = input->update_map_size;

    services::SharedPtr<int>* map_ids_cache_pip = input->map_ids_cache_pip;
    services::SharedPtr<int>* chunk_ids_cache_pip = input->chunk_ids_cache_pip;
    services::SharedPtr<int>* chunk_internal_offsets_cache_pip = input->chunk_internal_offsets_cache_pip;

    BlockDescriptor<int>** update_queue_pos = input->update_queue_pos;
    BlockDescriptor<float>** update_queue_counts = input->update_queue_counts;
    BlockDescriptor<int>** update_queue_index = input->update_queue_index;

    int**** comb_num_indexes = input->comb_num_indexes;
    int** comb_num_indexes_set = input->comb_num_indexes_set;

    daal::algorithms::subgraph::interface1::Graph* g = input->getGraphPtr();

    //move the kernel from input class to here
    int active_child = part->get_active_index(sub_id);
    int passive_child = part->get_passive_index(sub_id);

    int num_combinations_verts_sub = input->choose_table[num_colors][input->num_verts_table[sub_id]];
    int active_index = part->get_active_index(sub_id);
    int num_verts_a = input->num_verts_table[active_index];

    // colorset combinations from active child
    // combination of colors for active child
    int num_combinations_active_ato = input->choose_table[input->num_verts_table[sub_id]][num_verts_a];

    // to calculate chunk size
    int comb_len = dt->get_num_color_set(part->get_passive_index(sub_id)); 

    std::printf("Update Comb len: %d\n", comb_len);
    std::fflush;

    // start update 
    // first loop over local v
    int effect_count = 0;
    double total_update_counts = 0.0;

    float** decompress_counts = new float*[thread_num];
    double** update_at_n = new double*[thread_num];

    for(int i = 0; i<thread_num; i++)
    {
        decompress_counts[i] = new float[comb_len];
        update_at_n[i] = new double[num_combinations_verts_sub];
    }

    // #pragma omp parallel for schedule(static) num_threads(thread_num) 
    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for (int v = 0; v < num_vert_g; ++v) 
    {

        if (dt->is_vertex_init_active(v))
        {
            int thread_id = omp_get_thread_num(); 
            // clean update container
            std::memset(update_at_n[thread_id], 0, num_combinations_verts_sub*sizeof(double));

            int adj_list_size = update_map_size.get()[v];
            if (adj_list_size == 0)
                continue;

            // store the abs adj id for v
            int* adj_list = update_map[v].get(); 
            float* counts_a = dt->get_active(v);

            int* map_ids = map_ids_cache_pip[v].get();
            int* chunk_ids = chunk_ids_cache_pip[v].get(); 
            int* chunk_internal_offsets = chunk_internal_offsets_cache_pip[v].get();

            int compress_interval = 0;

            //second loop over nbrs, decompress adj from Scset
            for(int i = 0; i< adj_list_size; i++)
            {
                int* adj_offset_list = (update_queue_pos[map_ids[i]][chunk_ids[i]]).getBlockPtr();

                // get the offset pos
                int start_pos = adj_offset_list[chunk_internal_offsets[i]];
                int end_pos = adj_offset_list[chunk_internal_offsets[i] + 1];

                if (start_pos != end_pos)
                {

                    // the num of compressed counts
                    compress_interval = end_pos - start_pos;

                    // get the compressed counts list, all nonzero elements
                    float* adj_counts_list = (update_queue_counts[map_ids[i]][chunk_ids[i]]).getBlockPtr();
                    int* adj_index_list = (update_queue_index[map_ids[i]][chunk_ids[i]]).getBlockPtr();

                    // ----------- start decompress process -----------
                    std::memset(decompress_counts[thread_id], 0, comb_len*sizeof(float));

                    for(int x = 0; x< compress_interval; x++)
                        decompress_counts[thread_id][(int)adj_index_list[start_pos + x]] = adj_counts_list[start_pos + x];

                    // ----------- Finish decompress process -----------
                    //third loop over comb_num for cur subtemplate
                    for(int n = 0; n< num_combinations_verts_sub; n++)
                    {
                        // more details
                        int* comb_indexes_a = comb_num_indexes[0][sub_id][n];
                        int* comb_indexes_p = comb_num_indexes[1][sub_id][n];

                        // for passive 
                        int p = num_combinations_active_ato -1;

                        // fourth loop over comb_num for active/passive subtemplates
                        for(int a = 0; a < num_combinations_active_ato; ++a, --p)
                        {
                            float count_a = counts_a[comb_indexes_a[a]];
                            if (count_a > 0)
                            {
                                update_at_n[thread_id][n] += ((double)count_a*decompress_counts[thread_id][comb_indexes_p[p]]);
                            }

                        }

                    } // finish all combination sets for cur template

                } // finish all nonzero adj

            } // finish all adj of a v

            //write upated value 
            double partial_local_counts = 0.0;
            for(int n = 0; n< num_combinations_verts_sub; n++)
            {
                if (sub_id != 0)
                {
                    dt->update_comm(v, comb_num_indexes_set[sub_id][n], (float)update_at_n[thread_id][n]);
                }
                else
                {
                    partial_local_counts += update_at_n[thread_id][n];
                }
            }

            if (sub_id == 0)
            {
                #pragma omp atomic
                total_update_counts += partial_local_counts; 
            }

        } // finishe an active v

    } // finish all the v on thread

    if (decompress_counts != NULL)
    {
        for(int i=0;i<thread_num;i++)
            delete[] decompress_counts[i];

        delete[] decompress_counts;
    }

    if (update_at_n != NULL)
    {
        for(int i=0;i<thread_num;i++)
            delete[] update_at_n[i];

        delete[] update_at_n;
    }

    if (sub_id == 0)
    {
        std::printf("Final updated counts is %e\n", total_update_counts);
        std::fflush;
    }

    par->_update_counts = total_update_counts;

}

template <typename interm, daal::algorithms::subgraph::Method method, CpuType cpu>
void subgraphDistriKernel<interm, method, cpu>::updateRemoteCountsPip(Parameter* &par, Input* &input)
{
    std::printf("Distrikernel updateRemoteCounts\n");
    std::fflush;

    //setup omp affinity
    // setenv("KMP_AFFINITY","granularity=core,compact",1);
    int set_flag = setenv("KMP_AFFINITY","granularity=fine,compact",1);
    // int set_flag = setenv("KMP_AFFINITY","granularity=core,scatter",1);
    if (set_flag == 0)
    {
        std::printf("omp affinity bind successful\n");
        std::fflush;
    }
    
    struct timespec ts1;
	struct timespec ts2;
    int64_t diff = 0;
    double compute_time = 0;

    daal::algorithms::subgraph::interface1::dynamic_table_array* dt = input->getDTTable();
    daal::algorithms::subgraph::interface1::partitioner* part = input->getPartitioner();

    int update_pip_id = par->_pip_id;

    int thread_num = par->_thread_num;
    int num_vert_g = input->getLocalVNum();
    int sub_id = par->_sub_itr;
    int num_colors = input->getColorNum();

    services::SharedPtr<int>* update_map = input->update_map;
    services::SharedPtr<int> update_map_size = input->update_map_size;

    services::SharedPtr<int>* map_ids_cache_pip = input->map_ids_cache_pip;
    services::SharedPtr<int>* chunk_ids_cache_pip = input->chunk_ids_cache_pip;
    services::SharedPtr<int>* chunk_internal_offsets_cache_pip = input->chunk_internal_offsets_cache_pip;

    BlockDescriptor<int>** update_queue_pos = input->update_queue_pos;
    BlockDescriptor<float>** update_queue_counts = input->update_queue_counts;
    BlockDescriptor<int>** update_queue_index = input->update_queue_index;

    int**** comb_num_indexes = input->comb_num_indexes;
    int** comb_num_indexes_set = input->comb_num_indexes_set;

    daal::algorithms::subgraph::interface1::Graph* g = input->getGraphPtr();

    //move the kernel from input class to here
    int active_child = part->get_active_index(sub_id);
    int passive_child = part->get_passive_index(sub_id);

    int num_combinations_verts_sub = input->choose_table[num_colors][input->num_verts_table[sub_id]];
    int active_index = part->get_active_index(sub_id);
    int num_verts_a = input->num_verts_table[active_index];

    // colorset combinations from active child
    // combination of colors for active child
    int num_combinations_active_ato = input->choose_table[input->num_verts_table[sub_id]][num_verts_a];

    // to calculate chunk size
    int comb_len = dt->get_num_color_set(part->get_passive_index(sub_id)); 

    std::printf("Update Comb len: %d\n", comb_len);
    std::fflush;

    // start update 
    // first loop over local v
    int effect_count = 0;
    double total_update_counts = 0.0;

    float** decompress_counts = new float*[thread_num];
    double** update_at_n = new double*[thread_num];

    for(int i = 0; i<thread_num; i++)
    {
        decompress_counts[i] = new float[comb_len];
        update_at_n[i] = new double[num_combinations_verts_sub];
    }

    // #pragma omp parallel for schedule(static) num_threads(thread_num) 
    #pragma omp parallel for schedule(guided) num_threads(thread_num) 
    for (int v = 0; v < num_vert_g; ++v) 
    {

        if (dt->is_vertex_init_active(v))
        {
            int thread_id = omp_get_thread_num(); 
            // clean update container
            std::memset(update_at_n[thread_id], 0, num_combinations_verts_sub*sizeof(double));

            int adj_list_size = update_map_size.get()[v];
            if (adj_list_size == 0)
                continue;

            // store the abs adj id for v
            int* adj_list = update_map[v].get(); 
            float* counts_a = dt->get_active(v);

            int* map_ids = map_ids_cache_pip[v].get();
            int* chunk_ids = chunk_ids_cache_pip[v].get(); 
            int* chunk_internal_offsets = chunk_internal_offsets_cache_pip[v].get();

            int compress_interval = 0;

            //second loop over nbrs, decompress adj from Scset
            for(int i = 0; i< adj_list_size; i++)
            {

                //check if the adj is within the update pipeline
                if (map_ids[i] != update_pip_id || update_queue_pos[map_ids[i]] == NULL)
                {
                    continue;
                }

                int* adj_offset_list = (update_queue_pos[map_ids[i]][chunk_ids[i]]).getBlockPtr();

                // get the offset pos
                int start_pos = adj_offset_list[chunk_internal_offsets[i]];
                int end_pos = adj_offset_list[chunk_internal_offsets[i] + 1];

                if (start_pos != end_pos)
                {

                    // the num of compressed counts
                    compress_interval = end_pos - start_pos;

                    // get the compressed counts list, all nonzero elements
                    float* adj_counts_list = (update_queue_counts[map_ids[i]][chunk_ids[i]]).getBlockPtr();
                    int* adj_index_list = (update_queue_index[map_ids[i]][chunk_ids[i]]).getBlockPtr();

                    // ----------- start decompress process -----------
                    std::memset(decompress_counts[thread_id], 0, comb_len*sizeof(float));

                    for(int x = 0; x< compress_interval; x++)
                        decompress_counts[thread_id][(int)adj_index_list[start_pos + x]] = adj_counts_list[start_pos + x];

                    // ----------- Finish decompress process -----------
                    //third loop over comb_num for cur subtemplate
                    for(int n = 0; n< num_combinations_verts_sub; n++)
                    {
                        // more details
                        int* comb_indexes_a = comb_num_indexes[0][sub_id][n];
                        int* comb_indexes_p = comb_num_indexes[1][sub_id][n];

                        // for passive 
                        int p = num_combinations_active_ato -1;

                        // fourth loop over comb_num for active/passive subtemplates
                        for(int a = 0; a < num_combinations_active_ato; ++a, --p)
                        {
                            float count_a = counts_a[comb_indexes_a[a]];
                            if (count_a > 0)
                            {
                                update_at_n[thread_id][n] += ((double)count_a*decompress_counts[thread_id][comb_indexes_p[p]]);
                            }

                        }

                    } // finish all combination sets for cur template

                } // finish all nonzero adj

            } // finish all adj of a v

            //write upated value 
            double partial_local_counts = 0.0;
            for(int n = 0; n< num_combinations_verts_sub; n++)
            {
                if (sub_id != 0)
                {
                    dt->update_comm(v, comb_num_indexes_set[sub_id][n], (float)update_at_n[thread_id][n]);
                }
                else
                {
                    partial_local_counts += update_at_n[thread_id][n];
                }
            }

            if (sub_id == 0)
            {
                #pragma omp atomic
                total_update_counts += partial_local_counts; 
            }

        } // finishe an active v

    } // finish all the v on thread

    if (decompress_counts != NULL)
    {
        for(int i=0;i<thread_num;i++)
            delete[] decompress_counts[i];

        delete[] decompress_counts;
    }

    if (update_at_n != NULL)
    {
        for(int i=0;i<thread_num;i++)
            delete[] update_at_n[i];

        delete[] update_at_n;
    }

    if (sub_id == 0)
    {
        std::printf("Final updated counts is %e\n", total_update_counts);
        std::fflush;
    }

    par->_update_counts = total_update_counts;

}



} // namespace daal::internal
}
}
} // namespace daal

#endif

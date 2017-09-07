/* file: subgraph_input.cpp */
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
//  Implementation of subgraph classes.
//  non-template func
//--
*/

#include "algorithms/subgraph/subgraph_types.h"
#include "services/env_detect.h"
#include "service_micro_table.h"
#include "services/thpool.h"
#include <cstdlib> 
#include <cstdio> 
#include <cstring> 
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <utility>

#include <pthread.h>
#include <unistd.h>

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace subgraph
{
namespace interface1
{

//aux func used in input
void quicksort(int *arr, const int left, const int right, const int sz);
int partition(int *arr, const int left, const int right);
int* dynamic_to_static(std::vector<int>& arr);
int get_max(std::vector<int> arr1, std::vector<int> arr2);

// input has two numerictables
// 0: filenames
// 1: fileoffsets
Input::Input() : daal::algorithms::Input(5) {}

NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void Input::set(InputId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

size_t Input::getNumberOfColumns(InputId id) const
{
    NumericTablePtr dataTable = get(id);
    if(dataTable)
        return dataTable->getNumberOfColumns();
    else
        return 0;
}

size_t Input::getNumberOfRows(InputId id) const
{
    NumericTablePtr dataTable = get(id);
    if(dataTable)
        return dataTable->getNumberOfRows();
    else
        return 0;
}

daal::services::interface1::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const {services::Status s; return s;}

struct readG_task{

    readG_task(int file_id, hdfsFS* handle, int* fileOffsetPtr, int* fileNamesPtr, std::vector<v_adj_elem*>* v_adj, int* max_v_id_local)
    {
        _file_id = file_id;
        _handle = handle;
        _fileOffsetPtr = fileOffsetPtr;
        _fileNamesPtr = fileNamesPtr;
        _v_adj = v_adj;
        _max_v_id_local = max_v_id_local;
    }

    int _file_id;
    hdfsFS* _handle;
    int* _fileOffsetPtr;
    int* _fileNamesPtr;
    std::vector<v_adj_elem*>* _v_adj;
    int* _max_v_id_local;

};

int partition(int *arr, const int left, const int right) 
{
    const int mid = left + (right - left) / 2;
    const int pivot = arr[mid];
    // move the mid point value to the front.
    std::swap(arr[mid],arr[left]);
    int i = left + 1;
    int j = right;
    while (i <= j) 
    {
        while(i <= j && arr[i] <= pivot) 
        {
            i++;
        }

        while(i <= j && arr[j] > pivot) 
        {
            j--;
        }

        if (i < j) 
        {
            std::swap(arr[i], arr[j]);
        }

    }

    std::swap(arr[i - 1],arr[left]);
    return i - 1;
}

void quicksort(int *arr, const int left, const int right, const int sz)
{

    if (left >= right) {
        return;
    }

    int part = partition(arr, left, right);
    quicksort(arr, left, part - 1, sz);
    quicksort(arr, part + 1, right, sz);

}

int* dynamic_to_static(std::vector<int>& arr)
{
    int* new_array = new int[arr.size()];
    for (size_t i = 0; i < arr.size(); ++i)
        new_array[i] = arr[i];

    return new_array;
}

int get_max(std::vector<int> arr1, std::vector<int> arr2)
{
    int maximum = 0; 
    int size = arr1.size();

    for (int i = 0; i < size; i++)
    {
        if (maximum < arr1[i])
        {
            maximum = arr1[i];
        }
        if (maximum < arr2[i])
        {
            maximum = arr2[i];
        }
    }

    return maximum;
}
/**
 * @brief test thread func for thread pool
 */
void thd_task_read(int thd_id, void* arg)
{

    readG_task* task = (readG_task*)arg;

    int file_id = task->_file_id;

    std::printf("Thread %d working on task file %d\n", thd_id, file_id);
    std::fflush;

    hdfsFS fs_handle = (task->_handle)[thd_id];
    std::vector<v_adj_elem*>* v_adj_table = &((task->_v_adj)[thd_id]);

    int max_id_thd = (task->_max_v_id_local)[thd_id];

    int* file_off_ptr = task->_fileOffsetPtr;
    int* file_name_ptr = task->_fileNamesPtr;
    int len = file_off_ptr[file_id+1] - file_off_ptr[file_id];
    char* file = new char[len+1];
    for(int j=0;j<len;j++)
        file[j] = (char)(file_name_ptr[file_off_ptr[file_id] + j]);

    // terminate char buf
    file[len] = '\0';
    std::printf("FilePath: %s\n", file);
    std::fflush;

    if (hdfsExists(fs_handle, file) < 0)
        return;

    hdfsFile readFile = hdfsOpenFile(fs_handle, file, O_RDONLY, 0, 0, 0);
    int file_size = hdfsAvailable(fs_handle, readFile);
    int buf_size = 70000;
    char* buf = new char[buf_size];

    std::string tail;
    while(1)
    {
    
        std::memset(buf, '\0', buf_size);
        tSize ret = hdfsRead(fs_handle, readFile, (void*)buf, buf_size); 
        // std::printf("FileSize: %d, Read size: %d\n", file_size, (int)ret);
        // std::fflush;

        std::string obj;
        if (tail.empty())
        {
            if (ret > 0)
            {
                // obj.assign(buf, buf_size);
                obj.assign(buf, ret);
                //may cause redundant '\0' if the buf already contains a '\0'
                obj += '\0';

            }
        }
        else
        {
            if (ret > 0)
            {
                // obj.assign(buf, buf_size);
                obj.assign(buf, ret);
                //may cause  redundant '\0' if the buf already contains a '\0'
                obj += '\0';
                
                //remove '\0' at the end of tail
                std::string tail_v;
                tail_v.assign(tail.c_str(), std::strlen(tail.c_str()));
                obj = tail_v + obj;
            }
            else
                obj = tail;

        }

        if (obj.length() == 0)
            break;

        std::istringstream obj_ss(obj);
        std::string elem;
        while(std::getline(obj_ss, elem, '\n'))
        {
            if (obj_ss.eof())
            {
                tail.clear();
                //trim dangling '\0'
                if (elem[0] != '\0')
                    tail = elem;

                if (ret > 0 || elem[0] == '\0')
                    break;
            }

            std::string header;
            std::string nbrs;

            std::istringstream elem_ss(elem);
            elem_ss >> header;
            elem_ss >> nbrs;

            std::istringstream nbrs_ss(nbrs);
            std::string s;

            //check header contains '\0' 
            int v_id_add = std::stoi(header);
            v_adj_elem* add_one = new v_adj_elem(v_id_add);

            while(std::getline(nbrs_ss, s,','))
            {
                add_one->_adjs.push_back(std::stoi(s));
            }

            //load one vert id and nbr list into table
            if (add_one->_adjs.size() > 0)
            {
                v_adj_table->push_back(add_one);
                max_id_thd = v_id_add> max_id_thd? v_id_add: max_id_thd;
            }
            
        }

        //check again
        if (ret <= 0)
            break;
    }

    hdfsCloseFile(fs_handle, readFile);
    // record thread local max v_id
    (task->_max_v_id_local)[thd_id] = max_id_thd;

    delete[] buf;
    delete[] file;

}

/**
 * @brief read in graph in parallel by using pthread
 */
void Input::readGraph()
{
    // use all the avaiable cpus (threads to read in data)
    thread_num = (sysconf(_SC_NPROCESSORS_ONLN));
    fs = new hdfsFS[thread_num];
    for(int i=0;i<thread_num;i++)
        fs[i] = hdfsConnect("default", 0);

    v_adj = new std::vector<v_adj_elem*>[thread_num];
    int* max_v_id_thdl = new int[thread_num];
    std::memset(max_v_id_thdl, 0, thread_num*sizeof(int));

    NumericTablePtr filenames_array = get(filenames);
    NumericTablePtr filenames_offset = get(fileoffset);

    int file_num = filenames_offset->getNumberOfColumns() - 1;

    daal::internal::BlockMicroTable<int, readWrite, sse2> mtFileOffset(filenames_offset.get());
    mtFileOffset.getBlockOfRows(0, 1, &fileOffsetPtr);

    daal::internal::BlockMicroTable<int, readWrite, sse2> mtFileNames(filenames_array.get());
    mtFileNames.getBlockOfRows(0, 1, &fileNamesPtr);

    std::printf("Finish create hdfsFS files\n");
    std::fflush;

    // create the thread pool
    threadpool thpool = thpool_init(thread_num);
    readG_task** task_queue = new readG_task*[file_num];

    for(int i=0;i<file_num;i++)
    {
        task_queue[i] = new readG_task(i, fs, fileOffsetPtr, fileNamesPtr, v_adj, max_v_id_thdl);
        std::printf("Finish create taskqueue %d\n", i);
        std::fflush;

		thpool_add_work(thpool, (void (*)(int,void*))thd_task_read, task_queue[i]);
    }

    thpool_wait(thpool);
	thpool_destroy(thpool);

    //sum up the total v_ids num on this mapper
    g.vert_num_count = 0;
    g.adj_len = 0;

    for(int i = 0; i< thread_num;i++)
    {
        g.vert_num_count += v_adj[i].size();
        for(int j=0;j<v_adj[i].size(); j++)
        {
            g.adj_len += ((v_adj[i])[j])->_adjs.size();
        }
        
        g.max_v_id_local = max_v_id_thdl[i] > g.max_v_id_local? max_v_id_thdl[i] : g.max_v_id_local;
    }

    std::printf("Finish Reading all the vert num: %d, total nbrs: %d, local max vid: %d\n", g.vert_num_count, g.adj_len, g.max_v_id_local);
    std::fflush;

    // free task queue
    for(int i=0;i<file_num;i++)
        delete task_queue[i];

    delete[] task_queue;
    delete[] max_v_id_thdl; 

}

void Input::readTemplate()
{
  
    hdfsFS tfs_handle = hdfsConnect("default", 0);
    NumericTablePtr filenames_array = get(tfilenames);
    NumericTablePtr filenames_offset = get(tfileoffset);

    int file_num = filenames_offset->getNumberOfColumns() - 1;

    int* tfileOffsetPtr = NULL;
    daal::internal::BlockMicroTable<int, readWrite, sse2> mtFileOffset(filenames_offset.get());
    mtFileOffset.getBlockOfRows(0, 1, &tfileOffsetPtr);

    int* tfileNamesPtr = NULL;
    daal::internal::BlockMicroTable<int, readWrite, sse2> mtFileNames(filenames_array.get());
    mtFileNames.getBlockOfRows(0, 1, &tfileNamesPtr);

    std::printf("Finish create hdfsFS files for template\n");
    std::fflush;

    // for single template
    int file_id = 0;
    int len = tfileOffsetPtr[file_id+1] - tfileOffsetPtr[file_id];
    char* file = new char[len+1];
    for(int j=0;j<len;j++)
        file[j] = (char)(tfileNamesPtr[tfileOffsetPtr[file_id] + j]);

    // terminate char buf
    file[len] = '\0';
    std::printf("Template FilePath: %s\n", file);
    std::fflush;

    if (hdfsExists(tfs_handle, file) < 0)
    {
        std::printf("Cannot open file: %s\n", file);
        std::fflush;
    }

    hdfsFile readFile = hdfsOpenFile(tfs_handle, file, O_RDONLY, 0, 0, 0);
    int file_size = hdfsAvailable(tfs_handle, readFile);
    int buf_size = 70000;
    char* buf = new char[buf_size];

    //start read in template data
    std::string tail;
    bool read_ng = false;
    bool read_mg = false;
    while(1)
    {
        std::memset(buf, '\0', buf_size);
        tSize ret = hdfsRead(tfs_handle, readFile, (void*)buf, buf_size); 
        std::printf("FileSize: %d, Read size: %d\n", file_size, (int)ret);
        std::fflush;

        std::string obj;
        if (tail.empty())
        {
            if (ret > 0)
            {
                obj.assign(buf, ret);
                obj += '\0';
            }
        }
        else
        {
            if (ret > 0)
            {
                obj.assign(buf, ret);
                obj += '\0';
                std::string tail_v;
                tail_v.assign(tail.c_str(), std::strlen(tail.c_str()));
                obj = tail_v + obj;
            }
            else
                obj = tail;
        }

        if (obj.length() == 0)
            break;

        //get a batch of file content
        std::istringstream obj_ss(obj);
        std::string elem;
        while(std::getline(obj_ss, elem, '\n'))
        {
            if (obj_ss.eof())
            {
                tail.clear();
                if (elem[0] != '\0')
                    tail = elem;

                if (ret > 0 || elem[0] == '\0')
                    break;
            }

            //parse the line
            std::istringstream elem_ss(elem);

            if (!read_ng)
            {
                std::string ng;
                elem_ss >> ng;
                
                std::printf("t ng: %s\n", ng.c_str());
                std::fflush;

                read_ng = true;
                t_ng = std::stoi(ng);
                continue;
            }
            
            if (!read_mg)
            {
                std::string mg;
                elem_ss >> mg;
                
                std::printf("t mg: %s\n", mg.c_str());
                std::fflush;
                read_mg = true;
                t_mg = std::stoi(mg);
                continue;
            }

            //read the pair of edges
            std::string src_e;
            std::string dst_e;

            elem_ss >> src_e;
            elem_ss >> dst_e;

            t_src.push_back(std::stoi(src_e));
            t_dst.push_back(std::stoi(dst_e));

            std::printf("src dege: %s, dst edge: %s\n", src_e.c_str(), dst_e.c_str());
            std::fflush;
        }

        //check again
        if (ret <= 0)
            break;
    }

    hdfsCloseFile(tfs_handle, readFile);
    hdfsDisconnect(tfs_handle);
}

void Input::init_Template()
{
    t.initTemplate(t_ng, t_mg, &t_src[0], &t_dst[0]);
    part = new partitioner(t, false, NULL);
    part->sort_subtemplates();
    part->clear_temparrays();
}

size_t Input::getTVNum()
{
    return t.vert_num_count;
}

size_t Input::getTENum()
{
    return t.num_edges;
}

/**
 * @brief initialization of internal graph data structure
 */
void Input::init_Graph()
{
    // std::printf("Start init Graph\n");
    // std::fflush;
    //mapping from global v_id starts from 1 to local id
    g.vertex_local_ids = new int[g.max_v_id+1];
    for(int p=0;p<g.max_v_id+1;p++)
        g.vertex_local_ids[p] = -1;

    //undirected graph, num edges equals adj array size
    g.num_edges = g.adj_len;

    //global abs vertex ids
    g.vertex_ids = new int[g.vert_num_count];
    g.max_deg = 0;

    //load v ids from v_adj to vertex_ids
    g.adj_index_table = new v_adj_elem*[g.vert_num_count];

    int itr=0;
    for(int i=0; i<thread_num; i++)
    {
        for(int j=0; j<v_adj[i].size(); j++)
        {
            g.vertex_ids[itr] = ((v_adj[i])[j])->_v_id;
            int v_id = g.vertex_ids[itr];
            g.vertex_local_ids[v_id] = itr;
            g.adj_index_table[itr] = ((v_adj[i])[j]);

            int deg = ((v_adj[i])[j])->_adjs.size();
            g.max_deg = deg > g.max_deg? deg: g.max_deg;
            itr++;
        }
    }

    // load vertex_ids into daal table
    NumericTablePtr localVTable = get(localV);
    if (localVTable != NULL)
    {
        daal::internal::BlockMicroTable<int, writeOnly, sse2> mtlocalVTable(localVTable.get());
        int* localVTablePtr = NULL;
        mtlocalVTable.getBlockOfRows(0, 1, &localVTablePtr);
        memcpy(localVTablePtr, g.vertex_ids, g.vert_num_count*sizeof(int));
        mtlocalVTable.release();
    }

    std::printf("Finish init Graph\n");
    std::fflush;
}

void Input::setGlobalMaxV(size_t id)
{
    g.max_v_id = id;
    std::printf("global vMax id: %d\n", g.max_v_id);
    std::fflush;
}

size_t Input::getReadInThd()
{
    return thread_num;
}

size_t Input::getLocalVNum()
{
    return g.vert_num_count;
}
    
size_t Input::getLocalMaxV()
{
    return g.max_v_id_local;
}
    
size_t Input::getLocalADJLen()
{
    return g.adj_len;
}

void Input::free_input()
{
    if (fs != NULL)
    {
        for(int i=0;i<thread_num;i++)
            hdfsDisconnect(fs[i]);

        delete[] fs;
    }

    if (v_adj != NULL)
    {
        for(int i=0;i<thread_num;i++)
        {
            for(int j=0;j<v_adj[i].size();j++)
            {
                if ((v_adj[i])[j] != NULL)
                    delete (v_adj[i])[j];
            }
        }

        delete[] v_adj;
    }

    g.freeMem();
    t.freeMem();
}

// ------------------------ func for partitioner------------------------
partitioner::partitioner(Graph& t, bool label, int* label_map)
{
    //debug printout graph t
    for(int i = 0; i<t.vert_num_count; i++)
    {
        std::printf("t vert: %d\n", t.vertex_ids[i]);
        std::fflush;
        int* adj_list = t.adjacent_vertices(i);
        int deg = t.out_degree(i);
        for(int j=0;j<deg;j++)
        {
            std::printf("%d,", adj_list[j]);
            std::fflush;
        }
            
        std::printf("\n");
        std::fflush;
    }

    init_arrays();
    labeled = label;
    subtemplates_create[0] = t;
    //start from 1
    current_creation_index = 1;

    if(labeled){
        label_maps.push_back(label_map);
    }

    parents.push_back(null_val);

    int root = 0;

    //recursively partition the template
    partition_recursive(0, root);

    fin_arrays();
}

int partitioner::split_sub(int s, int root, int other_root)
{
    subtemplate = subtemplates_create[s];
    int* labels_sub = NULL;

    if(labeled)
        labels_sub = label_maps[s];

    //source and destination arrays for edges
    std::vector<int> srcs;
    std::vector<int> dsts;

    //get the previous vertex to avoid backtracking
    int previous_vertex = other_root;

    //loop through the rest of the vertices
    //if a new edge is found, add it
    std::vector<int> next;
    //start from the root
    //record all the edges in template execpt for other root branch
    next.push_back(root);
    int cur_next = 0;
    while( cur_next < next.size()){

        int u = next[cur_next++];
        int* adjs = subtemplate.adjacent_vertices(u);
        int end = subtemplate.out_degree(u);

        //loop over all the adjacent vertices of u
        for(int i = 0; i < end; i++){
            int v = adjs[i];
            bool add_edge = true;

            //avoiding add repeated edges
            for(int j = 0; j < dsts.size(); j++){
                if(srcs[j] == v && dsts[j] == u){
                    add_edge = false;
                    break;
                }
            }

            if( add_edge && v != previous_vertex){
                srcs.push_back(u);
                dsts.push_back(v);
                next.push_back(v);
            }
        }
    }

    //if empty, just add the single vert;
    int n;
    int m;
    int* labels = NULL;

    if( srcs.size() > 0){
        m = srcs.size();
        n = m + 1;

        if(labeled)
        {
            //extract_uniques
            std::set<int> unique_ids;
            for(int x = 0; x<srcs.size();x++)
                unique_ids.insert(srcs[x]);

            labels = new int[unique_ids.size()];

        }

        check_nums(root, srcs, dsts, labels, labels_sub);

    }else{
        //single node
        m = 0;
        n = 1;
        srcs.push_back(0);

        if( labeled){
            labels = new int[1];
            labels[0] = labels_sub[root];
        }
    }

    int* srcs_array = dynamic_to_static(srcs);
    int* dsts_array = dynamic_to_static(dsts);

    //create a subtemplate
    subtemplates_create[current_creation_index].initTemplate(n, m, srcs_array, dsts_array);

    if (srcs_array != NULL)
        delete[] srcs_array;

    if (dsts_array != NULL)
        delete[] dsts_array;

    current_creation_index++;

    if(labeled)
        label_maps.push_back(labels);

    return srcs[0];

}

void partitioner::check_nums(int root, std::vector<int>& srcs, std::vector<int>& dsts, int* labels, int* labels_sub)
{
    int maximum = get_max(srcs, dsts);
    int size = srcs.size();

    int* mappings = new int[maximum + 1];

    for(int i = 0; i < maximum + 1; ++i){
        mappings[i] = -1;
    }

    int new_map = 0;
    mappings[root] = new_map++;

    for(int i = 0; i < size; ++i){
        if( mappings[srcs[i]] == -1)
            mappings[srcs[i]] = new_map++;

        if( mappings[dsts[i]] == -1 )
            mappings[dsts[i]] = new_map++;
    }

    for(int i = 0; i < size; ++i){
        srcs[i]  = mappings[srcs[i]];
        dsts[i]  = mappings[dsts[i]];
    }

    if( labeled ){
        for (int i = 0; i < maximum; ++i){
            labels[ mappings[i] ] = labels_sub[i];
        }
    }

    if (mappings != NULL)
        delete[] mappings;

}

void partitioner::sort_subtemplates()
{

    bool swapped;
    Graph temp_g;

    do{
        swapped = false;
        for(int i = 2; i < subtemplate_count; ++i){
            if( parents[i] < parents[i-1]){

                //copy the subtemplate obj
                temp_g = subtemplates[i];
                int temp_pr = parents[i];
                int temp_a = active_children[i];
                int temp_p = passive_children[i];
                int* temp_l = NULL;
                if(labeled)
                    temp_l = label_maps[i];

                //if either have children, need to update their parents
                if( active_children[i] != null_val)
                    parents[active_children[i]] = i-1;

                if(active_children[i-1] != null_val)
                    parents[active_children[i-1]] = i;

                if( passive_children[i] != null_val)
                    parents[passive_children[i]] = i-1;

                if( passive_children[i-1] != null_val)
                    parents[passive_children[i-1]] = i;

                // need to update their parents
                if( active_children[parents[i]] == i)
                    active_children[parents[i]] = i-1;
                else if( passive_children[parents[i]] == i )
                    passive_children[parents[i]] = i-1;

                if( active_children[parents[i-1]] == i-1)
                    active_children[parents[i-1]] = i;
                else if(passive_children[parents[i-1]] == i-1 )
                    passive_children[parents[i-1]] = i;

                // make the switch copy data
                subtemplates[i] = subtemplates[i-1];
                parents[i] =  parents[i-1];
                active_children[i] = active_children[i-1];
                passive_children[i] = passive_children[i-1];

                if(labeled)
                    label_maps[i] =  label_maps[i-1];

                subtemplates[i-1] = temp_g;
                parents[i-1] =  temp_pr;
                active_children[i-1] =  temp_a;
                passive_children[i-1]  = temp_p;
                if(labeled)
                    label_maps[i-1]  = temp_l;

                swapped = true;
            }
        }
    }while(swapped);

}

void partitioner::partition_recursive(int s, int root)
{

    //split the current subtemplate using the current root
    int* roots = split(s, root);

    //debug
    std::printf("Split roots s: %d, r: %d\n", s, root);
    std::fflush;

    //set the parent/child tree structure
    int a = current_creation_index - 2;
    int p = current_creation_index - 1;
    set_active_child(s, a);
    set_passive_child(s, p);
    set_parent(a, s);
    set_parent(p, s);

    //specify new roots and continue partitioning
    int num_verts_a = subtemplates_create[a].vert_num_count;
    int num_verts_p = subtemplates_create[p].vert_num_count;


    if( num_verts_a > 1){
        int activeRoot = roots[0];
        partition_recursive(a, activeRoot);
    }else{
        set_active_child(a, null_val);
        set_passive_child(a, null_val);
    }

    //debug
    std::printf("Finish recursive active %d\n", a);
    std::fflush;

    if( num_verts_p > 1){
        int passiveRoot = roots[1];
        partition_recursive(p, passiveRoot);
    }else{
        set_active_child(p, null_val);
        set_passive_child(p, null_val);
    }

    //debug
    std::printf("Finish recursive passive %d\n", p);
    std::fflush;

    //debug
    std::printf("Finish recursive s: %d, r: %d\n", s, root);
    std::fflush;

}

int* partitioner::split(int s, int root)
{
    //get new root
    //get adjacency vertices of the root vertex
    int* adjs = subtemplates_create[s].adjacent_vertices(root);

    //get the first adjacent vertex
    int u = adjs[0];

    //split this subtemplate between root and node u
    //create a subtemplate rooted at root
    int active_root = split_sub(s, root, u);
    //create a subtemplate rooted at u
    int passive_root = split_sub(s, u, root);

    int* retval = new int[2];
    retval[0] = active_root;
    retval[1] = passive_root;
    return retval;
}

void partitioner::fin_arrays()
{
    subtemplate_count = current_creation_index;
    subtemplates = new Graph[subtemplate_count];

    for(int i = 0; i < subtemplate_count; ++i){
        //copy
        subtemplates[i] = subtemplates_create[i];
        subtemplates_create[i].freeMem();
    }

    delete[] subtemplates_create;
    subtemplates_create = NULL;

    count_needed = new bool[subtemplate_count];
    for(int i = 0; i < subtemplate_count; ++i){
        count_needed[i] = true;
    }

}

void partitioner::clear_temparrays()
{
    if (subtemplates != NULL)
    {
        for(int i=0;i<subtemplate_count;i++)
            subtemplates[i].freeMem();

        delete [] subtemplates;
        subtemplates = NULL;
    }

    if (count_needed != NULL)
    {
        delete [] count_needed;
        count_needed = NULL;
    }
}

// -------------------- impl of Graph struct  --------------------
void Graph::initTemplate(int ng, int mg, int* src, int* dst)
{
    //free memory if the graph is re-init
    freeMem();

    vert_num_count = ng;
    num_edges = 2*mg;
    adj_len = num_edges;
    max_deg = 0;

    vertex_local_ids = new int[vert_num_count+1];
    vertex_ids = new int[vert_num_count];
    adj_index_table = new v_adj_elem*[vert_num_count];
    //initialize adj table
    for(int i=0; i<vert_num_count;i++)
        adj_index_table[i] = new v_adj_elem(0);

    for(int i = 0; i < mg; ++i)
    {
        adj_index_table[src[i]]->_v_id = src[i];
        adj_index_table[src[i]]->_adjs.push_back(dst[i]);

        adj_index_table[dst[i]]->_v_id = dst[i];
        adj_index_table[dst[i]]->_adjs.push_back(src[i]);

        vertex_local_ids[src[i]] = src[i];
        vertex_local_ids[dst[i]] = dst[i];
    }

    for(int i = 0; i < vert_num_count; ++i)
    {
        vertex_ids[i] = i;
        max_deg = adj_index_table[i]->_adjs.size() > max_deg? adj_index_table[i]->_adjs.size() : max_deg;
    }

    isTemplate = true;
}

void Graph::freeMem()
{
    if (vertex_ids != NULL)
    {
        delete[] vertex_ids; // absolute v_id
        vertex_ids = NULL;
    }

    if (vertex_local_ids != NULL)
    {
        delete[] vertex_local_ids; // mapping from absolute v_id to relative v_id
        vertex_local_ids = NULL;
    }

    if (isTemplate && adj_index_table != NULL)
    {
        for(int i=0; i<vert_num_count;i++)
            delete adj_index_table[i];
    }

    if (adj_index_table != NULL)
    {
        delete[] adj_index_table; //a table to index adj list for each local vert
        adj_index_table = NULL;
    }
}

Graph& Graph::operator= (const Graph& param)
{
    this->freeMem();
    vert_num_count = param.vert_num_count;
    max_v_id_local = param.max_v_id_local;
    max_v_id = param.max_v_id;
    adj_len = param.adj_len;
    num_edges = param.num_edges;
    max_deg = param.max_deg;

    vertex_ids = new int[vert_num_count];
    std::memcpy(vertex_ids, param.vertex_ids, vert_num_count*sizeof(int));

    vertex_local_ids = new int[vert_num_count+1];
    std::memcpy(vertex_local_ids, param.vertex_local_ids, (vert_num_count+1)*sizeof(int));

    adj_index_table = new v_adj_elem*[vert_num_count];
    for(int i=0;i<vert_num_count;i++)
    {
        adj_index_table[i] = new v_adj_elem((param.adj_index_table[i])->_v_id);
        adj_index_table[i]->_adjs =  (param.adj_index_table[i])->_adjs;
    }

    isTemplate = param.isTemplate;
    return *this;
}
} // namespace interface1
} // namespace subgraph
} // namespace algorithm
} // namespace daal

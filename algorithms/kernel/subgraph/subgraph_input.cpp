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
// #include "hdfs.h"
// #include "ctpl_stl.h"
// #include <future>
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

// input has two numerictables
// 0: filenames
// 1: fileoffsets
Input::Input() : daal::algorithms::Input(3) {}

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

    // read char one by one
    std::string tail;
    while(1)
    {
    
        std::memset(buf, '\0', buf_size);
        tSize ret = hdfsRead(fs_handle, readFile, (void*)buf, buf_size); 
        std::printf("FileSize: %d, Read size: %d\n", file_size, (int)ret);
        std::fflush;

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
                //check the length of obj
                // int len1 = (int)obj.length();
                // int len2 = std::strlen(obj.c_str());
                // if (len1 != len2)
                //     std::printf("Length of connected obj: %d, c-string len: %d\n", len1, len2);
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
            // int len1 = (int)header.length();
            // int len2 = std::strlen(header.c_str());
            // if (len1 != len2)
            //     std::printf("Length of header: %d, c-string len: %d\n", len1, len2);
            int v_id_add = std::stoi(header);
            v_adj_elem* add_one = new v_adj_elem(v_id_add);

            while(std::getline(nbrs_ss, s,','))
            {
                // while (s[0] == '\0' && s.length() > 0)
                //     s = s.substr(1,s.length()-1);
                //check wehter \0 embedded in s
                // int len1 = (int)s.length();
                // int len2 = std::strlen(s.c_str());
                // if (len1 != len2)
                //      std::printf("Length of nbr: %d, c-string len: %d\n", len1, len2);
                // if (s[0] != '\0')
                    add_one->_adjs.push_back(std::stoi(s));
                // else
                // {
                    // std::printf("empty nbr and len: %d\n", (int)s.length());
                    // std::fflush;
                // }
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
    vert_num_count = 0;
    adj_len = 0;
    for(int i = 0; i< thread_num;i++)
    {
        vert_num_count += v_adj[i].size();
        for(int j=0;j<v_adj[i].size(); j++)
        {
            adj_len += ((v_adj[i])[j])->_adjs.size();
        }
        
        max_v_id_local = max_v_id_thdl[i] > max_v_id_local? max_v_id_thdl[i] : max_v_id_local;
    }

    std::printf("Finish Reading all the vert num: %d, total nbrs: %d, local max vid: %d\n", vert_num_count, adj_len, max_v_id_local);
    std::fflush;

    // free task queue
    for(int i=0;i<file_num;i++)
        delete task_queue[i];

    delete[] task_queue;
    delete[] max_v_id_thdl; 

}


/**
 * @brief initialization of internal graph data structure
 */
void Input::init_Graph()
{
    // std::printf("Start init Graph\n");
    // std::fflush;

    adjacency_array = new int[adj_len];
    //mapping from global v_id starts from 1 to local id
    vertex_local_ids = new int[max_v_id+1];

    for(int p=0;p<max_v_id+1;p++)
        vertex_local_ids[p] = -1;

    //undirected graph, num edges equals adj array size
    num_edges = adj_len;

    degree_list = new int[vert_num_count + 1];
    degree_list[0] = 0;

    //global abs vertex ids
    vertex_ids = new int[vert_num_count];
    int* temp_deg_list = new int[vert_num_count];
    max_deg = 0;

    //load v ids from v_adj to vertex_ids
    int itr=0;
    for(int i=0; i<thread_num; i++)
    {
        for(int j=0; j<v_adj[i].size(); j++)
        {
            vertex_ids[itr] = ((v_adj[i])[j])->_v_id;
            int v_id = vertex_ids[itr];
            vertex_local_ids[v_id] = itr;
            temp_deg_list[itr] = ((v_adj[i])[j])->_adjs.size();
            max_deg = temp_deg_list[itr] > max_deg? temp_deg_list[itr]: max_deg;

            degree_list[itr + 1] = degree_list[itr] + temp_deg_list[itr];
            std::memcpy(adjacency_array+degree_list[itr], &(((v_adj[i])[j])->_adjs)[0], (temp_deg_list[itr])*sizeof(int));
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
        memcpy(localVTablePtr, vertex_ids, vert_num_count*sizeof(int));
        mtlocalVTable.release();
    }

    std::printf("Finish init Graph\n");
    std::fflush;

    if (temp_deg_list != NULL)
        delete[] temp_deg_list;

}

void Input::setGlobalMaxV(size_t id)
{
    max_v_id = id;
    std::printf("global vMax id: %d\n", max_v_id);
    std::fflush;
}

size_t Input::getReadInThd()
{
    return thread_num;
}

size_t Input::getLocalVNum()
{
    return vert_num_count;
}
    
size_t Input::getLocalMaxV()
{
    return max_v_id_local;
}
    
size_t Input::getLocalADJLen()
{
    return adj_len;
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

    if (adjacency_array != NULL)
        delete[] adjacency_array;

    if (degree_list != NULL)
        delete[] degree_list;

    if (vertex_ids != NULL)
        delete[] vertex_ids;

    if (vertex_local_ids != NULL)
        delete[] vertex_local_ids;

}

} // namespace interface1
} // namespace subgraph
} // namespace algorithm
} // namespace daal

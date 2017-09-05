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

#include <pthread.h>

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

// input has two numerictables
// 0: filenames
// 1: fileoffsets
Input::Input() : daal::algorithms::Input(2) {}

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

    readG_task(int file_id, hdfsFS* handle, int* fileOffsetPtr, int* fileNamesPtr, std::vector<v_adj_elem*>* v_adj)
    {
        _file_id = file_id;
        _handle = handle;
        _fileOffsetPtr = fileOffsetPtr;
        _fileNamesPtr = fileNamesPtr;
        _v_adj = v_adj;
    }

    int _file_id;
    hdfsFS* _handle;
    int* _fileOffsetPtr;
    int* _fileNamesPtr;
    std::vector<v_adj_elem*>* _v_adj;

};


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
            }

            obj = tail + obj;
        }

        if (obj.length() == 0)
            break;

        std::istringstream obj_ss(obj);
        std::string elem;
        while(std::getline(obj_ss, elem, '\n'))
        {
            if (obj_ss.eof())
            {
                // std::printf("incomplete line: %s\n", elem.c_str());
                // may contain multiple '\0' 
                tail.clear();
                if (elem[0] != '\0')
                {
                    tail = elem;
                }

                if (ret > 0 || elem[0] == '\0')
                    break;
            }

            std::string header;
            std::string nbrs;

            std::istringstream elem_ss(elem);
            elem_ss >> header;
            elem_ss >> nbrs;

            std::istringstream nbrs_ss(nbrs);
            // int nb_num = 0;
            std::string s;

            v_adj_elem* add_one = new v_adj_elem(std::stoi(header));

            while(std::getline(nbrs_ss, s,','))
            {
                //
                while (s[0] == '\0' && s.length() > 0)
                    s = s.substr(1,s.length()-1);

                if (s[0] != '\0')
                {
                    add_one->_adjs.push_back(std::stoi(s));
                    // nb_num++;

                }
                // else
                // {
                    // std::printf("empty nbr value len: %d\n", (int)s.length());
                    // std::fflush;
                // }
            }

            //load one vert id and nbr list into table
            // if (nb_num > 0)
            if (add_one->_adjs.size() > 0)
                v_adj_table->push_back(add_one);
            
        }

        //check again
        if (ret <= 0)
            break;
    }

    hdfsCloseFile(fs_handle, readFile);

    delete[] buf;
    delete[] file;

}

void Input::free_readgraph_task(int thd_num)
{
    if (fs != NULL)
    {
        for(int i=0;i<thd_num;i++)
            hdfsDisconnect(fs[i]);

        delete[] fs;
    }

    if (v_adj != NULL)
    {
        for(int i=0;i<thd_num;i++)
        {
            for(int j=0;j<v_adj[i].size();j++)
            {
                if ((v_adj[i])[j] != NULL)
                    delete (v_adj[i])[j];
            }
        }

        delete[] v_adj;
    }

}

/**
 * @brief read in graph in parallel by using pthread
 */
void Input::readGraph()
{
    int thread_num = 24;
    fs = new hdfsFS[thread_num];
    for(int i=0;i<thread_num;i++)
        fs[i] = hdfsConnect("default", 0);

    v_adj = new std::vector<v_adj_elem*>[thread_num];

    NumericTablePtr filenames_array = get(filenames);
    NumericTablePtr filenames_offset = get(fileoffset);

    int file_num = filenames_offset->getNumberOfColumns() - 1;

    daal::internal::BlockMicroTable<int, readWrite, sse2> mtFileOffset(filenames_offset.get());
    // int* fileOffsetPtr = 0;
    mtFileOffset.getBlockOfRows(0, 1, &fileOffsetPtr);

    daal::internal::BlockMicroTable<int, readWrite, sse2> mtFileNames(filenames_array.get());
    // int* fileNamesPtr = 0;
    mtFileNames.getBlockOfRows(0, 1, &fileNamesPtr);

    std::printf("Finish create hdfsFS files\n");
    std::fflush;

    // create the thread pool
    threadpool thpool = thpool_init(thread_num);
    readG_task** task_queue = new readG_task*[file_num];

    for(int i=0;i<file_num;i++)
    {
        task_queue[i] = new readG_task(i, fs, fileOffsetPtr, fileNamesPtr, v_adj);

        std::printf("Finish create taskqueue %d\n", i);
        std::fflush;

		thpool_add_work(thpool, (void (*)(int,void*))thd_task_read, task_queue[i]);
    }

    thpool_wait(thpool);
	thpool_destroy(thpool);

    //sum up the total v_ids num on this mapper
    int total_v_num = 0;
    int total_nbr=0;
    for(int i = 0; i< thread_num;i++)
    {
        total_v_num += v_adj[i].size();
        for(int j=0;j<v_adj[i].size(); j++)
        {
            total_nbr += ((v_adj[i])[j])->_adjs.size();
        }

    }

    std::printf("Finish Reading all the vert num: %d, total nbrs: %d\n", total_v_num, total_nbr);
    std::fflush;

    free_readgraph_task(thread_num);
    // free task queue
    for(int i=0;i<file_num;i++)
        delete task_queue[i];

    delete[] task_queue;

}

void Input::readGraph_Single()
{
    // daal::services::Environment* env = daal::services::Environment::getInstance();
    // CpuType cpuid = (CpuType)(env->getCpuId());
    NumericTablePtr filenames_array = get(filenames);
    NumericTablePtr filenames_offset = get(fileoffset);

    int file_num = filenames_offset->getNumberOfColumns() - 1;
    // std::printf("File num: %zu\n", );
    // std::fflush(stdout);

    daal::internal::BlockMicroTable<int, readWrite, sse2> mtFileOffset(filenames_offset.get());
    int* fileOffsetPtr = 0;
    mtFileOffset.getBlockOfRows(0, 1, &fileOffsetPtr);

    daal::internal::BlockMicroTable<int, readWrite, sse2> mtFileNames(filenames_array.get());
    int* fileNamesPtr = 0;
    mtFileNames.getBlockOfRows(0, 1, &fileNamesPtr);

    // hdfsFS fs = NULL; 
    hdfsFS fs = hdfsConnect("default", 0);
    hdfsFile readFile = NULL;
    char* buf = new char[1];

    std::printf("total file num: %d\n", file_num);
    std::fflush;

    int read_file_count = 0;
    for(int i=0;i<file_num;i++)
    {

        int len = fileOffsetPtr[i+1] - fileOffsetPtr[i];
        char* file = new char[len+1];
        for(int j=0;j<len;j++)
            file[j] = (char)(fileNamesPtr[fileOffsetPtr[i] + j]);

        // terminate char buf
        file[len] = '\0';

        std::printf("FilePath: %s\n", file);
        std::fflush;

        if (hdfsExists(fs, file) < 0)
            continue;

        readFile = hdfsOpenFile(fs, file, O_RDONLY, 0, 0, 0);
        int file_size = hdfsAvailable(fs, readFile);
        std::printf("Filesize: %d\n", file_size);
        std::fflush;

        read_file_count++;
        std::printf("Readed %d files\n", read_file_count);
        std::fflush;

        // read char one by one
        tSize ret = 0;
        std::string vert;
        while(1)
        {
            ret = hdfsRead(fs, readFile, (void*)buf, 1); 
            if (ret <= 0)
                break;

            if (buf[0] == '\n')
            {
                std::string vert_id;
                std::istringstream iss(vert);
                iss >> vert_id;
                std::printf("vert id: %s\n", vert_id.c_str());
                std::fflush;

                std::string nbrs;
                iss >> nbrs;

                //retrieve all the nb values
                std::istringstream vert_nbrs(nbrs);
                int nb_num = 0;
                std::string s;
                while(std::getline(vert_nbrs,s,','))
                {
                    nb_num++;
                    if (nb_num < 10)
                    {
                        std::printf("vert nb: %s\n", s.c_str());
                        std::fflush;
                    }

                }

                std::printf("vert nb num: %d\n", nb_num);
                std::fflush;
                vert.clear();
            }
            else
                vert += buf[0];
        }

        hdfsCloseFile(fs, readFile);
        delete[] file;
    }

    hdfsDisconnect(fs);
    std::printf("red file num: %d\n", read_file_count);
    std::fflush;

}


} // namespace interface1
} // namespace subgraph
} // namespace algorithm
} // namespace daal

# Modifications by Harp-DAAL-2018 to DAAL-2018-beta-update1 version


## Get block of column values from a NumericTable by multithreading (data stored at java side) 

In file: include/data_management/data/numeric_table.h

```c++
virtual void getBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<double>** block) {} 
virtual void getBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<float>** block) {} 
virtual void getBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<int>** block) {} 
```

```c++
virtual void releaseBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, BlockDescriptor<double>** block) {} 
virtual void releaseBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, BlockDescriptor<float>** block) {} 
virtual void releaseBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, BlockDescriptor<int>** block) {} 
```

## BlockDescriptor uses sharedPtr instead of raw pointer

In file: include/data_management/data/numeric_table.h

```c++
private:
    services::SharedPtr<DataType> _ptr;
    services::SharedPtr<DataType> _aux_ptr;
    services::SharedPtr<DataType> _buffer; /*<! Pointer to the buffer */
    size_t    _capacity;                   /*<! Buffer size in bytes */
    services::SharedPtr<byte> *_pPtr;
    byte *_rawPtr;
    void freeBuffer()
    inline bool resizeBuffer( size_t nColumns, size_t nRows, size_t auxMemorySize = 0 )
```

## Comment out unused algorithm modules

In File: lang_interface/java/com/intel/daal/data_management/data/Factory.java 

```java
//if (objectId == SerializationTag.SERIALIZATION_SVM_MODEL_ID.getValue()) {
//     return new com.intel.daal.algorithms.svm.Model(context, cObject);
//}
```

## Add methods to some NumericTable for computation model (e.g., rotation)

In File: lang_interface/java/com/intel/daal/data_management/data/SOANumericTable.java

setArrayOnly will avoid repeatedly changing feature vals of each column in SOANumericTable. 

```java
public void setArrayOnly(double[] arr, long idx)
public void setArrayOnly(float[] arr, long idx)
public void setArrayOnly(long[] arr, long idx)
public void setArrayOnly(int[] arr, long idx)
```

In File: lang_interface/java/com/intel/daal/data_management/data/SOANumericTableImpl.java 

```java
public void setArrayOnly(double[] arr, long idx)
public void setArrayOnly(float[] arr, long idx)
public void setArrayOnly(long[] arr, long idx)
public void setArrayOnly(int[] arr, long idx)
```

## Add methods to JavaNumericTable to transfer data between native memory and JVM managed heap memory

In File: lang_service/java/com/intel/daal/data_management/data/java_numeric_table.h

```c++
void getBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, size_t vector_idx, size_t value_num,
ReadWriteMode rwflag, BlockDescriptor<double>** block) DAAL_C11_OVERRIDE
void getBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, size_t vector_idx, size_t value_num,
ReadWriteMode rwflag, BlockDescriptor<float>** block) DAAL_C11_OVERRIDE
void getBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, size_t vector_idx, size_t value_num,
ReadWriteMode rwflag, BlockDescriptor<int>** block) DAAL_C11_OVERRIDE
```

```c++
void releaseBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, BlockDescriptor<double>** block) DAAL_C11_OVERRIDE
void releaseBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, BlockDescriptor<float>** block) DAAL_C11_OVERRIDE
void releaseBlockOfColumnValuesBM(size_t feature_start, size_t feature_len, BlockDescriptor<int>** block) DAAL_C11_OVERRIDE
```

```c++
template<typename T>
void getTFeatureBM(size_t feature_start, size_t feature_len, size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T>** block,
                     const char *javaMethodName, const char *javaMethodSignature)
template<typename T>
    void releaseTFeatureBM(size_t feature_start, size_t feature_len, BlockDescriptor<T>** block, const char *javaMethodName)
```

1. No need to explicitly call the jvm->DetachCurrentThread() to detach the threads after transfer data

## Change some MACRO from DAAL-2017 to DAAL-2018

In File: algorithms/kernel/kernel.h

```c++
#define __DAAL_CALL_KERNEL(env, KernelClass, templateArguments, method, ...)            \
    {                                                                                   \
        return ((KernelClass<templateArguments, cpu> *)(_kernel))->method(__VA_ARGS__); \
    }

#define __DAAL_CALL_KERNEL_STATUS(env, KernelClass, templateArguments, method, ...)            \
        ((KernelClass<templateArguments, cpu> *)(_kernel))->method(__VA_ARGS__);
```
The original macro /__DAAL_CALL_KERNEL added a return operation after the execution, which will 
affect the codes following the macro call 
In the new version, instead, we use /__DAAL_CALL_KERNEL_STATUS macro 

## Modifications to return types for internal functions 

In DAAL-2018, services::Status is added to many internal functions (e.g., check, allocate) as a return type (these functions have a void type in last version of DAAL)
Also, the /_errors is removed from data structure like NumeircTable and Algorithm. The errors are returned by services::Status and print out by users 

## Modifications to random number (distribution) generator

In File: externals/service_rng.h

There are two classes: 1) A class BaseRNGs, 2) class RNGs
1. BaseRNGs specifies seed (by default: 777)and base random generator (by default: MT19937)
2. RNGs provides template for float or double type, it owns different types of distribution generators 
3. DAAL team shall add another constructor for BaseRNGs with a user defined seed (e.g., time(0));




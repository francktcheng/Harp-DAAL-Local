# Harp-DAAL-Local 

Harp-DAAL-Local is a customized Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) used by Harp-DAAL framework (https://github.com/DSC-SPIDAL/harp/tree/master/harp-daal-app). 
Harp-DAAL-Local inherits all of the functionalites of Intel DAAL while having additional data structures and algorithms designed for Harp-DAAL distributed framework. 
The current Harp-DAAL-Local is forked from Intel DAAL version 2018 beta update1

## License
Harp-DAAL-Local is licensed under Apache License 2.0.

## Online Documentation
Harp-DAAL-Local keeps the same source code structure and APIs with Intel DAAL, thus, users could always refer the Intel DAAL documentation 
on the [Intel(R) Data Analytics Acceleration Library 2017 Documentation](https://software.intel.com/en-us/intel-daal-support/documentation) web page.

### Validated Operating Systems
* Red Hat Enterprise Linux Server release 6.9 (Santiago)
* CentOS Linux release 7.2.1511

### Validated C/C++ Compilers 
* Intel(R) C++ Compiler 16.0.1 20151021 for Linux* OS
* Intel(R) C++ Compiler 17.0.2 20170213 for Linux* OS
* GNU Compiler 4.8.5 20150623

### Validated Java* Compilers:
* Java\* SE 8 from Sun Microsystems*

## Installation
Currently, users can only install Harp-DAAL-Local from sources

#### Required Software
* C/C++ compiler 
* Java\* JDK 

#### Installation Steps
1. Clone the sources from GitHub* as follows:
```bash
git clone --recursive https://github.com/francktcheng/Harp-DAAL-Local.git
```
2. Set an environment variable for one of the supported C/C++ compilers and Java compilers. For instance
```bash
source /opt/intel/compilers_and_libraries_2017/linux/bin/compilervars.sh intel64
export JAVA_HOME=/opt/jdk1.8.0_101
export PATH=$JAVA_HOME/bin:$PATH
```
3. Edit the makefile.lst file, only keeping the algorithms used by Harp-DAAL framework. The current version provides 7 algorithms as follows
```bash
implicit_als
kmeans
mf_sgd
pca
qr
svd
neural_networks
```
4. Build Harp-DAAL-Local via the command-line interface with the following commands:

 *  on Linux\* using Intel(R) C++ Compiler:

            make daal PLAT=lnx32e

 *  on Linux\* using GNU Compiler Collection\*:

            make daal PLAT=lnx32e COMPILER=gnu

Built libraries are located in the \__release_lnx/daal directory.


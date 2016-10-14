/* file: VPoint.java */
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

package com.intel.daal.algorithms.mf_sgd;

public class VPoint
{
	public int _wPos;
	public int _hPos;
	public float _val;

	public VPoint(int wPos, int hPos, float val)
	{
		_wPos = wPos;
		_hPos = hPos;
		_val = val; 
	}


}

import pandas as pd
import numpy as np
import NewInterleaveBits as lb
# from Partitioner import RangePartitioner
import csample
import workload
import os
import pickle
import random
import bisect
from scipy.stats import rankdata

# 蓄水池采样，输入：待采样的列、采样个数
# 返回：对应的列的采样数据，作为分区边界
def ReservoirSample(data, columns, sample_nums):
    length = len(columns)
    Samples = []
    for i in range(length):
        data_col = data.loc[:,columns[i]]
        nums = sample_nums if sample_nums < data.shape[0] else data.shape[0]
        samples = csample.reservoir(data_col, nums)
        samples.sort()
        Samples.append(samples)
    return Samples

# 从采样数据中获取分区边界
# INPUTS: samples 采样点、partition_nums 待分区数量、total_nums 数据总量、col_nums 列数
# OUTPUT：返回 partition_nums - 1 个边界值 
def GetRangeBoundsWithSample(Samples, partition_nums, total_nums, col_nums):
    sample_nums = len(Samples[0])
    weight = total_nums / sample_nums
    target = total_nums / partition_nums
    
    # 不能直接 [[] * col_nums] 这是浅拷贝，其中的每个列表共用一块内存，修改其中一个另外的会一起改变
    range_bounds = [[] for i in range(col_nums)]
    
    for i in range(col_nums):
        step = 0
        nums = 0
        samples = Samples[i]
        for candidate in samples:
            step += weight
            if step >= target:
                nums += 1
                if nums >= partition_nums:
                    break
                step = 0
                range_bounds[i].append(candidate)
    return range_bounds


# 排序法：排序后精确获取分区边界
# INPUTS: data 源数据、column 选作 Zorder 的列、partition_nums 分区数量
# OUTPUT: 返回 partition_nums + 1 个边界值
def SortDataAndGetRangeBound(data, columns, partition_nums):
    #order_data = rankdata(data[column], method='ordinal') - 1
    col_nums = len(columns)
    if partition_nums > data.shape[0]:
        for i in range(col_nums):
            range_bounds.append(data[columns[i]])
        return range_bounds
    range_bounds = [[] for i in range(col_nums)]
    for c in range(col_nums):
        order_data = pd.DataFrame()
        order_data = data[columns[c]].sort_values()
        partitions, RangeBounds = pd.qcut(order_data, q=partition_nums, retbins=True)
        #print(partitions.value_counts())
        RangeBounds = RangeBounds[1:len(RangeBounds)]
        range_bounds[c] = RangeBounds
    return range_bounds


# INPUTS: data 原数据、RangeBounds 分区边界、Columns 需要做重排的列、zorderfile 新的文件，在最后一列加入 zvalue
def GetPartitionIDtoZorder(data, RangeBounds, Columns, zorderfile):
    col_nums = len(Columns)
    total_nums = data.shape[0]
    partitionIDs = [[] for i in range(total_nums)]
    for i in range(total_nums):
        for c in range(col_nums):
            num = data.loc[i, Columns[c]]
            id = bisect.bisect_left(RangeBounds[c], num)
            partitionIDs[i].append(id)    
        zvalue = lb.interleavem(*partitionIDs[i])
        """ if i % 100 == 0:
            print(f'ID: {partitionIDs[i]}, zvalue:{zvalue}\n') """
        data.loc[i, "zvalue"] = zvalue
    data.to_csv(zorderfile, index=False, sep='|')

def GetColumns():
    is_zorder = False
    # Get the algorithm actions
    with open('/home/ning/my_spark/share/CoWorkAlg/ColSelect.txt','r') as f:
        lines = f.readlines()
        last_line = lines[-1]
    ColSelect = eval(last_line)
    # turn actions to the real columns
    ZorderCol = []
    for i in range(len(ColSelect)):
        if ColSelect[i] == 1:
            ZorderCol.append(workload.orgin_col[i])
    # Judge whether the columns has done zorder, if done, skip it
    with open('/home/ning/zorder/GetZorder/ZorderColRes.pkl','rb') as f:
        ZorderedCol = pickle.load(f)
    if ZorderCol in ZorderedCol:
        is_zorder = True
        pass
    else:
        ZorderedCol.append(ZorderCol)
        with open('/home/ning/zorder/GetZorder/ZorderColRes.pkl','wb') as f:
            pickle.dump(ZorderedCol,f)
    return is_zorder,ZorderCol
    
if __name__ == "__main__":
    dir = '/home/ning/my_spark/share/tpch-for-spark/tpch_data_1/lineitem/'
    file = "lineitem"
    seperate = '|'
    
    
    """ source_file = dir + file + "/" + file + ".csv"
    order_file = dir + file + "/" + file +  "_sorted_data.csv"
    zorder_file = dir + file + "/" + file + "_zorder_data.csv"
    zvalue_file = dir + file + "/" + file + "_z_value.csv"
    count_file = dir + file + "/" + file + "_count.csv" """

    source_file = dir + file + ".csv"
    dest_file = dir + file + "_zvalued_data.csv"
    zorder_file = dir + file + "_zorder.csv"
    count_file = dir + file + "_count.csv"

    fast_interleave_bits_enabled = False
    data = pd.read_csv(source_file, sep=seperate)
    
    
    # 每一维度数据量总数
    total_nums = data.shape[0]
    print(f'total_nums: {total_nums}\n')

    # 分区数量
    partition_nums = 1000
    # 蓄水池采样点数量，从中选取分区边界
    sample_nums = partition_nums * 60
    
    # is_zorder,columns = GetColumns()
    is_zorder = False
    columns = ['l_orderkey']
    if (is_zorder):
        pass
    else:
        col_nums = len(columns)
        #dataToZorder = data.loc[:, columns]

        # 蓄水池采样
        samples = ReservoirSample(data, columns, sample_nums)
    
        # 获取边界
        #range_bounds = GetRangeBoundsWithSample(samples, partition_nums, total_nums, col_nums)
        #print(f'sample: {range_bounds[0][0:100]}')
        range_bounds = SortDataAndGetRangeBound(data, columns, partition_nums)
        # print(f'order: {range_bounds[0][0:100]}')
        # 获取每个数据的分区号，并计算 zvalue
        zorder_file = dir + file + "_zorder.csv"
        # if os.path.exists(zorder_file):
        #     os.remove(zorder_file)
        GetPartitionIDtoZorder(data, range_bounds, columns, zorder_file)
        print('---------')
        # 统计生成的 zvalue 的个数
        """ data = pd.read_csv(zorder_file, sep=seperate)
        data.sort_values('zvalue', inplace=True)
        #data.loc[:,'zvalue'].to_csv(zvalue_file)
        vc = data['zvalue'].value_counts()
        vc.to_csv(count_file) """

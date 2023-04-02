import pandas as pd
import numpy as np
import NewInterleaveBits as lb
import csample
import random
import bisect
import pickle
import workload
import subprocess
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
# OUTPUT: 返回 partition_nums - 1 个边界值
def SortDataAndGetRangeBound(data, columns, partition_nums):
    #order_data = rankdata(data[column], method='ordinal') - 1
    col_nums = len(columns)
    total_nums = data.shape[0]
    # 分区数量 > 数据量，则另分区数量 = 数据量
    if partition_nums > total_nums:
        partition_nums = total_nums
    
    range_bounds = [[] for i in range(col_nums)]
    
    target = total_nums // partition_nums
    for c in range(col_nums):
        RangeBounds = []
        order_data = data[columns[c]].sort_values()
        values = order_data.nunique()
        # 基数是否小于 partition_nums
        if values <= partition_nums:
            RangeBounds = data[columns[c]].value_counts().index.tolist()
            RangeBounds.sort()
        else:
            for i in range(partition_nums - 1):
                RangeBounds.append(order_data.iloc[(i+1) * target - 1])
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
        # 单列的情况，直接用排序顺序作为 zvalue
        if col_nums == 1:
            zvalue = partitionIDs[i][0]
        else:
            zvalue = lb.interleavem(*partitionIDs[i])
        data.loc[i, "zvalue"] = zvalue
    data.to_csv(zorderfile, index=False, sep='|')

def GetColumns(i):
    is_zorder = False
    # Get the algorithm actions
    with open('/home/ning/my_spark/share/CoWorkAlg/AllColumnShow.txt','r') as f:
        lines = f.readlines()
        select_line = lines[i - 1]
    action_array = eval(select_line)
    with open('/home/ning/my_spark/share/CoWorkAlg/SkipFilesRes.pkl','rb') as f:
        skip_files_res = pickle.load(f)
    ZorderCol = []
    for i in range(len(action_array)):
            if action_array[i] == 1:
                ZorderCol.append(workload.orgin_col[i])
    Done = False
    action_array = str(action_array)
    if action_array in skip_files_res:
        Done = True
    return Done,ZorderCol

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
    zvalue_file = dir + file + "_zvalue.csv"
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
    for i in range(1,16):
        with open('/home/ning/my_spark/share/CoWorkAlg/count.pkl','wb') as f:
            pickle.dump(i,f)
        is_zorder,columns = GetColumns(i)
        if is_zorder == True:
            pass
        else:
            col_nums = len(columns)
            #dataToZorder = data.loc[:, columns]

            # 蓄水池采样
            samples = ReservoirSample(data, columns, sample_nums)
            
            # 获取边界
            range_bounds = GetRangeBoundsWithSample(samples, partition_nums, total_nums, col_nums)
            # print(f'sample: {range_bounds[0][0:100]}')
            range_bounds = SortDataAndGetRangeBound(data, columns, partition_nums)
            # print(f'order: {range_bounds[0][0:100]}')

            # 获取每个数据的分区号，并计算 zvalue
            zorder_file = dir + file + "_zorder.csv"
            print('GetPartitionIDtoZorder begin')
            GetPartitionIDtoZorder(data, range_bounds, columns, zorder_file)
            print('GetPartitionIDtoZorder end')

        subprocess.run(['docker', 'exec','-it', 'my_spark-spark-1', '/opt/bitnami/python/bin/python','/opt/share/CoWorkAlg/execu_sql.py'])

        # 统计生成的 zvalue 的个数
        """ data = pd.read_csv(zorder_file, sep=seperate)
        data.sort_values('zvalue', inplace=True)
        #data.loc[:,'zvalue'].to_csv(zvalue_file)
        vc = data['zvalue'].value_counts()
        vc.to_csv(count_file) """

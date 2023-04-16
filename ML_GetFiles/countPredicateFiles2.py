import pandas as pd
import numpy as np
import NewInterleaveBits as lb
import csample
import bisect
import math
from scipy.stats import rankdata
# from ..Cardinality_Estimation_pg.Gen_workload import *
import os
import random
from pandas.core.frame import DataFrame
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, Birch, FeatureAgglomeration, MeanShift, estimate_bandwidth
# 蓄水池采样，输入：待采样的列、采样个数
# 返回：对应的列的采样数据，作为分区边界


def ReservoirSample(data, columns, sample_nums):
    # print(type(data))
    length = len(columns)
    Samples = []
    for i in range(length):
        data_col = data.loc[:, columns[i]]
        samples = csample.reservoir(data_col, sample_nums)
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
        values = len(set(samples))
        # 基数是否小于 partition_nums
        if values <= partition_nums:
            RangeBounds = list(set(samples))
            RangeBounds.sort()
            print(f'card less than p_nums: ')
            range_bounds[i] = RangeBounds
            # RangeBounds = RangeBounds[1:len(RangeBounds)]
        else:
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
    # order_data = rankdata(data[column], method='ordinal') - 1
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


# INPUTS: data 原数据、RangeBounds 分区边界、Columns 需要做重排的列
def GetPartitionIDToCalDist(data, RangeBounds, Columns):
    col_nums = len(Columns)
    total_nums = data.shape[0]
    for index, row in data.iterrows():
        for c in range(col_nums):
            num = row[Columns[c]]
            id = bisect.bisect_left(RangeBounds[c], num)
            data.loc[index, Columns[c] + "_order"] = id

    return data

def GetColumn(fullname, columns):
    '''for i in range(len(columns)):
        strs = " ".join(columns[i])
        if fullname.find(strs) != -1:
            return i
    return 0'''
    pos = []
    for i in range(len(columns)):
        if fullname.find(columns[i]) != -1:
            pos.append(i)
    return pos


# ----------------------------------------------------
import numpy as np
import argparse as args
import pandas as pd
import random
import psycopg2
import datetime
from numpy import array
import ast
from datasets import load_dataset
import re
import math

def GeneratePredict():
    # lineitem init
    all_predict_cols = ['l_orderkey','l_partkey','l_suppkey','l_extendedprice','l_shipdate','l_commitdate','l_receiptdate']
    file_name = '/home/ning/postgres/tpch/dbgen/lineitem.tbl'
    index = [0,1,2,5,10,11,12]

    # nation init
    # all_predict_cols = ['n_nationkey','n_name','n_regionkey','n_comment']
    # file_name = '/home/ning/postgres/tpch/dbgen/nation.tbl'
    # index = [0,1,2,3]
    
    table = pd.read_csv(file_name,header=None,delimiter='|')
    predict = []
    for i in range(100):    
        nums_filter = random.randint(2,7)
        seed = random.randint(1,100)
        rng = np.random.RandomState(seed)
        random_row = rng.randint(0,table.shape[0])
        s = table.iloc[random_row]   
        vals = s.values
        vals = vals[index]
        idxs = rng.choice(len(all_predict_cols),replace=False,size = nums_filter)
        cols = np.take(all_predict_cols,idxs)
        ops = rng.choice(['<=','>='],size = nums_filter)
        vals = vals[idxs]
        one_predict = np.array([cols,ops,vals])
        predict.append(one_predict)
        i += 1
    # print(predict,type(predict))
    predict = str(predict)
    # print(predict)
    with open('Cardinality_Estimation_pg/gen_rand_predicts100sql.txt','w') as f:
        f.write(predict)

def GetPredictCombin():
    with open('Cardinality_Estimation_pg/gen_rand_predicts100sql.txt', 'r') as f:
        input_str = f.read()
        input_list = eval(input_str)

    TotalWhere = []
    # returnlist = []
    for array in input_list:
        # print(array)
        where = []
        eachsqlwhere = []
        for condition in array.T:
            column_name = condition[0]
            comparison_operator = condition[1]
            value = condition[2]
            where = np.array([column_name,comparison_operator,value])
            eachsqlwhere.append(where)
        TotalWhere.append(eachsqlwhere)
    returnlist = []
    for eachsqlwhere in TotalWhere:
        FinalPredict = ''
        for eachpredict in eachsqlwhere:
            if '-' in str(eachpredict):
                sqlwhere = eachpredict[0] + ' ' + eachpredict[1] + ' ' + "'" + eachpredict[2] + "'"
            else:
                sqlwhere = eachpredict[0] + ' ' + eachpredict[1] + ' ' + eachpredict[2]
            FinalPredict = FinalPredict +  sqlwhere + " AND "
        FinalPredict = FinalPredict.strip(" AND ")
        returnlist.append(FinalPredict)
    return returnlist

def GetErow(explain_result):
    pattern = r'rows=(\d+)'
    match = re.search(pattern,str(explain_result))
    if match:
        value = match.group(1)
    return value

def GetSQLBasedPredicts(PredictCombin):
    final_workload = []
    common_sql_part = '''select * from lineitem where '''
    for each_sql_predict in PredictCombin:
        final_sql = common_sql_part + str(each_sql_predict)
        final_workload.append(final_sql)
    return final_workload


def GetEachSQLErowsRatio(PredictCombin):
    conn = psycopg2.connect(database = "postgres",user = "ning", password = "",host = "127.0.0.1", port = "5432")
    cur = conn.cursor()
    conn.set_session(autocommit=True)

    sql = '''explain select * from lineitem'''
    cur.execute(sql)
    results=cur.fetchall()
    All_rows = GetErow(results)
    skip_files_ration = []
    read_rows = []
    
    CommonSql = '''explain select * from lineitem where '''
    for predictsql in PredictCombin:
        sql = CommonSql + str(predictsql)
        cur.execute(sql)
        sql_result = cur.fetchall()
        each_sql_erow = GetErow(sql_result)
        each_sql_erow = eval(each_sql_erow)
        read_rows.append(each_sql_erow)
        skip_files_ration.append(float(each_sql_erow)/float(All_rows))
    # print(skip_files_ration)
    conn.commit()
    conn.close()
    # print(read_rows)
    return read_rows

def GetWorkloadErowsRatio(workload,sql_nums):
    conn = psycopg2.connect(database = "postgres",user = "ning", password = "",host = "127.0.0.1", port = "5432")
    cur = conn.cursor()
    conn.set_session(autocommit=True)

    sql = '''explain select * from lineitem'''
    cur.execute(sql)
    results=cur.fetchall()
    all_row = GetErow(results)
    sum_erows= eval(all_row) * (sql_nums - len(workload))

    ReadRows = []
    
    for sql in workload:
        sql = sql[0]
        sql = 'explain ' + sql
        cur.execute(sql)
        sql_result = cur.fetchall()
        each_sql_erow = GetErow(sql_result)
        each_sql_erow = (str(each_sql_erow)).strip('\n')
        ReadRows.append(eval(each_sql_erow))
        # sum_erows += int(each_sql_erow)
    # print(skip_files_ration)
    conn.commit()
    conn.close()
    # skip_ratio = float(sum_erows)/(float(all_row) * sql_nums)
    # return skip_ratio

def GetSingleDimSelectRatio(PredictCombin):
    all_predict_cols = ['l_orderkey','l_partkey','l_suppkey','l_extendedprice','l_shipdate','l_commitdate','l_receiptdate']
    
    all_single_col_predict = []
    for col in all_predict_cols:
        single_col_predict = []
        pattern = col + '\s*[<>=]+\s*[\d\'-]+\s*'
        for query in PredictCombin:
            matches = re.findall(pattern,query)
            # print(matches)
            for match in matches:
                single_col_predict.append(match.strip())
                # 这里不能支持同时大于和小于某一个谓词
                # single_col_predict = single_col_predict + " AND "
            # single_col_predict.
        all_single_col_predict.append(single_col_predict)
    # print(all_single_col_predict) 
#
    # 后面集成函数
    conn = psycopg2.connect(database = "postgres",user = "ning", password = "",host = "127.0.0.1", port = "5432")
    cur = conn.cursor()
    conn.set_session(autocommit=True)

    sql = '''explain select * from lineitem'''
    cur.execute(sql)
    results=cur.fetchall()
    all_rows = GetErow(results)
    sql_nums = len(PredictCombin)
    CommonSql = '''explain select * from lineitem where '''
    # 记录每一维的的基数估计
    single_col_select_ratio = {}
    each_col_erow = 0
    i = 0
    # print(all_single_col_predict)
    for each_col in all_single_col_predict:
        each_col_erow = eval(all_rows) * (sql_nums - len(each_col))
        for each_col_sql in each_col:
            sql = CommonSql + str(each_col_sql)
            # print(sql)
            cur.execute(sql)
            sql_result = cur.fetchall()
            each_col_sql_erow = GetErow(sql_result)
            each_col_erow += int(each_col_sql_erow)
        single_col_select_ratio[all_predict_cols[i]] = (float(each_col_erow) / (float(all_rows) * sql_nums))
        i += 1
            # skip_files_ration.append(float(each_sql_erow)/float(all_rows))
    # print(skip_files_ration)
    # print(single_col_select_ratio)
    conn.commit()
    conn.close()

    # print(single_col_select_ratio)
    key = max(single_col_select_ratio.keys(),key=lambda x:single_col_select_ratio[x])
    # print(key)
    single_cols_max_erows_ratio = single_col_select_ratio[key]
    single_cols_min_erows_ratio = 1 - single_cols_max_erows_ratio
    
    return single_cols_min_erows_ratio

def GetColumnName(action_array):
    column_name = []
    action_array = eval(action_array)
    for i in range(len(action_array)):
        if action_array[i] == 1:
            column_name.append(all_predict_cols[i])
    return column_name

def GetSQLBasedSelectCols(PredictCombin,select_cols):
    # 假如是三列 l_suppkey,l_commitdate,l_receiptdate
    # select_cols = ['l_suppkey','l_commitdate','l_receiptdate']
    new_sql_predicts = []
    new_sqls_based_select_cols = []
    for query in PredictCombin:
        new_sql_predict = ''
        for col in select_cols:
            pattern = col + '\s*[<>=]+\s*[\d\'-]+\s*' 
            matches = re.findall(pattern,query)
            # print(matches)
            for match in matches:
                new_sql_predict = new_sql_predict + match.strip() + " AND "
        new_sql_predict = new_sql_predict.strip(" AND ")
        if (len(new_sql_predict)):
            new_sql_predicts.append(new_sql_predict)
            # new_sql = GetSQLBasedPredicts([new_sql_predict])
            # new_sqls_based_select_cols.append(new_sql)
    return new_sql_predicts
    # for i in new_sqls_based_select_cols:
    #     print(i)
    # print(len(new_sqls_based_select_cols))
    # return new_sqls_based_select_cols

def GetActionArray():
    with open('/home/ning/zorder/Actions_Rewards/Select_cols.txt','r') as f:
        lines = f.readlines()
        action_array = lines[-1]
    return action_array
def GetSelectCols():
    action_array = GetActionArray()
    # action_array = [0,0,0,1,0,1,0]
    # action_array = str(action_array)
    Select_Cols = GetColumnName(action_array)
    return Select_Cols

def WriteRewards(predicte_files):
    with open('/home/ning/zorder/ML_GetFiles/done_reward.txt','a') as f:
        final_reward = math.log(eval(str(predict_files)))
        final_reward = -final_reward
        f.write(str(final_reward))
        f.write('\n')

def WriteSignleMinSelectRatio(single_cols_min_erows_ratio):
    with open("/home/ning/zorder/Actions_Rewards/single_min_select_ratio.txt",'w') as f:
        f.writelines(single_cols_min_erows_ratio)


def GetWorkloadInput(PredictCombin):
    
    # print(type(PredictCombin))
    # print(len(PredictCombin))
    workload_input = {}

    for each_sql_predicts in PredictCombin:
        only_predeict = ''
        for col_name in all_predict_cols:
            if col_name in each_sql_predicts:
                only_predeict = only_predeict + '_' + col_name
        if only_predeict in workload_input.keys():
            workload_input[only_predeict] += 1
        else:
            workload_input.update({only_predeict:1})
    workload_input_array = list(workload_input.values())
    return workload_input_array
    # print(list(workload_input_array))
    # sum = 0
    # for i in workload_input_array:
    #     sum += i
    # print(sum)
def GetWorkloadInput2(PredictCombin):
    workload_input2 = {}
    for each_sql_predicts in PredictCombin:
        count = 0
        for col_name in all_predict_cols:
            if col_name in each_sql_predicts:
                count += 1
        if str(count) in workload_input2.keys():
            workload_input2[str(count)] += 1
        else:
            workload_input2.update({str(count):1})
    workload_input2_array = list(workload_input2.values())
    # print(workload_input2_array)
    return workload_input2_array

#  "(sorted_data['l_orderkey'] >= 3691075) & (sorted_data['l_partkey'] <= 79406)"
def GetMLPredict(PredictCombin,selected_cols):
    predict = []
    for each_sql_predict in PredictCombin:
        for col in selected_cols:
            partner = col
            substitute = "sorted_data['" + col + "']"
            each_sql_predict = re.sub(partner,substitute,each_sql_predict)
        partner = 'AND'
        substitute = ')&('
        each_sql_predict = re.sub(partner,substitute,each_sql_predict)
        each_sql_predict = '(' + each_sql_predict + ')'
        predict.append(each_sql_predict)
    predict = eval(str(predict))
    return predict
    # print(predict)
def GetMlColumn(PredictCombin,selected_cols):
    columns = []
    for each_sql_predict in PredictCombin:
        column = []
        for col in selected_cols:
            if col in each_sql_predict:
                column.append(col)
        columns.append(column)
    columns = eval(str(columns))
    return columns
# ---------------------------------------




if __name__ == "__main__":
    seperate = '|'
    all_predict_cols = ['l_orderkey','l_partkey','l_suppkey','l_extendedprice','l_shipdate','l_commitdate','l_receiptdate']

    fullname = "/home/ning/postgres/tpch/dbgen/lineitem.tbl"
    PredictCombin = GetPredictCombin()
    selected_cols = GetSelectCols()
    predicted_based_selected_cols = GetSQLBasedSelectCols(PredictCombin,selected_cols)
    predicates = GetMLPredict(predicted_based_selected_cols,selected_cols)
    # print('===============================predictes=================================')
    # print(predicates)
    # print(type(predicates))
    # print(predicates)
    columns = GetMlColumn(predicted_based_selected_cols,selected_cols)
    # print(columns)
    selectivities = GetEachSQLErowsRatio(predicted_based_selected_cols)
    # print(selectivities)
    original_data = pd.read_csv(fullname,sep='|')
    # print(original_data)
    for t in range(1):
        seed = random.randint(0, 2023) * random.randint(0, 100)
        # print(seed)
        data = original_data.sample(frac=0.01, random_state=seed)
        # print(t, '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        # 每一维度数据量总数
        total_nums = data.shape[0]
        # print(f'total_nums: {total_nums}\n')

        # 分区数量
        partition_nums = 1000
        # 蓄水池采样点数量，从中选取分区边界
        sample_nums = partition_nums * 60

        # 待排序的列
        col = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice', 'l_shipdate', 'l_commitdate', 'l_receiptdate']

        col_nums = len(col)

        # 蓄水池采样
        samples = ReservoirSample(data, col, sample_nums)
        # print(f'Getting samples: \n')
        # 获取边界
        # print(f'Getting range bounds')
        # 从采样点中获取边界
        range_bounds = GetRangeBoundsWithSample(
            samples, partition_nums, total_nums, col_nums)
        # print(f'range_bounds\n')

        # 获取每个数据的分区号，并计算平均距离
        # print(f'Sorting data... \n')
        # 获取分区边界
        sorted_data = GetPartitionIDToCalDist(data, range_bounds, col)
        # print("Sorting finished")
        #print(sorted_data.head())

        Distances = [284.3265491314493, 792.0978793239323, 1077.54020607677, 1274.759977263888, 1430.2722577343843, 1554.2718279301532, 1707.320718682262, 1852.312872150045]
        # Distances = [284.3265491314493, 792.0978793239323, 1077.54020607677, 1274.759977263888, 1430.2722577343843, 1554.2718279301532, 1707.320718682262]
        total_files_nums = 0
        for i in range(len(predicates)):
            
            #if(i == 0):
            #    continue
            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


            query = predicates[i]
            col = columns[i]

            # print(col)
            col_nums = len(col)
    

            #Distance = math.sqrt(col_nums) * 999 / 2
            Distance = Distances[col_nums - 1]
            # print(f'distance: {Distance}')
            #Distance = (1 + math.log(col_nums)) * 999
            
            
            
            predicated_data = sorted_data.loc[eval(query)]
            # print(predicated_data.shape)
            
            sample_points = predicated_data.shape[0] if predicated_data.shape[0] < 10000 else 10000
            predicated_data = predicated_data.sample(n=sample_points) 
            
            # print(f'After sampling... {predicated_data.shape}') 
            
            Orders = []
            for c in range(col_nums):
                Orders.append(col[c] + "_order")

            orders_data = predicated_data.loc[:, Orders]
            
            arr = orders_data.to_numpy()
            
            #bandwith = estimate_bandwidth(arr, quantile=1)
            #print(bandwith)
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=Distance).fit(arr)
            #clustering = MeanShift(bandwidth=Distance).fit(arr)
            #print(clustering)
            unique, counts = np.unique(clustering.labels_, return_counts=True)

            # print(f"file_id:, {unique}\n")
            # print(f"numbers in this file:, {counts}\n")

            total_predicated = selectivities[i]
            predict_files = total_predicated / predicated_data.shape[0] * len(unique) 
            # print(f'predicted files: {predict_files}')
            total_files_nums += predict_files
        WriteRewards(total_files_nums)
  
          
            


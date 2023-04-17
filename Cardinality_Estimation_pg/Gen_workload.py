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

    sorted_list = sorted(single_col_select_ratio.items(),key=lambda x:x[1])
    print(sorted_list)

    # print(single_col_select_ratio)
    key = min(single_col_select_ratio.keys(),key=lambda x:single_col_select_ratio[x])
    print(key)
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
            new_sql = GetSQLBasedPredicts([new_sql_predict])
            new_sqls_based_select_cols.append(new_sql)

    # for i in new_sqls_based_select_cols:
    #     print(i)
    # print(len(new_sqls_based_select_cols))
    return new_sqls_based_select_cols

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

def WriteRewards(workload_erows_ratio):
    with open('/home/ning/zorder/Actions_Rewards/workload_erows_ratio.txt','a') as f:
        workload_erows_ratio_to_reward = 1 - workload_erows_ratio
        f.write(str(workload_erows_ratio_to_reward))
        f.write('\n')

def WriteSignleMinSelectRatio(single_cols_min_erows_ratio):
    with open("/home/ning/zorder/Actions_Rewards/single_min_select_ratio.txt",'w') as f:
        f.writelines(single_cols_min_erows_ratio)


def GetWorkloadInput(PredictCombin):
    
    print(type(PredictCombin))
    print(len(PredictCombin))
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
def GetMLPredict(PredictCombin):
    predict = []
    for each_sql_predict in PredictCombin:
        for col in all_predict_cols:
            partner = col
            substitute = "sorted_data['" + col + "']"
            each_sql_predict = re.sub(partner,substitute,each_sql_predict)
        partner = 'AND'
        substitute = ')&('
        each_sql_predict = re.sub(partner,substitute,each_sql_predict)
        each_sql_predict = '(' + each_sql_predict + ')'
        predict.append(each_sql_predict)
    predict = eval(predict)
    return predict
    # print(predict)
def GetMlColumn(PredictCombin):
    columns = []
    for each_sql_predict in PredictCombin:
        column = []
        for col in all_predict_cols:
            if col in each_sql_predict:
                column.append(col)
        columns.append(column)
    columns = eval(columns)
    return columns
if __name__ == "__main__":

    all_predict_cols = ['l_orderkey','l_partkey','l_suppkey','l_extendedprice','l_shipdate','l_commitdate','l_receiptdate']
    # GeneratePredict()
    PredictCombin = GetPredictCombin()
    # print(PredictCombin)
    sql_nums = len(PredictCombin)
    # GetMLPredict(PredictCombin)
    # GetMlColumn(PredictCombin)
    # print(sql_nums)
    # GetWorkloadInput2(PredictCombin)
    # print(PredictCombin)
    # workload = GetSQLBasedPredicts(PredictCombin)
    # for i in workload:
    #     sql = '"' + i + '"' +','
    #     print(sql)

    single_cols_min_erows_ratio = GetSingleDimSelectRatio(PredictCombin)
    WriteSignleMinSelectRatio(str(single_cols_min_erows_ratio))
    # select_cols = GetSelectCols()
    # sql_based_select_cols = GetSQLBasedSelectCols(PredictCombin,select_cols)
    # workload_erows_ratio = GetWorkloadErowsRatio(sql_based_select_cols,sql_nums)
    # WriteRewards(workload_erows_ratio)
    # print(workload_erows_ratio)
    # workload = GetSQLBasedPredicts(PredictCombin)
    # workload = GetWorkloadInput(PredictCombin)
    # GetWorkloadErowsRatio(,sql_nums)
    # GetEachSQLErowsRatio(PredictCombin)



   
    
    
   

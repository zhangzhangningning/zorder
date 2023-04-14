import re

# cols = ['l_suppkey','l_receiptdate']


# PredictComin = ['l_suppkey <= 1445 AND l_orderkey >= 2651684', "l_receiptdate >= '1993-01-30' AND l_partkey >= 77275", 'l_extendedprice >= 23854.92 AND l_suppkey <= 2435', 'l_extendedprice >= 13740.44 AND l_partkey <= 142920', "l_commitdate >= '1995-02-06' AND l_suppkey >= 5456 AND l_partkey <= 195455", "l_shipdate <= '1998-08-03' AND l_partkey <= 102227 AND l_extendedprice <= 51627.24", "l_receiptdate >= '1993-01-30' AND l_partkey >= 77275 AND l_shipdate <= '1993-01-13' AND l_commitdate <= '1992-12-26'", "l_shipdate <= '1997-08-15' AND l_suppkey >= 7422 AND l_receiptdate <= '1997-08-23' AND l_orderkey <= 5969217", "l_commitdate <= '1992-04-04' AND l_extendedprice >= 24028.31", "l_receiptdate >= '1998-10-10' AND l_partkey <= 64144"]


# for query in PredictComin:
#             newsql = []
#             for col in cols:
#                 pattern = col + '\s*[<>=]+\s*[\d\'-]+\s*' 
#                 matches = re.findall(pattern,query)
#                 # print(matches)
#                 for match in matches:
#                     newsql.append(match.strip())
#             sql = 
# str1 = 'zhangning'
# str2 = 'ningzhang'
# print(str1 + str2)

import random
for i in range (10):
    action = random.randint(0,1)
    print(action)
orgin_col = ['l_orderkey','l_quantity','l_partkey','l_discount']
workload1 = [
    # 共计是101个文件 wokload编码为(2,2,1,1,2)
    # 设计到的列有 l_orderkey,l_quantity,l_shipdate,l_discount 共四列
    "SELECT * FROM lineitem WHERE l_orderkey < 300000",
    "SELECT * FROM lineitem WHERE l_orderkey > 1524568 AND l_orderkey < 5422415",
    "SELECT * FROM lineitem WHERE l_orderkey < 3564525 AND l_quantity > 19",
    "SELECT * FROM lineitem WHERE l_orderkey > 225214 AND l_orderkey < 5225416 AND l_quantity > 27",
    "SELECT * FROM lineitem WHERE l_shipdate BETWEEN '1994-01-01' AND '1995-01-03'",
    "SELECT * FROM lineitem WHERE l_shipdate BETWEEN '1994-01-01' AND '1996-01-03' AND l_partkey < 86201 AND l_orderkey < 2542544",
    "SELECT * FROM lineitem WHERE l_partkey < 125324 AND l_orderkey < 356204",
    "SELECT * FROM lineitem WHERE l_partkey < 456252 AND l_partkey > 125245 AND l_orderkey > 1522362 AND l_orderkey < 4452782 ",
]
workload1show = [2,2,1,1,2]


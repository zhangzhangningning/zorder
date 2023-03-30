orgin_col = ['l_orderkey','l_quantity','l_shipdate','l_discount']
workload1 = [
    # 共计是101个文件 wokload编码为(2,2,1,1,2)
    # 设计到的列有 l_orderkey,l_quantity,l_shipdate,l_discount 共四列
    "SELECT * FROM lineitem WHERE l_orderkey < 100001",
    "SELECT * FROM lineitem WHERE l_orderkey > 100001 AND l_orderkey < 6000000",
    "SELECT * FROM lineitem WHERE l_orderkey < 100001 AND l_quantity > 20",
    "SELECT * FROM lineitem WHERE l_orderkey > 10001 AND l_orderkey < 900000 AND l_quantity > 20",
    "SELECT * FROM lineitem WHERE l_shipdate BETWEEN '1994-01-01' AND '1994-01-03'",
    "SELECT * FROM lineitem WHERE l_shipdate BETWEEN '1994-01-01' AND '1994-01-03' AND l_discount < 0.05 AND l_orderkey < 600000",
    "SELECT * FROM lineitem WHERE l_discount < 0.05 AND l_orderkey < 600000",
    "SELECT * FROM lineitem WHERE l_discount < 0.05 AND l_orderkey > 600000 AND l_orderkey < 1000000 ",
]
workload1show = [2,2,1,1,2]


with open('/home/ning/zorder/ML_GetFiles/reward.txt','r') as f:
    orgin_lines = f.readlines()
length = len(orgin_lines)

case = []

for i in range(length):
    if orgin_lines[i] in case:
        pass
    else:
        case.append(orgin_lines[i])

import numpy

aa = numpy.array(case)
aa.sort()
# bb = aa.tolist()
print(aa)
# print(min(bb))
# print(case)
print(len(case))
with open('/home/ning/zorder/ML_GetFiles/reward.txt','r') as f:
    orgin_lines = f.readlines()
length = len(orgin_lines)

case = []

for i in range(length):
    if orgin_lines[i] in case:
        pass
    else:
        case.append(orgin_lines[i])
print(len(case))
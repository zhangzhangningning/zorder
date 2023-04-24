
with open('/home/ning/zorder/ML_GetFiles/reward.txt','r') as f:
    orgin_lines = f.readlines()
length = len(orgin_lines)

with open('/home/ning/zorder/ML_GetFiles/count_reward.txt','w') as f:
    f.write('')

with open('/home/ning/zorder/ML_GetFiles/count_reward.txt','a') as f:
    for i in range(length):
        reward = orgin_lines[i].split()
        reward = reward[0] + '\n'
        f.write(reward)
with open('/home/ning/zorder/ML_GetFiles/count_reward.txt','r') as f:
    orgin_lines = f.readlines()
length = len(orgin_lines)

with open('/home/ning/zorder/ML_GetFiles/finnal_reward.txt', 'w') as f:
    f.write("")
with open('/home/ning/zorder/ML_GetFiles/finnal_reward.txt', 'a') as f:
    for i in range (length):
        reward = orgin_lines[i].split()
        reward = reward[0]
        line = str(i) + " " + str(reward) +'\n'
        f.writelines(line)


import matplotlib.pyplot as plt

with open('/home/ning/zorder/ML_GetFiles/finnal_reward.txt', 'r') as f:
    lines = f.readlines()

x = []
y1 = []
y2 = []

# for line in lines[:25000]:
for line in lines:
    data = line.strip().split(' ')
    x.append(int(data[0]))
    y1.append(float(data[1]))
    # y2.append(float(data[2]))

plt.plot(x, y1, label='A2C')
# plt.plot(x, y2, label='A2C')
plt.xlabel('epsoide')
plt.ylabel('Reward')
plt.title('110-7395-3h')
plt.legend()
plt.show()
plt.savefig('/home/ning/zorder/ML_GetFiles/plot.png')
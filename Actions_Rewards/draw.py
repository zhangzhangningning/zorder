
with open('/home/ning/zorder/Actions_Rewards/rewards.txt','r') as f:
    orgin_lines = f.readlines()
length = len(orgin_lines)
# print(orgin_lines)

with open('/home/ning/zorder/Actions_Rewards/finnal_reward.txt', 'w') as f:
    f.write("")
with open('/home/ning/zorder/Actions_Rewards/finnal_reward.txt', 'a') as f:
    for i in range (length):
        if (eval(orgin_lines[i]) != 0):
            line = str(i) + " " + str(orgin_lines[i])
            f.write(line)


import matplotlib.pyplot as plt

with open('/home/ning/zorder/Actions_Rewards/finnal_reward.txt', 'r') as f:
    lines = f.readlines()

x = []
y1 = []
y2 = []
# print(lines)
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
plt.title('6-7-7129-1min')
plt.legend()
plt.show()
plt.savefig('/home/ning/zorder/Actions_Rewards/plot.png')
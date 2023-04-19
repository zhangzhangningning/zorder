import pickle
# with open('/home/ning/zorder/ML_GetFiles/done_reward.txt','w') as f:
#     f.write('')
# with open('/home/ning/zorder/ML_GetFiles/reward.txt','w') as f:
#     f.write('')
# with open('/home/ning/zorder/ML_GetFiles/selected_cols.txt','w') as f:
#     f.write('')


# with open('/home/ning/zorder/ML_GetFiles/done_epsiode.pkl','rb') as f:
   # # pickle.dump({},f)
#     result = pickle.load(f)

# del result[str('[0 1 1 0 1 1 0]')]

# with open('/home/ning/zorder/ML_GetFiles/done_epsiode.pkl','wb') as f:
#     pickle.dump({},f)
# #     pickle.dump(result,f)

with open('/home/ning/zorder/ML_GetFiles/done_epsiode1.pkl','rb') as f:
    # pickle.dump({},f)
    result = pickle.load(f)
# print(result)
print(len(result))

for i, j in result.items():
    print(i,j)

# print(type(result))
# # reward =result.values()


# case = []

# for i in range(reward):
#     if reward[i] in case:
#         pass
#     else:
#         case.append(reward[i])
# print(len(case))
# print(reward)
# print(len(result.keys()))
# m = max(result.keys(),key = (lambda x:result[x]))
# print(m)

# # with open('/home/ning/zorder/ML_GetFiles/done_epsiode.pkl','wb') as f:
#     pickle.dump({},f)

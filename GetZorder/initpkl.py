import pickle

with open('/home/ning/zorder/GetZorder/ZorderColRes.pkl','wb') as f:
    pickle.dump([],f)

with open('/home/ning/zorder/GetZorder/SkipFilesRes.pkl','wb') as f:
    pickle.dump({},f)

# with open('/opt/share/CoWorkAlg/SkipFilesRes.pkl','rb') as f:
#     load = pickle.load(f)

# print(load)
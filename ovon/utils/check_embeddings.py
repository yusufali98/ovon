import pickle


with open('/srv/kira-lab/share4/yali30/ovon_duplicate/ovon/ovon_stretch_cache_naoki_eval_new.pkl', 'rb') as f:
    data = pickle.load(f)
print(len(data.keys()))
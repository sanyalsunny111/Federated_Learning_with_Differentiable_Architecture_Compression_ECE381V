import json
net_config_fict=json.load(open('net_cfg.json','r'))
print(net_config_fict.keys())
print(net_config_fict['0'])
print(type(net_config_fict['0']['cfg']))


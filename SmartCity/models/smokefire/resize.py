import torch
import torch.nn as nn
from torch.nn import functional as F

res = torch.rand(8, 1080, 1920, 3)
#print(res.shape)

res = res.permute(0, 3, 1, 2)
#print(res.shape)  # 8,3,1080,1920

res = F.interpolate(res, size=[224, 224])
print('res:',res.shape)  # 8,3,224,224

res_view = res.view(res.shape[0], -1)
print('res_view:',res_view.shape)

mean = torch.mean(res_view, dim=1).reshape(res.shape[0],1,1,1)
max, _ = torch.max(torch.abs(res_view), 1)
max = max.reshape(res.shape[0],1,1,1)
print(mean.shape)  # 8,1
print(max.shape)  # 8,1

res_out = res - mean
res_out = res_out / max

print(res_out.shape)

# 【1】
import torch
a = torch.tensor([0.3215,0.6430,0.96])
print(a)
b = torch.tensor([-39.3587,39.0374])
print(b)
c = torch.tensor([-1.7800])
x, y ,z= torch.meshgrid(a, b,c)
print("x ",x)
print("y ",y)
print("z ",z)
 

 
 
 

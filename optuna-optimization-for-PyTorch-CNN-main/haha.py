import torch

# 创建一个形状为 (8, 512, 512) 的张量，仅包含 0, 1, 2, 3 四个数
mask = torch.randint(0, 4, (1, 4, 4))
print(mask)
# 创建一个与 mask 形状相同的全 2 张量
result = torch.full_like(mask, 2)

# 找到 mask 中值不为 0 和 1 的位置
condition = (mask != 0) & (mask != 1)

# 将这些位置的值设为 1
result[condition] = 1

print(result)
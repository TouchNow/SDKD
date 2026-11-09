import torch
import pandas as pd

# 1. 路径设置
# torch_path = "/mnt/home/chendingyao/code/OFAKD-main/output/train/20260325-030322-resnet50-224/checkpoint/model_best.pth.tar"
# d2_path = "/mnt/home/chendingyao/ReviewKD_Detection/Detection/pretrained/r50-r101.pth"
# output_path = "/mnt/home/chendingyao/ReviewKD_Detection/Detection/pretrained/r50-r101-hkcsd.pth"
# rename_map_path = "Mapping.csv"
torch_path = "/root/autodl-tmp/Detection/pretrained/model_best.pth.tar"
d2_path = "/root/autodl-tmp/Detection/pretrained/r50-r101.pth"
# 2. 加载模型权重
torch_weights = torch.load(torch_path, map_location="cpu")["state_dict"]

d2_weights = torch.load(d2_path, map_location="cpu")

# 3. 加载映射表
rename_map = pd.read_csv(rename_map_path)

# 4. 替换 Detectron2 权重
replaced_count = 0
missing_in_torch = []

for idx, row in rename_map.iterrows():
    torch_key = row["Torch Name"]
    d2_key = row["Detectron2 Name"]

    if torch_key in torch_weights and d2_key in d2_weights:
        d2_weights[d2_key] = torch_weights[torch_key]
        replaced_count += 1
    elif torch_key not in torch_weights:
        missing_in_torch.append(torch_key)

# 5. 保存替换后的 Detectron2 权重
torch.save(d2_weights, output_path)

# 6. 打印信息
print(f"✅ 替换完成，成功替换 {replaced_count} 个参数，保存为：{output_path}")
if missing_in_torch:
    print("⚠️ 以下键在 torch 权重中未找到：")
    for key in missing_in_torch:
        print("  ", key)

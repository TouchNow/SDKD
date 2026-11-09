import torch


def merge_my_r50_backbone_to_d2_kd(torch_path, d2_path, output_path):
    print("正在加载权重...")

    # 1. 加载你的 ResNet50 权重
    torch_weights = torch.load(torch_path, map_location="cpu")
    if "state_dict" in torch_weights:
        torch_weights = torch_weights["state_dict"]

    # 2. 加载 Detectron2 KD 官方权重
    d2_dict = torch.load(d2_path, map_location="cpu")
    d2_weights = d2_dict["model"] if "model" in d2_dict else d2_dict

    replace_count = 0

    # 3. 遍历并仅替换 Backbone 参数
    for k, v in torch_weights.items():
        # ⚠️ 直接跳过 num_batches_tracked 和 分类层(fc)
        if "num_batches_tracked" in k or k.startswith("fc."):
            continue

        d2_k = ""

        # --- A. 映射 Stem 结构 ---
        if k == "conv1.weight":
            d2_k = "backbone.bottom_up.stem.conv1.weight"
        elif k.startswith("bn1."):
            attr = k.split(".")[-1]
            d2_k = f"backbone.bottom_up.stem.conv1.norm.{attr}"

        # --- B. 映射 Stages (layer1-4 -> res2-5) ---
        elif k.startswith("layer"):
            parts = k.split(".")
            layer_idx = int(parts[0].replace("layer", ""))
            res_idx = layer_idx + 1
            block_idx = parts[1]

            prefix = f"backbone.bottom_up.res{res_idx}.{block_idx}"
            sub_part = parts[2]

            if sub_part.startswith("conv"):
                conv_idx = sub_part
                d2_k = f"{prefix}.{conv_idx}.weight"

            elif sub_part.startswith("bn"):
                conv_idx = sub_part.replace("bn", "conv")
                attr = parts[3]
                d2_k = f"{prefix}.{conv_idx}.norm.{attr}"

            elif sub_part == "downsample":
                if parts[3] == "0":
                    d2_k = f"{prefix}.shortcut.weight"
                elif parts[3] == "1":
                    attr = parts[4]
                    d2_k = f"{prefix}.shortcut.norm.{attr}"

        # --- C. 执行替换 ---
        if d2_k and d2_k in d2_weights:
            if d2_weights[d2_k].shape == v.shape:
                d2_weights[d2_k] = v
                replace_count += 1
            else:
                print(f"⚠️ 形状不匹配跳过: {d2_k}")

    print(f"\n✅ 成功将你的 {replace_count} 个 Backbone 参数替换到了 Detectron2 模型中！")

    # 官方的 FC 层（如 backbone.bottom_up.fc1000.weight）未被触碰，保留原样。
    # 你的模型在加载时，如果你移除了配置里的分类头，这些多余的权重会自动被忽略。

    # 4. 保存文件
    torch.save(d2_dict, output_path)
    print(f"🎉 最终文件已保存至: {output_path}")


# ================================
if __name__ == "__main__":
    # MY_TORCH_WEIGHTS = "/mnt/home/chendingyao/code/OFAKD-main/output/train/20260325-030322-resnet50-224/checkpoint/model_best.pth.tar"
    # OFFICIAL_D2_WEIGHTS = "/root/autodl-tmp/Detection/pretrained/r50-r101.pth"
    # OUTPUT_WEIGHTS = "/root/autodl-tmp/Detection/pretrained/custom_kd_r50_r101.pth"
    MY_TORCH_WEIGHTS = "/mnt/home/chendingyao/ReviewKD_Detection/Detection/pretrained/hkcsd_res50_7860.pth.tar"
    OFFICIAL_D2_WEIGHTS = "/mnt/home/chendingyao/ReviewKD_Detection/Detection/pretrained/r50-r101.pth"
    OUTPUT_WEIGHTS = "/mnt/home/chendingyao/ReviewKD_Detection/Detection/pretrained/hkcsd_kd_r50_r101.pth"

    merge_my_r50_backbone_to_d2_kd(MY_TORCH_WEIGHTS, OFFICIAL_D2_WEIGHTS, OUTPUT_WEIGHTS)

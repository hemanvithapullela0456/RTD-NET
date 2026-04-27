from rtdnet_v4 import RTDNetV4

m = RTDNetV4(num_classes=30, base_ch=32)

for name, mod in m.named_children():
    p = sum(x.numel() for x in mod.parameters())
    print(f'{name}: {p:,}')
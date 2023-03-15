from mmdet.models.necks import FPN, DWT_FPN
from mmdet.models.backbones.mobilenet_v2 import MobileNetV2
from torchviz import make_dot

if __name__ == '__main__':
    import torch.nn as nn
    import torch
    from torchsummary import summary

    x = [torch.randn(1, 256, 256, 256), torch.randn(1, 512, 128, 128), torch.randn(1, 1024, 64, 64),
         torch.randn(1, 2048, 32, 32)]
    mode = DWT_FPN(in_channels=[256, 512, 1024, 2048],
                   out_channels=256,
                   num_outs=5)
    print(mode)
    outputs = mode.forward(x)
    out = mode(x)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')


    # g = make_dot(out)
    # # g.view()
    # g.render(filename='netStructure/myNetModel', view=False, format='png')  # 保存 pdf 到指定路径不打开

    # model = MobileNetV2()
    # print(model)

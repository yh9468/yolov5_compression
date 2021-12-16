import torch
import numpy as np

def get_yolov5_mask(model, rate, do_zero, CSP_list, args):
    try:
        net = model.module.state_dict()
        parameters = model.module.named_parameters()
    except:
        net = model.state_dict()
        parameters = model.named_parameters()
    importance_all = None
    
    for name, item in parameters:
        layer_num = int(name.split('.')[1])
        importance = None
        if layer_num < 9:
            if 'm.' in name and 'cv2' in name and 'bn.weight' in name:
                if layer_num == 4 or layer_num == 6:
                    if int(name.split('.')[3]) == 1:
                        key1 = name.replace('m.1.cv2.bn.weight', 'cv1.bn.weight')
                        key1_c = key1.replace('bn.weight', 'conv.weight')
                        key2 = name.replace('m.1.cv2.bn.weight', 'm.0.cv2.bn.weight')
                        key2_c = key2.replace('bn.weight', 'conv.weight')
                        # key3 = name.replace('m.2.cv2.bn.weight', 'm.1.cv2.bn.weight')
                        # key3_c = key3.replace('bn.weight', 'conv.weight')
                        conv_ = name.replace('bn.weight', 'conv.weight')
                        importance = item.data.pow(2) * net[conv_].data.view(item.size(0), -1).pow(2).mean(dim=1)
                        importance += net[key1].data.pow(2) * net[key1_c].data.view(item.size(0), -1).pow(2).mean(dim=1)
                        importance += net[key2].data.pow(2) * net[key2_c].data.view(item.size(0), -1).pow(2).mean(dim=1)
                        # importance += net[key3].data.pow(2) * net[key3_c].data.view(item.size(0), -1).pow(2).mean(dim=1)
                        importance /= 3
                else:
                    key1 = name.replace('m.0.cv2.bn.weight', 'cv1.bn.weight')
                    key1_c = key1.replace('bn.weight', 'conv.weight')
                    conv_ = name.replace('bn.weight', 'conv.weight')
                    importance = item.data.pow(2) * net[conv_].data.view(item.size(0), -1).pow(2).mean(dim=1)
                    importance += net[key1].data.pow(2) * net[key1_c].data.view(item.size(0), -1).pow(2).mean(dim=1)
                    importance /= 2
            elif 'cv1.bn.weight' in name:
                if layer_num not in CSP_list:
                    conv_ = name.replace('bn.weight', 'conv.weight')
                    importance = item.data.pow(2) * net[conv_].data.view(item.size(0), -1).pow(2).mean(dim=1)
                elif 'm.' in name:
                    conv_ = name.replace('bn.weight', 'conv.weight')
                    importance = item.data.pow(2) * net[conv_].data.view(item.size(0), -1).pow(2).mean(dim=1)
            elif 'bn.weight' in name:
                if 'cv' not in name and layer_num in CSP_list:
                    cv2_conv = name.replace('bn.weight', 'cv2.weight')
                    cv3_conv = name.replace('bn.weight', 'cv3.weight')
                    imp1_ = item.data[:item.size(0)//2].pow(2) * net[cv3_conv].data.view(item.size(0)//2, -1).pow(2).mean(dim=1)
                    imp2_ = item.data[item.size(0)//2:].pow(2) * net[cv2_conv].data.view(item.size(0)//2, -1).pow(2).mean(dim=1)
                    imp1_[imp1_>=imp1_.topk(1)[0][0]] = 100
                    imp2_[imp2_>=imp2_.topk(1)[0][0]] = 100
                    importance = torch.cat([imp1_, imp2_])
                else:
                    conv_ = name.replace('bn.weight', 'conv.weight')
                    importance = item.data.pow(2) * net[conv_].data.view(item.size(0), -1).pow(2).mean(dim=1)
        elif 'bn.weight' in name:
            if 'cv' not in name and layer_num in CSP_list:
                cv2_conv = name.replace('bn.weight', 'cv2.weight')
                cv3_conv = name.replace('bn.weight', 'cv3.weight')
                imp1_ = item.data[:item.size(0)//2].pow(2) * net[cv3_conv].data.view(item.size(0)//2, -1).pow(2).mean(dim=1)
                imp2_ = item.data[item.size(0)//2:].pow(2) * net[cv2_conv].data.view(item.size(0)//2, -1).pow(2).mean(dim=1)
                imp1_[imp1_>=imp1_.topk(1)[0][0]] = 100
                imp2_[imp2_>=imp2_.topk(1)[0][0]] = 100
                importance = torch.cat([imp1_, imp2_])
            else: 
                conv_ = name.replace('bn.weight', 'conv.weight')
                importance = item.data.pow(2) * net[conv_].data.view(item.size(0), -1).pow(2).mean(dim=1)
                

        if importance is not None:
            importance[importance>=importance.topk(1)[0][0]] = 100
            if importance_all is None:
                importance_all = importance.cpu().numpy()
            else:
                importance_all = np.append(importance_all, importance.cpu().numpy())
    
    threshold = np.sort(importance_all)[int(len(importance_all) * rate)]
    filter_mask = np.greater(importance_all, threshold).astype(float)

    if not do_zero:
        filter_mask[filter_mask==0] = 1e-9

    return filter_mask


def yolov5_prune(model, filter_mask, CSP_list, args):
    idx = 0
    try:
        net = model.module.state_dict()
        parameters = model.module.named_parameters()
    except:
        net = model.state_dict()
        parameters = model.named_parameters()
    
    filter_mask = torch.Tensor(filter_mask).cuda()
    for name, item in parameters:
        layer_num = int(name.split('.')[1])
        if layer_num < 9:
            if 'm.' in name and 'cv2' in name and 'bn.mask' in name:
                if layer_num == 4 or layer_num == 6:
                    if int(name.split('.')[3]) == 1:
                        key1 = name.replace('m.1.cv2.bn.mask', 'cv1.bn.mask')
                        key2 = name.replace('m.1.cv2.bn.mask', 'm.0.cv2.bn.mask')
                        # key3 = name.replace('m.1.cv2.bn.mask', 'm.1.cv2.bn.mask')
                        
                        item.data[:] = filter_mask[idx:idx+item.size(0)][:]
                        net[key1].data[:] = filter_mask[idx:idx+item.size(0)][:]
                        net[key2].data[:] = filter_mask[idx:idx+item.size(0)][:]
                        # net[key3].data[:] = filter_mask[idx:idx+item.size(0)][:]
                        idx += item.size(0)
                else:
                    key = name.replace('m.0.cv2.bn.mask', 'cv1.bn.mask')
                    item.data[:] = filter_mask[idx:idx+item.size(0)][:]
                    net[key].data[:] = filter_mask[idx:idx+item.size(0)][:]
                    idx += item.size(0)
            elif 'cv1.bn.mask' in name:
                if layer_num not in CSP_list:
                    item.data[:] = filter_mask[idx:idx+item.size(0)][:]
                    idx += item.size(0)
                elif 'm.' in name:
                    item.data[:] = filter_mask[idx:idx+item.size(0)][:]
                    idx += item.size(0)
            elif 'bn.mask' in name:
                item.data[:] = filter_mask[idx:idx+item.size(0)][:]
                idx += item.size(0)
        elif 'bn.mask' in name:
            item.data[:] = filter_mask[idx:idx+item.size(0)][:]
            idx += item.size(0)

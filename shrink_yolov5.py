import torch
import torch.nn as nn
from copy import deepcopy

def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)


def adapt_model_from_dict(parent_module, model_dict):
    state_dict = {}
    for key in model_dict:
        state_dict[key] = model_dict[key].size()

    new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d):
            conv = nn.Conv2d
            
            s = state_dict[n + '.weight']
            in_channels = s[1]
            out_channels = s[0]
            g = 1
            if old_module.groups > 1:
                in_channels = out_channels
                g = in_channels
            new_conv = conv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=old_module.kernel_size,
                bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
                groups=g, stride=old_module.stride)
            set_layer(new_module, n, new_conv)
        if isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
                affine=old_module.affine, track_running_stats=True)
            set_layer(new_module, n, new_bn)
        if isinstance(old_module, nn.Linear):
            # FIXME extra checks to ensure this is actually the FC classifier layer and not a diff Linear layer?
            num_features = state_dict[n + '.weight'][1]
            new_fc = nn.Linear(
                in_features=num_features, out_features=old_module.out_features, bias=old_module.bias is not None)
            set_layer(new_module, n, new_fc)
            if hasattr(new_module, 'num_features'):
                new_module.num_features = num_features
    new_module.eval()
    parent_module.eval()

    return new_module


pruned_state = torch.load('checkpoint.pt', map_location='cpu')

# Concat Layers
'''
-1 : pre_layer
others : layer_num
13 = [-1, 6]
17 = [-1, 4]
20 = [-1, 14]
23 = [-1, 10]
24 = [17, 20, 23] # Detect part
'''
layer_dict = {13:6, 17:4, 20:14, 23:10}
concat_dict = dict()
net = pruned_state['model'].state_dict()

CSP_list = set()
for key in net:
    layer_num = int(key.split('.')[1])
    if 'm.' in key:
        CSP_list.add(layer_num)

CSP_list = list(CSP_list)

key_list = list(net.keys())

pre_filters = list(range(12))
for name in key_list:
    layer_num = int(name.split('.')[1])
    if 'm.' in name and 'cv2' in name and 'bn.mask' in name:
        if layer_num == 4 or layer_num == 6:
            if int(name.split('.')[3]) == 2:
                cv1_conv = net[name.replace('m.2.cv2.bn.mask', 'cv1.conv.weight')]
                cv1_bn_mask = net[name.replace('m.2.cv2.bn.mask', 'cv1.bn.mask')]
                cv1_bn_weight = net[name.replace('m.2.cv2.bn.mask', 'cv1.bn.weight')]
                cv1_bn_bias = net[name.replace('m.2.cv2.bn.mask', 'cv1.bn.bias')]
                cv1_bn_rmean = net[name.replace('m.2.cv2.bn.mask', 'cv1.bn.running_mean')]
                cv1_bn_rvar = net[name.replace('m.2.cv2.bn.mask', 'cv1.bn.running_var')]
                num_filters = int(cv1_bn_mask.sum())
                filter_list = (-cv1_bn_mask).argsort()[:num_filters].sort()[0]
                cv1_conv.data = cv1_conv.data[filter_list, :, :, :]
                cv1_conv.data = cv1_conv.data[:, pre_filters, :, :]
                cv1_bn_weight.data = cv1_bn_weight.data[filter_list]
                cv1_bn_bias.data = cv1_bn_bias.data[filter_list]
                cv1_bn_rmean.data = cv1_bn_rmean.data[filter_list]
                cv1_bn_rvar.data = cv1_bn_rvar.data[filter_list]
                temp_pre_filters = pre_filters.clone()
                pre_filters = filter_list.clone()
                
                cv2_conv = net[name.replace('m.2.cv2.bn.mask', 'cv2.conv.weight')]
                cv2_bn_mask = net[name.replace('m.2.cv2.bn.mask', 'cv2.bn.mask')]
                cv2_bn_weight = net[name.replace('m.2.cv2.bn.mask', 'cv2.bn.weight')]
                cv2_bn_bias = net[name.replace('m.2.cv2.bn.mask', 'cv2.bn.bias')]
                cv2_bn_rmean = net[name.replace('m.2.cv2.bn.mask', 'cv2.bn.running_mean')]
                cv2_bn_rvar = net[name.replace('m.2.cv2.bn.mask', 'cv2.bn.running_var')]
                num_filters = int(cv2_bn_mask.sum())
                channel_size = cv2_conv.size(0)
                filter_list = (-cv2_bn_mask).argsort()[:num_filters].sort()[0]
                cv2_conv.data = cv2_conv.data[filter_list, :, :, :]
                cv2_conv.data = cv2_conv.data[:, temp_pre_filters, :, :]
                cv2_bn_weight.data = cv2_bn_weight.data[filter_list]
                cv2_bn_bias.data = cv2_bn_bias.data[filter_list]
                cv2_bn_rmean.data = cv2_bn_rmean.data[filter_list]
                cv2_bn_rvar.data = cv2_bn_rvar.data[filter_list]
                concat_pre_filters = torch.cat((pre_filters.clone(), filter_list.clone()+channel_size), dim=0)


                m0_cv1_conv = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv1.conv.weight')]
                m0_cv1_bn_mask = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv1.bn.mask')]
                m0_cv1_bn_weight = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv1.bn.weight')]
                m0_cv1_bn_bias = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv1.bn.bias')]
                m0_cv1_bn_rmean = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv1.bn.running_mean')]
                m0_cv1_bn_rvar = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv1.bn.running_var')]
                num_filters = int(m0_cv1_bn_mask.sum())
                filter_list = (-m0_cv1_bn_mask).argsort()[:num_filters].sort()[0]
                m0_cv1_conv.data = m0_cv1_conv.data[filter_list, :, :, :]
                m0_cv1_conv.data = m0_cv1_conv.data[:, pre_filters, :, :]
                m0_cv1_bn_weight.data = m0_cv1_bn_weight.data[filter_list]
                m0_cv1_bn_bias.data = m0_cv1_bn_bias.data[filter_list]
                m0_cv1_bn_rmean.data = m0_cv1_bn_rmean.data[filter_list]
                m0_cv1_bn_rvar.data = m0_cv1_bn_rvar.data[filter_list]
                pre_filters = filter_list.clone()

                m0_cv2_conv = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv2.conv.weight')]
                m0_cv2_bn_mask = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv2.bn.mask')]
                m0_cv2_bn_weight = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv2.bn.weight')]
                m0_cv2_bn_bias = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv2.bn.bias')]
                m0_cv2_bn_rmean = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv2.bn.running_mean')]
                m0_cv2_bn_rvar = net[name.replace('m.2.cv2.bn.mask', 'm.0.cv2.bn.running_var')]
                num_filters = int(m0_cv2_bn_mask.sum())
                filter_list = (-m0_cv2_bn_mask).argsort()[:num_filters].sort()[0]
                m0_cv2_conv.data = m0_cv2_conv.data[filter_list, :, :, :]
                m0_cv2_conv.data = m0_cv2_conv.data[:, pre_filters, :, :]
                m0_cv2_bn_weight.data = m0_cv2_bn_weight.data[filter_list]
                m0_cv2_bn_bias.data = m0_cv2_bn_bias.data[filter_list]
                m0_cv2_bn_rmean.data = m0_cv2_bn_rmean.data[filter_list]
                m0_cv2_bn_rvar.data = m0_cv2_bn_rvar.data[filter_list]
                pre_filters = filter_list.clone()

                m1_cv1_conv = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv1.conv.weight')]
                m1_cv1_bn_mask = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv1.bn.mask')]
                m1_cv1_bn_weight = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv1.bn.weight')]
                m1_cv1_bn_bias = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv1.bn.bias')]
                m1_cv1_bn_rmean = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv1.bn.running_mean')]
                m1_cv1_bn_rvar = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv1.bn.running_var')]
                num_filters = int(m1_cv1_bn_mask.sum())
                filter_list = (-m1_cv1_bn_mask).argsort()[:num_filters].sort()[0]
                m1_cv1_conv.data = m1_cv1_conv.data[filter_list, :, :, :]
                m1_cv1_conv.data = m1_cv1_conv.data[:, pre_filters, :, :]
                m1_cv1_bn_weight.data = m1_cv1_bn_weight.data[filter_list]
                m1_cv1_bn_bias.data = m1_cv1_bn_bias.data[filter_list]
                m1_cv1_bn_rmean.data = m1_cv1_bn_rmean.data[filter_list]
                m1_cv1_bn_rvar.data = m1_cv1_bn_rvar.data[filter_list]
                pre_filters = filter_list.clone()

                m1_cv2_conv = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv2.conv.weight')]
                m1_cv2_bn_mask = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv2.bn.mask')]
                m1_cv2_bn_weight = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv2.bn.weight')]
                m1_cv2_bn_bias = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv2.bn.bias')]
                m1_cv2_bn_rmean = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv2.bn.running_mean')]
                m1_cv2_bn_rvar = net[name.replace('m.2.cv2.bn.mask', 'm.1.cv2.bn.running_var')]
                num_filters = int(m1_cv2_bn_mask.sum())
                filter_list = (-m1_cv2_bn_mask).argsort()[:num_filters].sort()[0]
                m1_cv2_conv.data = m1_cv2_conv.data[filter_list, :, :, :]
                m1_cv2_conv.data = m1_cv2_conv.data[:, pre_filters, :, :]
                m1_cv2_bn_weight.data = m1_cv2_bn_weight.data[filter_list]
                m1_cv2_bn_bias.data = m1_cv2_bn_bias.data[filter_list]
                m1_cv2_bn_rmean.data = m1_cv2_bn_rmean.data[filter_list]
                m1_cv2_bn_rvar.data = m1_cv2_bn_rvar.data[filter_list]
                pre_filters = filter_list.clone()

                m2_cv1_conv = net[name.replace('m.2.cv2.bn.mask', 'm.2.cv1.conv.weight')]
                m2_cv1_bn_mask = net[name.replace('m.2.cv2.bn.mask', 'm.2.cv1.bn.mask')]
                m2_cv1_bn_weight = net[name.replace('m.2.cv2.bn.mask', 'm.2.cv1.bn.weight')]
                m2_cv1_bn_bias = net[name.replace('m.2.cv2.bn.mask', 'm.2.cv1.bn.bias')]
                m2_cv1_bn_rmean = net[name.replace('m.2.cv2.bn.mask', 'm.2.cv1.bn.running_mean')]
                m2_cv1_bn_rvar = net[name.replace('m.2.cv2.bn.mask', 'm.2.cv1.bn.running_var')]
                num_filters = int(m2_cv1_bn_mask.sum())
                filter_list = (-m2_cv1_bn_mask).argsort()[:num_filters].sort()[0]
                m2_cv1_conv.data = m2_cv1_conv.data[filter_list, :, :, :]
                m2_cv1_conv.data = m2_cv1_conv.data[:, pre_filters, :, :]
                m2_cv1_bn_weight.data = m2_cv1_bn_weight.data[filter_list]
                m2_cv1_bn_bias.data = m2_cv1_bn_bias.data[filter_list]
                m2_cv1_bn_rmean.data = m2_cv1_bn_rmean.data[filter_list]
                m2_cv1_bn_rvar.data = m2_cv1_bn_rvar.data[filter_list]
                pre_filters = filter_list[:]

                m2_cv2_conv = net[name.replace('m.2.cv2.bn.mask', 'm.2.cv2.conv.weight')]
                m2_cv2_bn_mask = net[name]
                m2_cv2_bn_weight = net[name.replace('m.2.cv2.bn.mask', 'm.2.cv2.bn.weight')]
                m2_cv2_bn_bias = net[name.replace('m.2.cv2.bn.mask', 'm.2.cv2.bn.bias')]
                m2_cv2_bn_rmean = net[name.replace('m.2.cv2.bn.mask', 'm.2.cv2.bn.running_mean')]
                m2_cv2_bn_rvar = net[name.replace('m.2.cv2.bn.mask', 'm.2.cv2.bn.running_var')]
                num_filters = int(m2_cv2_bn_mask.sum())
                filter_list = (-m2_cv2_bn_mask).argsort()[:num_filters].sort()[0]
                m2_cv2_conv.data = m2_cv2_conv.data[filter_list, :, :, :]
                m2_cv2_conv.data = m2_cv2_conv.data[:, pre_filters, :, :]
                m2_cv2_bn_weight.data = m2_cv2_bn_weight.data[filter_list]
                m2_cv2_bn_bias.data = m2_cv2_bn_bias.data[filter_list]
                m2_cv2_bn_rmean.data = m2_cv2_bn_rmean.data[filter_list]
                m2_cv2_bn_rvar.data = m2_cv2_bn_rvar.data[filter_list]
                pre_filters = filter_list[:]

                
                cv3_conv = net[name.replace('m.2.cv2.bn.mask', 'cv3.weight')]
                cv3_bn_mask = net[name.replace('m.2.cv2.bn.mask', 'cv3.bn.mask')]
                cv3_bn_weight = net[name.replace('m.2.cv2.bn.mask', 'cv3.bn.weight')]
                cv3_bn_bias = net[name.replace('m.2.cv2.bn.mask', 'cv3.bn.bias')]
                cv3_bn_rmean = net[name.replace('m.2.cv2.bn.mask', 'cv3.bn.running_mean')]
                cv3_bn_rvar = net[name.replace('m.2.cv2.bn.mask', 'cv3.bn.running_var')]
                num_filters = int(cv3_bn_mask.sum())
                filter_list = (-cv3_bn_mask).argsort()[:num_filters].sort()[0]
                cv3_conv.data = cv3_conv.data[filter_list, :, :, :]
                cv3_conv.data = cv3_conv.data[:, concat_pre_filters, :, :]
                cv3_bn_weight.data = cv3_bn_weight.data[filter_list]
                cv3_bn_bias.data = cv3_bn_bias.data[filter_list]
                cv3_bn_rmean.data = cv3_bn_rmean.data[filter_list]
                cv3_bn_rvar.data = cv3_bn_rvar.data[filter_list]
                pre_filters = filter_list.clone()
                concat_dict[layer_num] = pre_filters
        else:
            cv1_conv = net[name.replace('m.0.cv2.bn.mask', 'cv1.conv.weight')]
            if layer_num > 9:
                if layer_num in layer_dict:
                    pre_filters = torch.cat([pre_filters, cv1_conv.size(1)//2 + concat_dict[layer_dict[layer_num]]])
            cv1_bn_mask = net[name.replace('m.0.cv2.bn.mask', 'cv1.bn.mask')]
            cv1_bn_weight = net[name.replace('m.0.cv2.bn.mask', 'cv1.bn.weight')]
            cv1_bn_bias = net[name.replace('m.0.cv2.bn.mask', 'cv1.bn.bias')]
            cv1_bn_rmean = net[name.replace('m.0.cv2.bn.mask', 'cv1.bn.running_mean')]
            cv1_bn_rvar = net[name.replace('m.0.cv2.bn.mask', 'cv1.bn.running_var')]
            num_filters = int(cv1_bn_mask.sum())
            filter_list = (-cv1_bn_mask).argsort()[:num_filters].sort()[0]
            cv1_conv.data = cv1_conv.data[filter_list, :, :, :]
            cv1_conv.data = cv1_conv.data[:, pre_filters, :, :]
            cv1_bn_weight.data = cv1_bn_weight.data[filter_list]
            cv1_bn_bias.data = cv1_bn_bias.data[filter_list]
            cv1_bn_rmean.data = cv1_bn_rmean.data[filter_list]
            cv1_bn_rvar.data = cv1_bn_rvar.data[filter_list]
            temp_pre_filters = pre_filters.clone()
            pre_filters = filter_list.clone()

            
            cv2_conv = net[name.replace('m.0.cv2.bn.mask', 'cv2.conv.weight')]
            cv2_bn_mask = net[name.replace('m.0.cv2.bn.mask', 'cv2.bn.mask')]
            cv2_bn_weight = net[name.replace('m.0.cv2.bn.mask', 'cv2.bn.weight')]
            cv2_bn_bias = net[name.replace('m.0.cv2.bn.mask', 'cv2.bn.bias')]
            cv2_bn_rmean = net[name.replace('m.0.cv2.bn.mask', 'cv2.bn.running_mean')]
            cv2_bn_rvar = net[name.replace('m.0.cv2.bn.mask', 'cv2.bn.running_var')]
            num_filters = int(cv2_bn_mask.sum())
            channel_size = cv2_conv.size(0)
            filter_list = (-cv2_bn_mask).argsort()[:num_filters].sort()[0]
            cv2_conv.data = cv2_conv.data[filter_list, :, :, :]
            cv2_conv.data = cv2_conv.data[:, temp_pre_filters, :, :]
            cv2_bn_weight.data = cv2_bn_weight.data[filter_list]
            cv2_bn_bias.data = cv2_bn_bias.data[filter_list]
            cv2_bn_rmean.data = cv2_bn_rmean.data[filter_list]
            cv2_bn_rvar.data = cv2_bn_rvar.data[filter_list]
            concat_pre_filters = torch.cat((pre_filters.clone(), filter_list.clone()+channel_size), dim=0)

            m0_cv1_conv = net[name.replace('m.0.cv2.bn.mask', 'm.0.cv1.conv.weight')]
            m0_cv1_bn_mask = net[name.replace('m.0.cv2.bn.mask', 'm.0.cv1.bn.mask')]
            m0_cv1_bn_weight = net[name.replace('m.0.cv2.bn.mask', 'm.0.cv1.bn.weight')]
            m0_cv1_bn_bias = net[name.replace('m.0.cv2.bn.mask', 'm.0.cv1.bn.bias')]
            m0_cv1_bn_rmean = net[name.replace('m.0.cv2.bn.mask', 'm.0.cv1.bn.running_mean')]
            m0_cv1_bn_rvar = net[name.replace('m.0.cv2.bn.mask', 'm.0.cv1.bn.running_var')]
            num_filters = int(m0_cv1_bn_mask.sum())
            filter_list = (-m0_cv1_bn_mask).argsort()[:num_filters].sort()[0]
            m0_cv1_conv.data = m0_cv1_conv.data[filter_list, :, :, :]
            m0_cv1_conv.data = m0_cv1_conv.data[:, pre_filters, :, :]
            m0_cv1_bn_weight.data = m0_cv1_bn_weight.data[filter_list]
            m0_cv1_bn_bias.data = m0_cv1_bn_bias.data[filter_list]
            m0_cv1_bn_rmean.data = m0_cv1_bn_rmean.data[filter_list]
            m0_cv1_bn_rvar.data = m0_cv1_bn_rvar.data[filter_list]
            pre_filters = filter_list.clone()

            m0_cv2_conv = net[name.replace('m.0.cv2.bn.mask', 'm.0.cv2.conv.weight')]
            m0_cv2_bn_mask = net[name]
            m0_cv2_bn_weight = net[name.replace('m.0.cv2.bn.mask', 'm.0.cv2.bn.weight')]
            m0_cv2_bn_bias = net[name.replace('m.0.cv2.bn.mask', 'm.0.cv2.bn.bias')]
            m0_cv2_bn_rmean = net[name.replace('m.0.cv2.bn.mask', 'm.0.cv2.bn.running_mean')]
            m0_cv2_bn_rvar = net[name.replace('m.0.cv2.bn.mask', 'm.0.cv2.bn.running_var')]
            num_filters = int(m0_cv2_bn_mask.sum())
            filter_list = (-m0_cv2_bn_mask).argsort()[:num_filters].sort()[0]
            m0_cv2_conv.data = m0_cv2_conv.data[filter_list, :, :, :]
            m0_cv2_conv.data = m0_cv2_conv.data[:, pre_filters, :, :]
            m0_cv2_bn_weight.data = m0_cv2_bn_weight.data[filter_list]
            m0_cv2_bn_bias.data = m0_cv2_bn_bias.data[filter_list]
            m0_cv2_bn_rmean.data = m0_cv2_bn_rmean.data[filter_list]
            m0_cv2_bn_rvar.data = m0_cv2_bn_rvar.data[filter_list]
            pre_filters = filter_list.clone()
            
            cv3_conv = net[name.replace('m.0.cv2.bn.mask', 'cv3.weight')]
            cv3_bn_mask = net[name.replace('m.0.cv2.bn.mask', 'cv3.bn.mask')]
            cv3_bn_weight = net[name.replace('m.0.cv2.bn.mask', 'cv3.bn.weight')]
            cv3_bn_bias = net[name.replace('m.0.cv2.bn.mask', 'cv3.bn.bias')]
            cv3_bn_rmean = net[name.replace('m.0.cv2.bn.mask', 'cv3.bn.running_mean')]
            cv3_bn_rvar = net[name.replace('m.0.cv2.bn.mask', 'cv3.bn.running_var')]
            num_filters = int(cv3_bn_mask.sum())
            filter_list = (-cv3_bn_mask).argsort()[:num_filters].sort()[0]
            cv3_conv.data = cv3_conv.data[filter_list, :, :, :]
            cv3_conv.data = cv3_conv.data[:, concat_pre_filters, :, :]
            cv3_bn_weight.data = cv3_bn_weight.data[filter_list]
            cv3_bn_bias.data = cv3_bn_bias.data[filter_list]
            cv3_bn_rmean.data = cv3_bn_rmean.data[filter_list]
            cv3_bn_rvar.data = cv3_bn_rvar.data[filter_list]
            pre_filters = filter_list.clone()
            concat_dict[layer_num] = pre_filters

    elif 'bn.mask' in name and layer_num not in CSP_list:
        conv_weight = net[name.replace('bn.mask', 'conv.weight')]
        if layer_num == 8 and 'cv2' in name:
            pre_filters = torch.cat([pre_filters+(i*conv_weight.size(1)//4) for i in range(4)])
        bn_mask = net[name]
        bn_weight = net[name.replace('bn.mask', 'bn.weight')]
        bn_bias = net[name.replace('bn.mask', 'bn.bias')]
        bn_rmean = net[name.replace('bn.mask', 'bn.running_mean')]
        bn_rvar = net[name.replace('bn.mask', 'bn.running_var')]
        num_filters = int(bn_mask.sum())
        filter_list = (-bn_mask).argsort()[:num_filters].sort()[0]
        conv_weight.data = conv_weight.data[filter_list, :, :, :]
        conv_weight.data = conv_weight.data[:, pre_filters, :, :]
        bn_weight.data = bn_weight.data[filter_list]
        bn_bias.data = bn_bias.data[filter_list]
        bn_rmean.data = bn_rmean.data[filter_list]
        bn_rvar.data = bn_rvar.data[filter_list]
        pre_filters = filter_list.clone()
        concat_dict[layer_num] = pre_filters

    elif layer_num==24:
        if 'anchor' in name:
            continue
        subnum = int(name.split('.')[3])
        if len(net[name].size()) == 4:
            if subnum == 0:
                pre_filters = concat_dict[17]
            elif subnum == 1:
                pre_filters = concat_dict[20]
            else:
                pre_filters = concat_dict[23]
            detect_conv = net[name]
            detect_conv.data = detect_conv.data[:, pre_filters, :, :]

for key in key_list:
    if 'mask' in key:
        del net[key]

shrink_state = torch.load('checkpoint.pt', map_location='cpu')

new_model = adapt_model_from_dict(shrink_state['model'], net)
new_model.load_state_dict(net)
shrink_state['model'] = new_model

print()
print("Shrinked model dict...")
new_net = shrink_state['model'].state_dict()

for key in new_net:
    print(key, new_net[key].size())
print("Total Params :",sum([item.numel() for item in shrink_state['model'].parameters()]))

torch.save(shrink_state, 'shrinked_ckpt.pt')
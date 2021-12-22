"""YOLOv5-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from .common import *
from .experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from utils.general import bbox_iou
from models.distill_utils import NonLocalBlockND, dist2

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

logger = logging.getLogger(__name__)


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, teacher=False, hyp=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.teacher = teacher
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # make distillation format
        self.head = self.model[-1]
        self.backbone = self.model[:-1]
        self.hyp = hyp

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward_once(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())


        # Init weights, biases
        initialize_weights(self)

        # make distillation modules (have to change )
        """
        self.channel_wise_adaptation = nn.ModuleList([
            nn.Linear(320, 320),
            nn.Linear(640, 640),
            nn.Linear(1280, 1280),
        ])

        self.spatial_wise_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        ])
        """
        self.adaptation_layers = nn.ModuleList([
            nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
        ])
        """
        self.student_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=320, inter_channels=160),
                NonLocalBlockND(in_channels=640, inter_channels=320),
                NonLocalBlockND(in_channels=1280),
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=320, inter_channels=160),
                NonLocalBlockND(in_channels=640, inter_channels=320),
                NonLocalBlockND(in_channels=1280),
            ]
        )
        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
        ])
        """
        self.info()
        logger.info('')
        

    def extract_feat(self, x, profile=False):
        dt, y = [], []
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
            
            x = m(x) 
            y.append(x if m.i in self.save else None)  # save output
        if self.head.f != -1:  # if not from previous layer
            x = y[self.head.f] if isinstance(self.head.f, int) else [x if j == -1 else y[j] for j in self.head.f]  # from earlier layers
        return x, y, dt

    def forward(self, x, augment=False, profile=False, teacher_info=None, targets=None):
        if self.teacher:
            return self.extract_feat(x)[0]
        if augment:
            return self.forward_augment(x, teacher_info)  # augmented inference, None
        else:
            return self.forward_once(x, profile, t_feats=teacher_info, targets=targets)  # single-scale inference, train
        
    def forward_augment(self, x, teacher_info=None):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi, t_feats=teacher_info)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False, feature_vis=False, t_feats=None, targets=None):
        x, y, dt = self.extract_feat(x)
        s_feats = []
        for x_ in x:
            s_feats.append(x_)

        # temperature factor & spatial/channel wise attention weight factor
        t = 0.1
        s_ratio = 1.0

        #   for channel attention temperature
        c_t = 0.1
        c_s_ratio = 1.0

        # change the inference of yolo for using distillation
        if t_feats is not None:
            device = t_feats[0].device
            kd_feat_loss = torch.zeros(1, device=device)
            #kd_channel_loss = torch.zeros(1, device=device)
            #kd_spatial_loss = torch.zeros(1, device=device)
            #kd_nonlocal_loss = torch.zeros(1, device=device)
            """
            for _i in range(len(t_feats)):
                # spatial teacher mask
                t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [1], keepdim=True)
                size = t_attention_mask.size()  # B x 1 x H x W

                # x[0].size(0) = batch size
                t_attention_mask = t_attention_mask.view(x[0].size(0), -1)

                # apply HW part softmax and multiply H, W 
                t_attention_mask = torch.softmax(t_attention_mask / t, dim=1) * size[-1] * size[-2]
                t_attention_mask = t_attention_mask.view(size)

                # spatial student mask
                s_attention_mask = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
                size = s_attention_mask.size()
                s_attention_mask = s_attention_mask.view(x[0].size(0), -1)
                s_attention_mask = torch.softmax(s_attention_mask / t, dim=1) * size[-1] * size[-2]
                s_attention_mask = s_attention_mask.view(size)
                
                # channel teacher mask
                c_t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x 1
                c_size = c_t_attention_mask.size()
                c_t_attention_mask = c_t_attention_mask.view(x[0].size(0), -1)  # 2 x 256
                c_t_attention_mask = torch.softmax(c_t_attention_mask / c_t, dim=1) * 256
                c_t_attention_mask = c_t_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                # channel student mask
                c_s_attention_mask = torch.mean(torch.abs(x[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x 1
                c_size = c_s_attention_mask.size()
                c_s_attention_mask = c_s_attention_mask.view(x[0].size(0), -1)  # 2 x 256
                c_s_attention_mask = torch.softmax(c_s_attention_mask / c_t, dim=1) * 256
                c_s_attention_mask = c_s_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                #sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
                sum_attention_mask = t_attention_mask
                sum_attention_mask = sum_attention_mask.detach()

                # making final feature mask using in feature distillation
                c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
                c_sum_attention_mask = c_sum_attention_mask.detach()

                # feature loss by using teacher feature and student feature
                kd_feat_loss += dist2(t_feats[_i], self.adaptation_layers[_i](x[_i]), attention_mask=sum_attention_mask,
                                      channel_attention_mask=c_sum_attention_mask) * 7e-5 * 6

                # original torch L2 loss & using this for channel kd loss
                kd_channel_loss += torch.dist(torch.mean(t_feats[_i], [2, 3]),
                                              self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3 * 6
                
                # spatial kd loss
                t_spatial_pool = torch.mean(t_feats[_i], [1]).view(t_feats[_i].size(0), 1, t_feats[_i].size(2),
                                                                   t_feats[_i].size(3))
                s_spatial_pool = torch.mean(x[_i], [1]).view(x[_i].size(0), 1, x[_i].size(2),
                                                             x[_i].size(3))
                kd_spatial_loss += torch.dist(t_spatial_pool, self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3 * 6

            for _i in range(len(t_feats)):
                s_relation = self.student_non_local[_i](x[_i])
                t_relation = self.teacher_non_local[_i](t_feats[_i])
                kd_nonlocal_loss += torch.dist(self.non_local_adaptation[_i](s_relation), t_relation, p=2)
            """
        if self.head.f != -1:  # if not from previous layer
            x = y[self.head.f] if isinstance(self.head.f, int) else [x if j == -1 else y[j] for j in self.head.f]  # from earlier layers
        if profile:
            o = thop.profile(self.head, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
            t = time_synchronized()
            for _ in range(10):
                _ = self.head(x)
            dt.append((time_synchronized() - t) * 100)
            if self.head == self.model[0]:
                logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
            logger.info(f'{dt[-1]:10.2f} {o:10.2f} {self.head.np:10.0f}  {self.head.type}')

        preds = self.head(x)
        if targets==None:
            return preds
        masks = self.make_mask(preds, targets)

        for idx in range(len(t_feats)):
            diff = (t_feats[idx] - self.adaptation_layers[idx](s_feats[idx])) ** 2
            #   print(diff.size())      batchsize x 1 x W x H,
            #   print(attention_mask.size()) batchsize x 1 x W x H
            diff = diff * masks[idx]
            diff = torch.sum(diff) ** 0.5
            kd_feat_loss += diff * 7e-4 * 6

        y.append(preds if self.head.i in self.save else None)  # save output
        """
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
            
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            
            if feature_vis and m.type == 'models.common.SPP':
                feature_visualization(x, m.type, m.i)
        """
        if profile:
            logger.info('%.1fms total' % sum(dt))

        if t_feats is not None:
            #kd_nonlocal_loss = kd_nonlocal_loss * 7e-5 * 6
            kd_loss = kd_feat_loss #+ kd_channel_loss + kd_spatial_loss + kd_nonlocal_loss
            #loss_item = torch.cat((kd_feat_loss, kd_channel_loss, kd_spatial_loss , kd_nonlocal_loss, kd_loss)).detach()
            return preds, kd_loss #, loss_item, kd_nonlocal_loss
        else:
            return preds


    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.head.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets


        for i in range(self.head.nl):
            anchors = self.head.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
    
    
    ## CVPR PART ##
    def make_mask(self, pred, targets):
        device = targets.device
        _, tbox, indices, anchors = self.build_targets(pred, targets)  # targets
        masks = []
        # Losses
        for i, pi in enumerate(pred):  # layer index, layer predictions
            mask = torch.zeros(pi.shape[:-1], device=device)
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=False)  # iou(prediction, target)
                max_iou = iou.max()
                iou = (iou > max_iou * 0.5)

                for idx, act in enumerate(iou):
                    if act:
                        mask[b[idx], a[idx], gj[idx], gi[idx]] = 1
            mask = torch.max(mask, dim=1, keepdim=True)[0]
            masks.append(mask)
        return masks

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add AutoShape module
        logger.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, MaskConv, MaskConvv2, GhostConv, Bottleneck, MaskBottleneck, MaskBottleneckv2, GhostBottleneck, 
                SPP, MaskSPP, MaskSPPv2, DWConv, MixConv2d, Focus, MaskFocus, MaskFocusv2, CrossConv, 
                BottleneckCSP, MaskBottleneckCSP, MaskBottleneckCSPv2, C3, MaskC3, MaskC3v2, C3TR]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, MaskBottleneckCSP, MaskC3, MaskBottleneckCSPv2, MaskC3v2]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard

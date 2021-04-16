import torch
import torch.nn as nn

class upsample(nn.Module):
    def __init__(self, scale_factor):
        super(upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)

class merge(nn.Module):
    def forward(self, x, y):
        return x + y

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, stride=1):
        super(residual, self).__init__()
        p = (k - 1) // 2

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(p, p), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(p, p), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)


def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim, stride=2)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)


def _make_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

def _make_layer_revr(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, inp_dim) for _ in range(modules - 1)]
    layers += [residual(inp_dim, out_dim)]
    return nn.Sequential(*layers)

def _make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def _make_unpool_layer(dim):
    return upsample(scale_factor=2)

def _make_merge_layer(dim):
    return merge()


def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

class saccade_module(nn.Module):
    def __init__(
        self, n, dims, modules, make_up_layer=_make_layer,
        make_pool_layer=_make_pool_layer, make_hg_layer=_make_layer,
        make_low_layer=_make_layer, make_hg_layer_revr=_make_layer_revr,
        make_unpool_layer=_make_unpool_layer, make_merge_layer=_make_merge_layer
    ):
        super(saccade_module, self).__init__()

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.n    = n
        self.up1  = make_up_layer(curr_dim, curr_dim, curr_mod)
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)
        self.low2 = saccade_module(
            n - 1, dims[1:], modules[1:],
            make_up_layer=make_up_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
        self.up2  = make_unpool_layer(curr_dim)
        self.merg = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        # if self.n > 1:
        #     low2, mergs = self.low2(low1)
        # else:
        #     low2, mergs = self.low2(low1), []
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        merg = self.merg(up1, up2)
        #mergs.append(merg)
        #return merg, mergs
        return merg


class HourglassSaccade(nn.Module):
    def __init__(self, heads):
        super(HourglassSaccade,self).__init__()

        self.stacks  = 3
        self.heads = heads
        self.pre     = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(128, 256, stride=2)
        )
        self.hgs = nn.ModuleList([
                saccade_module(
                3, [256, 384, 384, 512], [1, 1, 1, 1],
                make_pool_layer=make_pool_layer,
                make_hg_layer=make_hg_layer
            ) for _ in range(self.stacks)
        ])
        self.cnvs    = nn.ModuleList([convolution(3, 256, 256) for _ in range(self.stacks)])
        self.inters  = nn.ModuleList([residual(256, 256) for _ in range(self.stacks - 1)])
        self.cnvs_   = nn.ModuleList([self._merge_mod() for _ in range(self.stacks - 1)])
        self.inters_ = nn.ModuleList([self._merge_mod() for _ in range(self.stacks - 1)])

        # self.att_modules = nn.ModuleList([
        #     nn.ModuleList([
        #         nn.Sequential(
        #             convolution(3, 384, 256, with_bn=False),
        #             nn.Conv2d(256, 1, (1, 1))
        #         ),
        #         nn.Sequential(
        #             convolution(3, 384, 256, with_bn=False),
        #             nn.Conv2d(256, 1, (1, 1))
        #         ),
        #         nn.Sequential(
        #             convolution(3, 256, 256, with_bn=False),
        #             nn.Conv2d(256, 1, (1, 1))
        #         )
        #     ]) for _ in range(self.stacks)
        # ])
        # for att_mod in self.att_modules:
        #     for att in att_mod:
        #         torch.nn.init.constant_(att[-1].bias, -2.19)

        curr_dim = 256
        cnv_dim = 256
        ## keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    make_kp_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(self.stacks)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_kp_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(self.stacks)
                ])
                self.__setattr__(head, module)

        self.relu = nn.ReLU(inplace=True)


    def _merge_mod(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, (1, 1), bias=False),
            nn.BatchNorm2d(256)
        )


    def forward(self, image):
        inter = self.pre(image)
        outs  = []
        for ind in range(self.stacks):
            kp_, cnv_  = self.hgs[ind], self.cnvs[ind]
            kp  = kp_(inter)
            cnv = cnv_(kp)

            # ORIGINAL
            out = {}
            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                y = layer(cnv)
                out[head] = y


            """
            # FOR SAVING ONNX FILE
            out = []
            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                y = layer(cnv)
                out.append(y)
            """
            
            outs.append(out)
            if ind < self.stacks - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return outs

def get_hourglass_saccade(num_layers, heads, head_conv):
  model = HourglassSaccade(heads)
  return model

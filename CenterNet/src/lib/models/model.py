from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import torch.onnx
import os

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.large_hourglass import get_large_hourglass_net
from .networks.large_hourglass_4 import get_large_hourglass_4_net
from .networks.pose_higher_hrnet import get_hrpose_net
from config import cfg, update_config

_model_factory = {
  'res': get_pose_net, # default Resnet with deconv
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'hrnet32': get_hrpose_net,
  'hrnet48': get_hrpose_net,
  'resdcn': get_pose_net_dcn,
  'hourglass': get_large_hourglass_net,
  'hourglass4':get_large_hourglass_4_net
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]

  if("hrnet" in arch):
    cfg_dir = "./cfg_files" if os.path.isdir("./cfg_files") else '../cfg_files'
    if("w48" in arch):
      update_config(cfg, "{}/hrnet_w48_512.yaml".format(cfg_dir))
    else:
      update_config(cfg, "{}/hrnet_w32_512.yaml".format(cfg_dir))
    model = get_model(num_layers=num_layers, cfg = cfg, heads=heads, head_conv=head_conv)
  else:
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None,freeze_backbone=False,freeze_blocks=""):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    new_k = k
    if k.startswith('model'):
      new_k = k[6:]
    if new_k.startswith('module') and not new_k.startswith('module_list'):
      state_dict[new_k[7:]] = state_dict_[k]
    else:
      state_dict[new_k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # Freezes backbone layer
  if freeze_backbone:
    print("Freezing Backbone Layers")
    final_layers = ["hm","wh","hps","reg","hm_hp","hp_offset"]
    for name,param in model.named_parameters():
      if name.split(".")[0] not in final_layers:
        param.requires_grad = False

  # Freezes specified backbone layers
  elif freeze_blocks != "":
    freeze_blocks = freeze_blocks.split(",") 
    if(model.__class__.__name__ == 'DLASeg'):
      for name,param in model.named_parameters():
        if name.split(".")[1] in freeze_blocks:
           print("FREEZING {} OF BACKBONE NETWORK".format(name))
           param.requires_grad = False
    elif(model.__class__.__name__ == 'HourglassNet'):
      for name,param in model.named_parameters():
        if any(block in name for block in freeze_blocks):
           print("FREEZING {} OF BACKBONE NETWORK".format(name))
           param.requires_grad = False


  #dummy_input = torch.randn(1, 3, 512, 512)
  #torch.onnx.export(model, dummy_input, "../model.onnx")


  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)



# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2022/7/26 18:56
# User      : Floyed
# Product   : PyCharm
# Project   : BrainCog
# File      : vgg_snn.py
# explain   :

from functools import partial
from torch.nn import functional as F
import torchvision,pprint
from copy import deepcopy
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.model_zoo.resnet19_snn import *
from braincog.model_zoo.resnet import resnet34
from braincog.model_zoo.sew_resnet import *
from braincog.model_zoo.vgg_snn import *
from braincog.datasets import is_dvs_data



def n_detach(self):
    self.mem=self.mem.detach()
    self.spike=self.spike.detach()
    
def detach(self):
    for mod in self.modules():
        if hasattr(mod, 'n_detach'):
            mod.n_detach()
BaseNode.n_detach=n_detach
BaseModule.detach=detach


def n_deepcopy(self,ori):
    self.mem=ori.mem.clone().detach()
    self.spike=ori.spike.clone().detach()
    #print(self.mem.shape)
    #return copy_node
    
    
def m_deepcopy(self,ori):
    for mod,orimod in zip(self.modules(),ori.modules()):
        if hasattr(mod, 'n_deepcopy') and hasattr(orimod, 'n_deepcopy'):
            mod.n_deepcopy(orimod)
BaseNode.n_deepcopy=n_deepcopy
BaseModule.deepcopy=m_deepcopy




 @register_model
class metarightsltet(BaseModule):

    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False
        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
 
        self.kdloss=nn.KLDivLoss()
        self.learner=eval(kwargs['learner'])(num_classes=num_classes,
                 step=step,
                 node_type=LIFNode,
                 encode_type='direct',
                 sum_output=False,
                 reshape_output=False,
                 *args,
                 **kwargs)
        self.copyopt=kwargs["copyopt"]
        self.loc=kwargs["loc"]
        
    def forward(self, inputs,target=None,loss_fn=None,softloss_fn=None):
        self.target=target
        self.loss_fn=loss_fn
        self.softloss_fn=softloss_fn
         
        outputs = self.learner(inputs)  
 
            
        outputs=[i for i in outputs]
 
        if self.training:
            outputstensor=torch.stack(outputs).transpose(0,1)
            outputsmask=(outputstensor.max(2)[1]==self.target.unsqueeze(1))
            softoutputs=(outputstensor*outputsmask.unsqueeze(2)).sum(1)
            #print(outputsmask.float().mean())
            #outputs=sum(outputs) / len(outputs)
            loss1=self.loss_fn(sum(outputs)/ len(outputs),self.target) 
            loss2=sum([self.softloss_fn(i,softoutputs) for i in outputs])/ len(outputs)
           
            return sum(outputs) / len(outputs),loss1,loss2
        return sum(outputs ) / len(outputs )
              
                   
 
 
                
 
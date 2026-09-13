import os, warnings, yaml, random, math, json, cv2, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', message='Detected call of.*lr_scheduler')

# ===================== CONSTANTS =====================
NUM_PARTS = 5
CLASS_NAMES = ['body', 'left_claw', 'right_claw', 'left_legs', 'right_legs']
CLASS_COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]

# ===================== DATASET =====================
class WeakAugmentation:
    def __call__(self, image, mask=None):
        if random.random()<0.5:
            image=cv2.flip(image,1)
            if mask is not None: mask=np.ascontiguousarray(np.flip(mask[[0,2,1,4,3]],axis=2))
        h,w=image.shape[:2]
        M=cv2.getRotationMatrix2D((w/2,h/2),random.uniform(-10,10),1.0)
        image=cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_LINEAR,borderValue=0)
        if mask is not None:
            mask=np.stack([cv2.warpAffine(mask[c],M,(w,h),flags=cv2.INTER_NEAREST,borderValue=0) for c in range(mask.shape[0])],axis=0)
        s=random.uniform(0.9,1.1)
        nw,nh=int(w*s),int(h*s)
        image=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask=np.stack([cv2.resize(mask[c].astype(np.float32),(nw,nh),interpolation=cv2.INTER_NEAREST) for c in range(mask.shape[0])],axis=0)
        if nh<h or nw<w:
            pt,pl=(h-nh)//2,(w-nw)//2
            image=cv2.copyMakeBorder(image,pt,h-nh-pt,pl,w-nw-pl,cv2.BORDER_CONSTANT,value=0)
            if mask is not None:
                mp=np.zeros((mask.shape[0],h,w),dtype=mask.dtype)
                for c in range(mask.shape[0]): mp[c]=cv2.copyMakeBorder(mask[c],pt,h-nh-pt,pl,w-nw-pl,cv2.BORDER_CONSTANT,value=0)
                mask=mp
        elif nh>h or nw>w:
            ct,cl=(nh-h)//2,(nw-w)//2
            image=image[ct:ct+h,cl:cl+w]
            if mask is not None: mask=mask[:,ct:ct+h,cl:cl+w]
        image=cv2.convertScaleAbs(image,alpha=1+random.uniform(-0.1,0.1),beta=random.uniform(-0.1,0.1)*255)
        return (image,mask) if mask is not None else image

class WeakPhotometricAugmentation:
    def __call__(self, image):
        alpha=1+random.uniform(-0.1,0.1); beta=random.uniform(-0.1,0.1)*255
        return cv2.convertScaleAbs(image,alpha=alpha,beta=beta)

class GeometricAugmentation:
    def __call__(self, image):
        if random.random()<0.5: image=cv2.flip(image,1)
        h,w=image.shape[:2]
        M=cv2.getRotationMatrix2D((w/2,h/2),random.uniform(-15,15),1.0)
        image=cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_LINEAR,borderValue=0)
        s=random.uniform(0.8,1.2)
        nw,nh=int(w*s),int(h*s)
        image=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
        if nh<h or nw<w:
            pt,pl=(h-nh)//2,(w-nw)//2
            image=cv2.copyMakeBorder(image,pt,h-nh-pt,pl,w-nw-pl,cv2.BORDER_CONSTANT,value=0)
        elif nh>h or nw>w:
            ct,cl=(nh-h)//2,(nw-w)//2
            image=image[ct:ct+h,cl:cl+w]
        return image

class StrongPhotometricAugmentation:
    def __call__(self, image):
        if random.random()<0.8:
            image=cv2.convertScaleAbs(image,alpha=1+random.uniform(-0.2,0.2),beta=random.uniform(-0.2,0.2)*255)
            hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:,0]=(hsv[:,:,0]+random.uniform(-30,30))%180
            hsv[:,:,1]=np.clip(hsv[:,:,1]*random.uniform(0.8,1.2),0,255)
            image=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB)
        if random.random()<0.3: image=cv2.GaussianBlur(image,(random.choice([3,5]),random.choice([3,5])),0)
        if random.random()<0.3: image=np.clip(image.astype(np.float32)+np.random.randn(*image.shape)*20,0,255).astype(np.uint8)
        return image
    def __call__(self, image):
        if random.random()<0.5: image=cv2.flip(image,1)
        h,w=image.shape[:2]
        M=cv2.getRotationMatrix2D((w/2,h/2),random.uniform(-20,20),1.0)
        image=cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_LINEAR,borderValue=0)
        s=random.uniform(0.8,1.2)
        nw,nh=int(w*s),int(h*s)
        image=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
        if nh<h or nw<w:
            pt,pl=(h-nh)//2,(w-nw)//2
            image=cv2.copyMakeBorder(image,pt,h-nh-pt,pl,w-nw-pl,cv2.BORDER_CONSTANT,value=0)
        elif nh>h or nw>w:
            ct,cl=(nh-h)//2,(nw-w)//2
            image=image[ct:ct+h,cl:cl+w]
        if random.random()<0.8:
            image=cv2.convertScaleAbs(image,alpha=1+random.uniform(-0.2,0.2),beta=random.uniform(-0.2,0.2)*255)
            hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:,0]=(hsv[:,:,0]+random.uniform(-30,30))%180
            hsv[:,:,1]=np.clip(hsv[:,:,1]*random.uniform(0.8,1.2),0,255)
            image=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB)
        if random.random()<0.3: image=cv2.GaussianBlur(image,(random.choice([3,5]),random.choice([3,5])),0)
        if random.random()<0.3: image=np.clip(image.astype(np.float32)+np.random.randn(*image.shape)*20,0,255).astype(np.uint8)
        return image

def normalize_image(image):
    image=image.astype(np.float32)/255.0
    m,s=np.array([0.485,0.456,0.406],dtype=np.float32),np.array([0.229,0.224,0.225],dtype=np.float32)
    return (image-m)/s

class LabeledDataset(Dataset):
    def __init__(self,data_dir,image_size=1024,augment=True):
        self.image_size=image_size; self.augment=augment; self.augmentor=WeakAugmentation() if augment else None
        meta_path=os.path.join(data_dir,'metadata.json')
        self.metadata=[] if not os.path.exists(meta_path) else json.load(open(meta_path,'r',encoding='utf-8'))
    def __len__(self): return len(self.metadata)
    def __getitem__(self,idx):
        item=self.metadata[idx]
        image=cv2.imread(item['image_path'])
        if image is None: image=np.zeros((self.image_size,self.image_size,3),dtype=np.uint8)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask=np.load(item['mask_path']).astype(np.float32)
        if self.augment and self.augmentor is not None:
            h,w=image.shape[:2]
            leg=np.zeros((h,w),dtype=bool)
            for i in [3,4]: leg|=mask[i]>0
            if leg.any() and random.random()<0.5:
                ys,xs=np.where(leg); dy,dx=int((ys.max()-ys.min())*0.3),int((xs.max()-xs.min())*0.3)
                y0,y1,x0,x1=max(0,ys.min()-dy),min(h,ys.max()+dy),max(0,xs.min()-dx),min(w,xs.max()+dx)
                image=cv2.resize(image[y0:y1,x0:x1],(w,h))
                mask=np.stack([cv2.resize(mask[c,y0:y1,x0:x1].astype(np.float32),(w,h),interpolation=cv2.INTER_NEAREST) for c in range(mask.shape[0])],0)
            image,mask=self.augmentor(image,mask)
        image=normalize_image(image)
        return torch.from_numpy(image).permute(2,0,1).float(),torch.from_numpy(mask).float()

class UnlabeledDataset(Dataset):
    def __init__(self,data_dir,image_size=1024,strong_aug=True):
        self.image_size=image_size; self.geo_aug=GeometricAugmentation(); self.weak_photo=WeakPhotometricAugmentation(); self.strong_photo=StrongPhotometricAugmentation() if strong_aug else None
        self.image_paths=[]
        fp=os.path.join(data_dir,'filelist.txt')
        if os.path.exists(fp):
            with open(fp,'r') as f: self.image_paths=[l.strip() for l in f if l.strip()]
        if not self.image_paths:
            for fn in sorted(os.listdir(data_dir)):
                if fn.lower().endswith(('.jpg','.jpeg','.png')): self.image_paths.append(os.path.join(data_dir,fn))
    def __len__(self): return len(self.image_paths)
    def __getitem__(self,idx):
        image=cv2.imread(self.image_paths[idx])
        if image is None: image=np.zeros((self.image_size,self.image_size,3),dtype=np.uint8)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # Shared geometric transform preserves spatial alignment
        geo_image=self.geo_aug(image.copy())
        image_w=self.weak_photo(geo_image.copy())
        image_s=self.strong_photo(geo_image.copy()) if self.strong_photo else geo_image.copy()
        return torch.from_numpy(normalize_image(image_w).transpose(2,0,1).copy()).float(),torch.from_numpy(normalize_image(image_s).transpose(2,0,1).copy()).float(),self.image_paths[idx]

# ===================== GRAPH MODULE =====================
ADJ_MATRIX=torch.tensor([[0,1,1,1,1],[1,0,1,0,0],[1,1,0,0,0],[1,0,0,0,1],[1,0,0,1,0]],dtype=torch.float32)
EDGE_TYPE=torch.tensor([[0,1,1,1,1],[1,0,2,0,0],[1,2,0,0,0],[1,0,0,0,2],[1,0,0,2,0]],dtype=torch.long)

class GATLayer(nn.Module):
    def __init__(self,in_dim,out_dim,num_heads=4,dropout=0.1):
        super().__init__(); self.num_heads=num_heads; self.out_dim=out_dim
        if num_heads>0:
            self.fc=nn.Linear(in_dim,out_dim//num_heads*num_heads,bias=False) if out_dim%num_heads==0 else nn.Linear(in_dim,out_dim,bias=False)
            self.head_dim=out_dim//num_heads if num_heads>0 and out_dim%num_heads==0 else out_dim
            self.attn_a=nn.Parameter(torch.zeros(1,num_heads,self.head_dim*2)); nn.init.xavier_normal_(self.attn_a)
        else: self.fc=nn.Linear(in_dim,out_dim); self.attn_a=None
        self.ln=nn.LayerNorm(out_dim); self.drop=nn.Dropout(dropout)
    def forward(self,x,adj,edge_type=None):
        B,N,D=x.shape; attn=None
        if self.num_heads>0:
            h=self.fc(x).view(B,N,self.num_heads,self.head_dim)
            h_i=h.unsqueeze(2).expand(-1,-1,N,-1,-1); h_j=h.unsqueeze(1).expand(-1,N,-1,-1,-1)
            e=F.leaky_relu((torch.cat([h_i,h_j],dim=-1)*self.attn_a.unsqueeze(0)).sum(dim=-1)).permute(0,3,1,2)
            e=e.masked_fill(adj.unsqueeze(0).unsqueeze(0).expand(B,self.num_heads,-1,-1)==0,float('-inf'))
            attn=self.drop(F.softmax(e,dim=-1))
            output=torch.bmm(attn.reshape(B*self.num_heads,N,N),h.permute(0,2,1,3).reshape(B*self.num_heads,N,self.head_dim)).reshape(B,self.num_heads,N,self.head_dim).permute(0,2,1,3).reshape(B,N,-1)
        else: output=F.relu(self.fc(x)); attn=None
        return self.drop(F.elu(self.ln(output))), attn

class StructureAwarePartGraph(nn.Module):
    def __init__(self,node_dim=128,hidden_dim=256,num_layers=2,num_heads=4,dropout=0.1):
        super().__init__(); self.register_buffer('adj_matrix',ADJ_MATRIX); self.register_buffer('edge_type',EDGE_TYPE)
        self.gat_layers=nn.ModuleList()
        self.gat_layers.append(GATLayer(node_dim,hidden_dim,num_heads,dropout))
        for _ in range(num_layers-2): self.gat_layers.append(GATLayer(hidden_dim,hidden_dim,num_heads,dropout))
        if num_layers>1: self.gat_layers.append(GATLayer(hidden_dim,node_dim,1,dropout))
    def forward(self,x,return_attention=False):
        adj=self.adj_matrix+torch.eye(5,device=x.device)
        attns=[]
        for layer in self.gat_layers: x,attn=layer(x,adj,self.edge_type); attns.append(attn) if attn is not None else None
        return (x,attns) if return_attention else x

class StructureAwarePartReasoner(nn.Module):
    def __init__(self,node_dim=128,hidden_dim=256,num_layers=2,num_heads=4,dropout=0.1):
        super().__init__()
        self.graph=StructureAwarePartGraph(node_dim,hidden_dim,num_layers,num_heads,dropout)
        self.spatial_encoder=nn.Sequential(nn.Linear(4,hidden_dim//2),nn.ReLU(),nn.Linear(hidden_dim//2,node_dim))
    def forward(self,node_features,return_attention=False):
        return self.graph(node_features,return_attention)
    def refine_pseudo_labels(self,pred_masks,pred_logits):
        B,N,H,W=pred_masks.shape; refined=pred_masks.clone(); adj=self.graph.adj_matrix.to(pred_masks.device)
        SPATIAL_PRIOR={(0,1):(-0.1,-0.2),(0,2):(0.1,-0.2),(0,3):(-0.15,0.1),(0,4):(0.15,0.1),(1,2):(0.3,0.0),(3,4):(0.3,0.0)}
        for b in range(B):
            centroids=[]
            for n in range(N):
                m=pred_masks[b,n]
                if m.sum()>0: ys,xs=torch.where(m>0.5); cy,cx=ys.float().mean()/H,xs.float().mean()/W
                else: cy,cx=0.5,0.5
                centroids.append(torch.tensor([cy,cx],device=m.device))
            centroids=torch.stack(centroids)
            for i in range(N):
                for j in range(N):
                    if adj[i,j]>0 and i!=j and (i,j) in SPATIAL_PRIOR:
                        dx_prior,dy_prior=SPATIAL_PRIOR[(i,j)]
                        ##########################################################
                        # spatial_dist=((centroids[j,1]-centroids[i,1]-dx_prior)**2+(centroids[j,0]-centroids[i,0]-dy_prior)**2).sqrt()
                        # if spatial_dist>0.3: refined[[b,j]if pred_logits[b,i].sigmoid().mean()>pred_logits[b,j].sigmoid().mean() else[b,i]]*=0.5
                        ##########################################################
                        pass
        return refined

# ===================== PROTOTYPE =====================
class PartPrototypeMemory(nn.Module):
    def __init__(self,num_classes=NUM_PARTS,feat_dim=128,momentum=0.999,temperature=0.07):
        super().__init__(); self.num_classes=num_classes; self.feat_dim=feat_dim; self.momentum=momentum; self.temperature=temperature
        self.prototypes=nn.Parameter(torch.zeros(num_classes,feat_dim)); nn.init.xavier_normal_(self.prototypes)
        self.projection=nn.Sequential(nn.Linear(feat_dim,feat_dim),nn.ReLU(),nn.Linear(feat_dim,feat_dim))
    @torch.no_grad()
    def update_prototypes(self,node_features,present_mask=None):
        projected=self.projection(node_features)
        for b in range(len(node_features)):
            for n in range(self.num_classes):
                if present_mask is not None and not present_mask[b,n]: continue
                self.prototypes[n].data.mul_(self.momentum).add_(projected[b,n].detach(),alpha=1-self.momentum)
        self.prototypes.data=F.normalize(self.prototypes.data,dim=-1)
    def contrastive_loss(self,node_features,present_mask=None):
        B,N,D=node_features.shape; projected=F.normalize(self.projection(node_features),dim=-1)
        prototypes=F.normalize(self.prototypes,dim=-1)
        sim=torch.bmm(projected,prototypes.unsqueeze(0).expand(B,-1,-1).transpose(1,2))
        if present_mask is not None:
            loss=0.0; count=0
            for b in range(B):
                present=torch.where(present_mask[b])[0]
                for i in present:
                    exp_pos=torch.exp(sim[b,i,i]/self.temperature)
                    exp_neg=torch.exp(sim[b,i,torch.arange(N,device=sim.device)!=i]/self.temperature).sum()
                    loss-=torch.log(exp_pos/(exp_pos+exp_neg+1e-8)); count+=1
            return loss/count if count>0 else torch.tensor(0.0,device=sim.device)
        pos_sim=sim.diagonal(dim1=1,dim2=2); exp_sim=torch.exp(sim/self.temperature)
        pos_exp=torch.exp(pos_sim/self.temperature)
        return -torch.log(pos_exp/(pos_exp+(exp_sim.sum(dim=-1)-pos_exp)+1e-8)).mean()
    def forward(self,node_features,present_mask=None):
        if self.training: self.update_prototypes(node_features,present_mask)
        return self.contrastive_loss(node_features,present_mask)

# ===================== MODEL =====================
class ResNetFPN(nn.Module):
    def __init__(self,output_dim=256):
        super().__init__(); base=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.conv1=base.conv1; self.bn1=base.bn1; self.relu=base.relu; self.maxpool=base.maxpool
        self.layer1=base.layer1; self.layer2=base.layer2; self.layer3=base.layer3; self.layer4=base.layer4
        self.lat4=nn.Conv2d(2048,output_dim,1); self.lat3=nn.Conv2d(1024,output_dim,1)
        self.lat2=nn.Conv2d(512,output_dim,1); self.lat1=nn.Conv2d(256,output_dim,1)
        self.smooth4=nn.Conv2d(output_dim,output_dim,3,padding=1); self.smooth3=nn.Conv2d(output_dim,output_dim,3,padding=1)
        self.smooth2=nn.Conv2d(output_dim,output_dim,3,padding=1); self.smooth1=nn.Conv2d(output_dim,output_dim,3,padding=1)
    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x))); x=self.maxpool(x)
        c2=self.layer1(x); c3=self.layer2(c2); c4=self.layer3(c3); c5=self.layer4(c4)
        p5=self.smooth4(self.lat4(c5))
        p4=self.smooth3(self.lat3(c4)+F.interpolate(p5,c4.shape[-2:],mode='bilinear',align_corners=False))
        p3=self.smooth2(self.lat2(c3)+F.interpolate(p4,c3.shape[-2:],mode='bilinear',align_corners=False))
        p2=self.smooth1(self.lat1(c2)+F.interpolate(p3,c2.shape[-2:],mode='bilinear',align_corners=False))
        return p2

class PartAwareAdapter(nn.Module):
    def __init__(self,encoder_dim=256,adapter_dim=128,num_classes=NUM_PARTS):
        super().__init__(); self.num_classes=num_classes; self.adapter_dim=adapter_dim
        self.part_projections=nn.ModuleList([nn.Sequential(nn.Conv2d(encoder_dim,adapter_dim,3,padding=1),nn.GroupNorm(16,adapter_dim),nn.ReLU(),nn.Conv2d(adapter_dim,adapter_dim,3,padding=1),nn.GroupNorm(16,adapter_dim),nn.ReLU()) for _ in range(num_classes)])
        self.part_pool=nn.AdaptiveAvgPool2d(1)
    def forward(self,features):
        B,C,H,W=features.shape; part_features,node_features=[],[]
        for i in range(self.num_classes):
            pf=self.part_projections[i](features); part_features.append(pf); node_features.append(self.part_pool(pf).view(B,-1))
        return torch.stack(part_features,dim=1),torch.stack(node_features,dim=1)

class MultiScaleDecoder(nn.Module):
    def __init__(self,num_classes=NUM_PARTS,adapter_dim=128,target_size=1024,num_ups=2):
        super().__init__(); self.num_classes=num_classes; self.target_size=target_size
        self.conv1x1=nn.Conv2d(adapter_dim,32,1); self.conv_d2=nn.Conv2d(adapter_dim,32,3,padding=2,dilation=2)
        self.conv_d4=nn.Conv2d(adapter_dim,32,3,padding=4,dilation=4); self.conv_d6=nn.Conv2d(adapter_dim,32,3,padding=6,dilation=6)
        self.fusion=nn.Sequential(nn.Conv2d(128,64,1),nn.GroupNorm(8,64),nn.ReLU(),nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.ReLU())
        stages,in_ch=[],32
        for i in range(num_ups):
            out_ch=16 if i==num_ups-1 else 32
            stages.append(nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1),nn.GroupNorm(8,out_ch),nn.ReLU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)))
            in_ch=out_ch
        self.up_blocks=nn.Sequential(*stages) if stages else nn.Identity()
        self.out_conv=nn.Conv2d(16 if num_ups>0 else 32,1,1)
    def forward(self,part_features):
        B,N,D,H,W=part_features.shape
        x=part_features.reshape(B*N,D,H,W)
        x=torch.cat([self.conv1x1(x),self.conv_d2(x),self.conv_d4(x),self.conv_d6(x)],dim=1)
        x=self.fusion(x); x=self.up_blocks(x); x=self.out_conv(x)
        x=x.reshape(B,N,*x.shape[-2:])
        if x.shape[-2]!=self.target_size: x=F.interpolate(x,size=(self.target_size,self.target_size),mode='bilinear',align_corners=False)
        return x

class PartSegmentModel(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.num_classes=cfg['model']['num_classes']; self.image_size=cfg['model']['image_size']
        self.encoder_dim=cfg['model']['encoder_embed_dim']; self.adapter_dim=cfg['model']['graph_node_dim']; self.graph_hidden=cfg['model']['graph_hidden_dim']; self.ablation=cfg['training']['ablation']
        self.encoder=ResNetFPN(output_dim=self.encoder_dim)
        self.part_adapter=PartAwareAdapter(encoder_dim=self.encoder_dim,adapter_dim=self.adapter_dim,num_classes=self.num_classes)
        use_graph=self.ablation in ['baseline_graph','baseline_graph_sym','baseline_graph_sym_refine','full','semi']
        if use_graph:
            self.part_graph=StructureAwarePartReasoner(node_dim=self.adapter_dim,hidden_dim=self.graph_hidden,num_layers=2,num_heads=4,dropout=0.1)
            self.graph_condition=nn.Sequential(nn.Linear(self.adapter_dim,self.adapter_dim),nn.ReLU(),nn.Linear(self.adapter_dim,self.adapter_dim*2))
        else: self.part_graph=None
        self.prototype_memory=PartPrototypeMemory(num_classes=self.num_classes,feat_dim=self.adapter_dim,momentum=0.999,temperature=0.07)
        self.mask_decoder=MultiScaleDecoder(num_classes=self.num_classes,adapter_dim=self.adapter_dim,target_size=self.image_size,num_ups=2)
        self._init_weights()
    def _init_weights(self):
        encoder_modules=set(self.encoder.modules())
        for m in self.modules():
            if m in encoder_modules: continue
            if isinstance(m,nn.BatchNorm2d): continue
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.constant_(m.bias,0) if m.bias is not None else None
    def forward(self,images,return_graph=False,return_proto_loss=False,present_mask=None):
        encoder_feats=self.encoder(images)
        part_features,node_features=self.part_adapter(encoder_feats)
        graph_output=None
        if self.part_graph is not None:
            refined_nodes=self.part_graph(node_features); graph_output=refined_nodes
            cond=self.graph_condition(refined_nodes)
            scale=cond[:,:,:self.adapter_dim].unsqueeze(-1).unsqueeze(-1); shift=cond[:,:,self.adapter_dim:].unsqueeze(-1).unsqueeze(-1)
            part_features=part_features*(1+scale)+shift
        logits=self.mask_decoder(part_features)
        proto_loss=self.prototype_memory(node_features,present_mask=present_mask) if (return_proto_loss or self.training) else None
        if return_graph and return_proto_loss: return logits,node_features,graph_output,proto_loss
        if return_graph: return logits,node_features,graph_output
        if return_proto_loss: return logits,proto_loss
        return logits
    def get_trainable_parameters(self): return [p for p in self.parameters() if p.requires_grad]

class EMAModel(nn.Module):
    def __init__(self,model,decay=0.999): super().__init__(); self.model=model; self.decay=decay; self.register_buffer('decay_buf',torch.tensor(decay))
    @torch.no_grad()
    def update(self,student):
        for t_param,s_param in zip(self.model.parameters(),student.parameters()): t_param.data.mul_(self.decay).add_(s_param.data,alpha=1-self.decay)
    def forward(self,images): return self.model(images)

# ===================== LOSSES =====================
class BCEDiceLoss(nn.Module):
    def __init__(self,bce_weight=1.0,dice_weight=1.0,class_weights=None):
        super().__init__(); self.bce_weight=bce_weight; self.dice_weight=dice_weight; self.class_weights=class_weights
    def set_class_weights(self,w): self.class_weights=w
    def forward(self,pred,target):
        B,C,H,W=pred.shape
        bce=F.binary_cross_entropy_with_logits(pred,target,reduction='none').view(B,C,-1).mean(dim=2)
        if self.class_weights is not None: bce=(bce*self.class_weights.to(pred.device).view(1,C)).mean()
        else: bce=bce.mean()
        p=torch.sigmoid(pred)
        intersection=(p*target).sum(dim=(2,3)); union=p.sum(dim=(2,3))+target.sum(dim=(2,3))
        dice_loss=1-(2*intersection+1e-5)/(union+1e-5)
        if self.class_weights is not None: dice_loss=(dice_loss*self.class_weights.to(pred.device).view(1,C)).mean()
        else: dice_loss=dice_loss.mean()
        return self.bce_weight*bce+self.dice_weight*dice_loss

class ConsistencyLoss(nn.Module):
    def __init__(self,threshold=0.7): super().__init__(); self.threshold=threshold
    def forward(self,student_pred,teacher_pred,edge_weights=None):
        tp=torch.sigmoid(teacher_pred.detach())
        confident=(tp>self.threshold)|(tp<1-self.threshold)
        if confident.sum()==0: return torch.tensor(0.0,device=student_pred.device)
        diff=(torch.sigmoid(student_pred)-tp)**2
        if edge_weights is not None: diff=diff*edge_weights
        return diff[confident].mean()

class GraphStructureLoss(nn.Module):
    ADJ_MATRIX=torch.tensor([[0,1,1,1,1],[1,0,1,0,0],[1,1,0,0,0],[1,0,0,0,1],[1,0,0,1,0]],dtype=torch.float32)
    def forward(self,node_features,pred_masks=None):
        B,N,D=node_features.shape; adj=self.ADJ_MATRIX.to(node_features.device)
        norm_feats=F.normalize(node_features,dim=-1); sim=torch.bmm(norm_feats,norm_feats.transpose(1,2))
        pos_mask=adj.unsqueeze(0).expand(B,-1,-1); pos_sim=(sim*pos_mask).sum(dim=(1,2))/pos_mask.sum(dim=(1,2)).clamp(min=1)
        neg_mask=(adj==0).float()-torch.eye(N,device=adj.device); neg_mask=neg_mask.clamp(min=0).unsqueeze(0).expand(B,-1,-1)
        neg_sim=(sim*neg_mask).sum(dim=(1,2))/neg_mask.sum(dim=(1,2)).clamp(min=1)
        contrast_loss=-(pos_sim-neg_sim).mean()
        smooth_loss,lr_loss=0.0,0.0
        for i in range(N):
            for j in range(N):
                if i!=j and adj[i,j]>0: smooth_loss=smooth_loss+(node_features[:,i]-node_features[:,j]).pow(2).mean()
        smooth_loss=smooth_loss/adj.sum().clamp(min=1)
        for li,ri in[(1,2),(3,4)]: lr_loss=lr_loss+(node_features[:,li]-node_features[:,ri]).pow(2).mean()
        lr_loss=lr_loss/2
        return contrast_loss+0.1*smooth_loss+0.1*lr_loss

class BoundaryAwareEdgeLoss(nn.Module):
    def forward(self,pred_masks,edge_weights=None):
        probs=torch.sigmoid(pred_masks); B,C,H,W=probs.shape
        if edge_weights is None: edge_weights=self.compute_boundary_weights(probs)
        p=probs.view(B*C,H,W); w=edge_weights.view(B*C,H,W)
        dx=torch.abs(p[:,:,1:]-p[:,:,:-1]); dy=torch.abs(p[:,1:,:]-p[:,:-1,:])
        wdx=w[:,:,1:]+w[:,:,:-1]; wdy=w[:,1:,:]+w[:,:-1,:]
        return (dx*wdx).mean()+(dy*wdy).mean()
    def compute_boundary_weights(self,probs):
        B,C,H,W=probs.shape; weights=torch.ones_like(probs)
        gx=F.pad(torch.abs(probs[:,:,:,1:]-probs[:,:,:,:-1]),(0,1,0,0),mode='replicate')
        gy=F.pad(torch.abs(probs[:,:,1:,:]-probs[:,:,:-1,:]),(0,0,0,1),mode='replicate')
        grad=(gx+gy)/2
        weights[grad>0.3]=0.3; weights[grad<=0.3]=1.5
        return weights

class TotalLoss(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.lambda_sup=cfg['training']['lambda_sup']; self.lambda_consistency=cfg['training']['lambda_consistency']
        self.lambda_graph=cfg['training']['lambda_graph']; self.lambda_edge=cfg['training']['lambda_edge']; self.ablation=cfg['training']['ablation']
        self.sup_loss=BCEDiceLoss(); self.consistency_loss=ConsistencyLoss(threshold=cfg['training']['pseudo_threshold']); self.graph_loss_fn=GraphStructureLoss(); self.edge_loss=BoundaryAwareEdgeLoss()
    def set_class_weights(self,w): self.sup_loss.set_class_weights(w)
    def forward(self,student_logits,teacher_logits,node_features,gt_masks=None,edge_weights=None):
        losses={}; total=0.0
        if gt_masks is not None:
            losses['sup']=self.sup_loss(student_logits,gt_masks); total+=self.lambda_sup*losses['sup']
        if teacher_logits is not None:
            losses['consistency']=self.consistency_loss(student_logits,teacher_logits,edge_weights)
            if self.ablation not in ['baseline']: total+=self.lambda_consistency*losses['consistency']
        if node_features is not None and self.ablation in ['baseline_graph','baseline_graph_sym','baseline_graph_sym_refine','full','semi']:
            losses['graph']=self.graph_loss_fn(node_features,student_logits); total+=self.lambda_graph*losses['graph']
        if self.ablation in ['full']:
            losses['edge']=self.edge_loss(student_logits,edge_weights); total+=self.lambda_edge*losses['edge']
        losses['total']=total; return losses

# ===================== METRICS =====================
class MetricsTracker:
    def __init__(self,num_classes=NUM_PARTS): self.num_classes=num_classes; self.reset()
    def reset(self): self.ious=[]; self.dices=[]; self.precisions=[]; self.recalls=[]; self.f1s=[]
    def update(self,pred,target):
        pred,target=pred.detach().cpu(),target.detach().cpu()
        pred_bin=(pred>0.5).float(); target_bin=(target>0.5).float()
        ious,dices=[],[]
        for c in range(self.num_classes):
            p_,t_=pred_bin[:,c],target_bin[:,c]; inter=(p_*t_).sum(dim=(1,2)); union=p_.sum(dim=(1,2))+t_.sum(dim=(1,2))-inter
            ious.append((inter/(union+1e-6)).mean().item()); dices.append((2*inter/(p_.sum(dim=(1,2))+t_.sum(dim=(1,2))+1e-6)).mean().item())
        self.ious.append(ious); self.dices.append(dices)
    def get_metrics(self):
        if not self.ious: return {}
        ious=np.array(self.ious); dices=np.array(self.dices)
        results={'mIoU':float(np.mean(ious)),'Dice':float(np.mean(dices)),'F1':float(np.mean(dices))}
        for c in range(self.num_classes): results[f'mIoU_class_{c}']=float(np.mean(ious[:,c]))
        return results

# ===================== TRAINING =====================
def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)

def compute_class_weights(dataset,num_classes=NUM_PARTS):
    counts=torch.zeros(num_classes); total=0
    for i in range(len(dataset)):
        _,mask=dataset[i]; total+=mask.shape[1]*mask.shape[2]
        for c in range(num_classes): counts[c]+=mask[c].sum()
    weights=total/(counts*num_classes+1e-6); return weights/weights.sum()*num_classes

def load_checkpoint(model,ema_model,optimizer,scheduler,path):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['student_state_dict'])
    if ema_model is not None and 'ema_state_dict' in ckpt: ema_model.load_state_dict(ckpt['ema_state_dict'])
    if optimizer is not None and 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
    start=ckpt.get('epoch',0)+1; best=ckpt.get('best_metric',0.0); print(f"Resumed from epoch {ckpt.get('epoch',0)}"); return start,best

def save_checkpoint(model,ema_model,optimizer,scheduler,epoch,best_metric,path):
    torch.save({'student_state_dict':model.state_dict(),'ema_state_dict':ema_model.state_dict() if ema_model is not None else None,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict() if scheduler is not None else None,'epoch':epoch,'best_metric':best_metric},path)
    print(f"Checkpoint saved: {path}")

def enable_dropout_only(model):
    for m in model.modules():
        if isinstance(m,nn.Dropout): m.train()
        elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)): m.eval()

def teacher_with_mc_dropout(teacher,images,num_samples=4):
    teacher.eval(); enable_dropout_only(teacher)
    preds=[teacher(images) for _ in range(num_samples)]
    teacher.eval(); mean_pred=torch.stack(preds,dim=0).mean(dim=0)
    probs=torch.sigmoid(mean_pred.float()).clamp(min=1e-7,max=1-1e-7)
    entropy=-(probs*torch.log(probs)+(1-probs)*torch.log(1-probs)); uncertainty=entropy.mean(dim=1,keepdim=True)
    return mean_pred,uncertainty

def train_epoch(student,teacher,teacher_ema,train_loader,unlabeled_loader,criterion,optimizer,device,epoch,cfg,graph_module,edge_loss_fn):
    student.train(); teacher.eval(); ablation=cfg['training']['ablation']
    kl_loss=nn.KLDivLoss(reduction='none')
    components={k:0.0 for k in ['sup','pseudo_kl','consistency','prototype','graph','edge','total']}
    li=iter(train_loader); ui=iter(unlabeled_loader) if unlabeled_loader is not None else None
    num_batches=len(train_loader); pbar=tqdm(range(num_batches),desc=f"Epoch {epoch}")
    scaler=GradScaler(device.type)
    for _ in pbar:
        try: imgs_l,masks_l=next(li)
        except StopIteration: li=iter(train_loader); imgs_l,masks_l=next(li)
        imgs_l,masks_l=imgs_l.to(device),masks_l.to(device)
        has_ul=ui is not None
        if has_ul:
            try: imgs_w,imgs_s,_=next(ui)
            except StopIteration: ui=iter(unlabeled_loader); imgs_w,imgs_s,_=next(ui)
            imgs_w,imgs_s=imgs_w.to(device),imgs_s.to(device)
        optimizer.zero_grad()
        with autocast(device.type):
            pm=(masks_l.sum(dim=(2,3))>0)
            labeled_logits,nf_l,_,proto=student(imgs_l,return_graph=True,return_proto_loss=True,present_mask=pm)
            losses=criterion(labeled_logits,None,nf_l,gt_masks=masks_l); total=losses.get('total',0)
            if ablation not in ['baseline']: losses['prototype']=proto; total+=0.05*proto
            if has_ul and ablation not in ['baseline']:
                with torch.no_grad():
                    if ablation=='semi':
                        t_logits=teacher(imgs_w); t_probs=torch.sigmoid(t_logits)
                    else:
                        t_logits,t_unc=teacher_with_mc_dropout(teacher,imgs_w,num_samples=3); t_probs=torch.sigmoid(t_logits)
                        u_w=1.0/(t_unc+1.0); u_w=u_w/u_w.mean()
                    soft=t_probs
                s_logits_s,nf_s,_=student(imgs_s,return_graph=True); s_probs=torch.sigmoid(s_logits_s)
                if ablation=='semi':
                    eps=1e-7; p=soft.float().clamp(eps,1-eps); q=s_probs.float().clamp(eps,1-eps)
                    conf=(p>0.95)|(p<1-0.95); kl=p*torch.log(p/q)+(1-p)*torch.log((1-p)/(1-q))
                    kl=(kl*conf.float()); pk=kl.sum()/conf.float().sum().clamp(min=1)
                    total+=cfg['training']['semi_pseudo_weight']*pk
                else:
                    sp=soft.float().clamp(eps:=1e-7,1-eps); sq=s_probs.float().clamp(eps,1-eps)
                    kl_px=kl_loss(torch.log(sq),sp.detach()); kl_s=kl_px.mean(dim=(1,2,3),keepdim=True)
                    pk=(kl_s*u_w.float()).mean(); total+=criterion.lambda_consistency*pk
                losses['pseudo_kl']=pk
                cons=criterion.consistency_loss(s_logits_s,t_logits); total+=0.1*criterion.lambda_consistency*cons; losses['consistency']=cons
                if ablation in ['baseline_graph','baseline_graph_sym','baseline_graph_sym_refine','full','semi']:
                    gl=criterion.graph_loss_fn(nf_s,s_logits_s); total+=criterion.lambda_graph*gl; losses['graph']=gl
                if ablation=='full':
                    ew=edge_loss_fn.compute_boundary_weights(t_probs.to(imgs_s.device)).detach()
                    el=edge_loss_fn(s_logits_s,ew); total+=criterion.lambda_edge*el; losses['edge']=el
        losses['total']=total
        scaler.scale(total).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(student.parameters(),1.0)
        scaler.step(optimizer); scaler.update(); teacher_ema.update(student)
        total_item=total.item()
        for k in components:
            if k in losses: components[k]+=losses[k].item()
        pbar.set_postfix({'loss':f"{total_item:.4f}",'sup':f"{losses.get('sup',0):.4f}"})
    return {k:v/num_batches for k,v in components.items()}

@torch.no_grad()
def validate(student,val_loader,device):
    student.eval(); tracker=MetricsTracker()
    for imgs,masks in tqdm(val_loader,desc="Val"):
        imgs,masks=imgs.to(device),masks.to(device)
        tracker.update(torch.sigmoid(student(imgs)),masks)
    return tracker.get_metrics()

def main():
    with open('config.yaml','r') as f: cfg=yaml.safe_load(f)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")
    torch.backends.cudnn.benchmark=True; set_seed(42)
    tc=cfg['training']; pd=cfg['data']['processed_dir']
    train_ds=LabeledDataset(os.path.join(pd,'train'),cfg['model']['image_size'],augment=True)
    val_ds=LabeledDataset(os.path.join(pd,'val'),cfg['model']['image_size'],augment=False)
    train_loader=DataLoader(train_ds,batch_size=tc['batch_size'],shuffle=True,num_workers=2,pin_memory=True,drop_last=len(train_ds)>tc['batch_size'])
    val_loader=DataLoader(val_ds,batch_size=tc['batch_size'],shuffle=False,num_workers=2,pin_memory=True)
    ul_loader=None
    if os.path.exists(os.path.join(pd,'unlabeled')):
        ul_ds=UnlabeledDataset(os.path.join(pd,'unlabeled'),cfg['model']['image_size'])
        if len(ul_ds)>0: ul_loader=DataLoader(ul_ds,batch_size=tc['batch_size'],shuffle=True,num_workers=2,pin_memory=True,drop_last=len(ul_ds)>tc['batch_size']); print(f"Unlabeled: {len(ul_ds)}")
    student=PartSegmentModel(cfg).to(device)
    ema=EMAModel(PartSegmentModel(cfg),decay=tc['ema_decay']).to(device); ema.model.eval()
    with torch.no_grad():
        for p,q in zip(ema.model.parameters(),student.parameters()): p.data.copy_(q.data)
    criterion=TotalLoss(cfg).to(device); edge_loss_fn=BoundaryAwareEdgeLoss(); gm=student.part_graph if hasattr(student,'part_graph') else None
    print("Computing class weights..."); cw=compute_class_weights(train_ds); print(f"Class weights: {cw.tolist()}"); criterion.set_class_weights(cw)
    opt=AdamW(student.get_trainable_parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    base_lr=tc['learning_rate']; wu=tc['warmup_epochs']; te=tc['num_epochs']; mlr=tc['min_lr']
    sched=LambdaLR(opt,lr_lambda=lambda e: (e+1)/wu if e<wu else max(mlr/base_lr,0.5*(1+np.cos(np.pi*(e-wu)/max(te-wu,1)))))
    start_epoch=0; best_metric=0.0; best_epoch=0; no_improve=0; patience=15
    out_dir=cfg['data']['output_dir']; os.makedirs(out_dir,exist_ok=True)
    resume=tc.get('resume',None)
    if resume and os.path.exists(resume): start_epoch,best_metric=load_checkpoint(student,ema,opt,sched,resume)
    print(f"Training {tc['num_epochs']} epochs, ablation: {tc['ablation']}, Train: {len(train_ds)}, Val: {len(val_ds)}")
    for epoch in range(start_epoch,tc['num_epochs']):
        tl=train_epoch(student,ema.model,ema,train_loader,ul_loader,criterion,opt,device,epoch,cfg,gm,edge_loss_fn)
        sched.step()
        print(f"  Losses: sup={tl.get('sup',0):.4f}  proto={tl.get('prototype',0):.4f}  pseudo_kl={tl.get('pseudo_kl',0):.4f}  cons={tl.get('consistency',0):.4f}  graph={tl.get('graph',0):.4f}  edge={tl.get('edge',0):.4f}")
        if epoch%cfg['eval']['val_interval']==0:
            vm=validate(student,val_loader,device); cn=cfg['model']['class_names']
            pc='  '.join([f"{cn[i]}={vm[f'mIoU_class_{i}']:.3f}" for i in range(len(cn))])
            print(f"Epoch {epoch}: mIoU={vm['mIoU']:.4f}  Dice={vm['Dice']:.4f}  F1={vm['F1']:.4f}\n  Per-class: {pc}")
            cur=vm[cfg['eval']['metric']]
            if cur>best_metric:
                best_metric=cur; best_epoch=epoch; no_improve=0
                save_checkpoint(student,ema,opt,sched,epoch,best_metric,os.path.join(out_dir,'best_model.pth'))
                print(f"New best: {cur:.4f}")
            else: no_improve+=1
            save_checkpoint(student,ema,opt,sched,epoch,best_metric,os.path.join(out_dir,'latest_checkpoint.pth'))
            if no_improve>=patience: print(f"Early stopping at epoch {epoch}, best epoch {best_epoch} with {best_metric:.4f}"); break
    print(f"Best {cfg['eval']['metric']}: {best_metric:.4f} at epoch {best_epoch}")

if __name__=='__main__': main()
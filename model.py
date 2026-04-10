import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.nn.functional import silu
from types import MethodType
import dnnlib
from networks import EDMPrecond, UNetBlock 

class ModifiedEDMPrecond(EDMPrecond):
    def forward(self, x, a_list, t, sigma, sigma_final,class_labels=None, force_fp32=False):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        sigma_final = sigma_final.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        # print(self.label_dim)
        if self.label_dim > 0:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)
        else:
            class_labels = None
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
    
        c_noise_final = sigma_final.log() / 4

        F_x, up_sample_list = self.model((c_in * x).to(dtype), a_list, t, c_noise.flatten(), c_noise_final.flatten(),class_labels=class_labels)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x,up_sample_list

    @classmethod
    def from_pkl(cls, pkl_path):
        
        with dnnlib.util.open_url(pkl_path, verbose=True) as f:
            data = pickle.load(f)
        pretrained = data["ema"]

        inner_kwargs = dict(pretrained._init_kwargs)
        model = cls(**inner_kwargs)
        model.model.load_state_dict(pretrained.model.state_dict(), strict=True)
        

        # Patch forward
        model.model.forward = ModifiedEDMPrecond.patch_forward(model.model)
        return model

    @staticmethod
    def patch_forward(model):
        def forward(self, x, a_list, t, noise_labels, noise_labels_final,class_labels, augment_labels=None):
            emb = self.map_noise(noise_labels)
            emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
            
            emb_final = self.map_noise(noise_labels_final)
            emb_final = emb_final.reshape(emb_final.shape[0], 2, -1).flip(1).reshape(*emb_final.shape)
            
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
                emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
            if self.map_augment is not None and augment_labels is not None:
                emb = emb + self.map_augment(augment_labels)
            emb = silu(self.map_layer0(emb))
            emb = silu(self.map_layer1(emb))
            
            emb_final = silu(self.map_layer0(emb_final))
            emb_final = silu(self.map_layer1(emb_final))

            def compute_w_t(t, w0, a):
                n_batch, w_size, h, w = w0.shape
                w0 = w0.view(n_batch, w_size, h * w)
                t = torch.tensor(t / 18).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(n_batch, w_size // 2, h * w)
                t = t.to(w0.device)
                alpha = a[:, :w_size // 2, :]
                beta = a[:, w_size // 2:, :]
                alpha_t = torch.exp(alpha * t) * torch.cos(beta * t)
                beta_t = torch.exp(alpha * t) * torch.sin(beta * t)
                w_ = torch.zeros_like(w0)
                w_[..., 0::2, :] = w0[..., 1::2, :]
                w_[..., 1::2, :] = w0[..., 0::2, :]
                w_[..., 1::2, :] *= -1
                beta_t_result = w_ * torch.repeat_interleave(beta_t, 2, dim=-2)
                alpha_t_result = w0 * torch.repeat_interleave(alpha_t, 2, dim=-2)
                w_t = alpha_t_result + beta_t_result
                return w_t.view(n_batch, w_size, h, w)

            skips = []
            aux = x
            for i, (name, block) in enumerate(self.enc.items()):
                if 'aux_down' in name:
                    print("aux_down")
                    aux = block(aux)
                elif 'aux_skip' in name:
                    print("aux_skip")
                    x = skips[-1] = x + block(aux)
                elif 'aux_residual' in name:
                    print("aux_residual")
                    x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
                else:
                    
                    x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                    x_koopman = compute_w_t(t, x, a_list[i])
                    skips.append(x_koopman)

           
            x = x_koopman
            aux = tmp = None
            up_sample_list = []
            for name, block in self.dec.items():
                if 'aux_up' in name:
                    aux = block(aux)
                    up_sample_list.append(aux)
                elif 'aux_norm' in name:
                    
                    tmp = block(x)
                    # up_sample_list.append(tmp)
                elif 'aux_conv' in name:
                    
                    tmp = block(silu(tmp))
                    aux = tmp if aux is None else tmp + aux
                    # up_sample_list.append(aux)
                else:
                   
                    if x.shape[1] != block.in_channels:
                        x = torch.cat([x, skips.pop()], dim=1)
                    x = block(x, emb_final)
                    up_sample_list.append(x)
                    
            return aux, up_sample_list

        return MethodType(forward, model)


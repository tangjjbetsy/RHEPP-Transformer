from model import *
from data_preprocessing import *
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig

import os
import copy
import numpy as np
import argparse
import pytorch_lightning as pl
MAX_LEN = 1000
MASK_WORD = [1,0,1,1,1,1]

class DealDataset(Dataset):
    def __init__(self, x_data, mask_data, y_data, performer):
        self.x_data = x_data
        self.mask_data = mask_data
        self.y_data = y_data
        self.performer = performer
        self.len = x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.mask_data[index], self.y_data[index], self.performer[index]

    def __len__(self):
        return self.len

def data_loader(data_X, data_mask, data_y, data_performer, shuffle=True):
    data = DealDataset(data_X, data_mask, data_y, data_performer)
    loader = DataLoader(dataset=data,           
                    batch_size=16, 
                    shuffle=shuffle,
                    num_workers=4, 
                    pin_memory=True)
    return loader

class DataModule(pl.LightningDataModule):
    def __init__(self, data=None, verbose=True):
        super().__init__()
        
        self.data = data
        self.verbose = verbose

    def train_dataloader(self):
        return data_loader(self.data['x_train'], self.data['mask_train'], self.data['y_train'], self.data['performer_train'])

    def val_dataloader(self):
        return data_loader(self.data['x_valid'], self.data['mask_valid'], self.data['y_valid'], self.data['performer_valid'], False)

    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'samples={len(self.samples)}, '
                f'avglen={self.avglen})')

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class MyLightningModule(pl.LightningModule):
    def __init__(self, 
                 model, 
                 learning_rate=1e-4, 
                 weight_decay=1e-7):
        
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.model = model
        self.midibert = model.midibert
        
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.L1Loss(reduction="none")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.Weightloss1 = torch.tensor(1.0, requires_grad=True).to(self.device)
        self.Weightloss2 = torch.tensor(1.0, requires_grad=True).to(self.device)
        self.Weightloss3 = torch.tensor(1.0, requires_grad=True).to(self.device)
        
        self.params = [self.Weightloss1, self.Weightloss2, self.Weightloss3]
    
    def forward(self, x, attn_masks, performers):
        return self.model(x, attn_masks, performers)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer_weights = optim.Adam(self.params, lr=0.001)
        self.lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=3000, max_iters=12000)
        return optimizer, optimizer_weights
        
    def compute_loss(self, predict, target, loss_mask):
        idx = [1,2,5]
        all_loss = []
        for i in range(3):  
            zero_index = torch.where(target[:,:,idx[i]].float() == 0)
            target_values = target[:,:,idx[i]].float()
            target_values[target_values == 0] = 1
            loss = self.loss_mse(predict[i].squeeze(), target[:,:,idx[i]].float())
            loss = loss * loss_mask[:, :, idx[i]]
            
            new_mask = torch.ones_like(loss)
            new_mask[zero_index] = new_mask[zero_index] * 1e-3
            loss = loss * new_mask
            loss = loss / torch.abs(target_values)
            
            loss = torch.sum(loss, dim=-1) / torch.sum(loss_mask[:,:,idx[i]])
            all_loss.append(loss)
        total_loss = [x.sum() for x in all_loss]
        return torch.stack(total_loss)
    
    def compute_accuracy(self, predict, target, loss_mask):
        temp = []
        for i in range(3):
            temp.append(torch.round(predict[i].squeeze()))
        temp = torch.stack(temp, dim = -1)
        
        if len(temp.shape) == 2:
            temp = temp[None]
        all_acc = []
        
        idx = [1,2,5]
        for i in range(3):
            acc = torch.sum((temp[:,:,i] == target[:,:,idx[i]]).float() * loss_mask[:,:,idx[i]])
            acc /= torch.sum(loss_mask[:,:,idx[i]])
            all_acc.append(acc)
        total_acc = [x.sum() for x in all_acc]
        return total_acc

    def get_mask_ind(self):
        mask_ind = random.sample([i for i in range(MAX_LEN)], round(MAX_LEN))
        mask50 = random.sample(mask_ind, round(len(mask_ind)*0.8))
        left = list((set(mask_ind) - set(mask50)))
        rand25 = random.sample(left, round(len(mask_ind)*0.2))
        cur25= list(set(left)-set(rand25))
        return mask50, rand25, cur25
    
    def create_masks(self, inputs):
        mask50, rand25, cur25 = self.get_mask_ind()
        input_seqs = copy.deepcopy(inputs)
        self.loss_mask = torch.ones(inputs.shape)
        for b in range(input_seqs.shape[0]):
            for i in mask50:
                mask_word = torch.tensor(MASK_WORD).to(self.device)
                input_seqs[b][i] *= mask_word 
            for i in rand25:
                rand_word = torch.tensor(self.midibert.get_rand_tok()).to(self.device)
                input_seqs[b][i][1] = rand_word 
        self.loss_mask = self.loss_mask.to(self.device)
        return input_seqs.long()
                
    def training_step(self, batch, batch_idx):
        inputs, masks, targets, performers = batch
        inputs = inputs[..., [0,1,2,3,4,6]]
        targets = targets[..., [0,1,2,3,4,6]]
        inputs_masked = self.create_masks(inputs)
        outputs = self(inputs_masked, masks, performers)
        acces = self.compute_accuracy(outputs, targets, self.loss_mask)
        losses = self.compute_loss(outputs, targets, self.loss_mask)
        opt, opt_w = self.optimizers()
        
        if self.global_step == 0:
            # init weights
            self.T = torch.sum(torch.tensor(self.params)).detach() # sum of weights
            self.l0 = losses.detach()
                
        weighted_loss = (self.params[0] * losses[0] \
                        + self.params[1] * losses[1] \
                        + self.params[2] * losses[2])/torch.sum(torch.tensor(self.params))
        # clear gradients of network
        opt.zero_grad()
        # backward pass for weigthted task 
        self.manual_backward(weighted_loss, retain_graph=True)
        
        # compute the L2 norm of the gradients for each task
        gw = []
        
        for i in range(len(losses)):
            with torch.autograd.set_detect_anomaly(True):
                dl = torch.autograd.grad(self.params[i]*losses[i], 
                                         self.model.mask_lm.proj[i].parameters(), 
                                         retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
        gw = torch.stack(gw)
        # compute loss ratio per task
        loss_ratio = losses.detach() / self.l0
        # compute the relative inverse training rate per task
        rt = loss_ratio / loss_ratio.mean()
        # compute the average gradient norm
        gw_avg = gw.mean().detach()
        # compute the GradNorm loss
        constant = (gw_avg * rt ** 0.16).detach()
        gradnorm_loss = torch.abs(gw - constant).sum()
        
        # clear gradients of weights
        opt_w.zero_grad()
        # backward pass for GradNorm
        self.manual_backward(gradnorm_loss)
        # gradnorm_loss.backward()
        
        
        # update model weights
        opt.step()
        self.lr_scheduler.step()
        # update loss weights
        opt_w.step()
        # renormalize weights
        coef = 3 / (self.Weightloss1 + self.Weightloss2 + self.Weightloss3)
        self.params = [coef*self.Weightloss1, 
                       coef*self.Weightloss2, 
                       coef*self.Weightloss3]
        
        accuracy = (acces[0] + acces[1] + acces[2])/3
        
        self.log('loss_weight', {'velocity_weight': self.Weightloss1,
                                 'duration_weight': self.Weightloss2,
                                 'ioi_weight': self.Weightloss3
                                 }, on_epoch=True, prog_bar=False, on_step=False)
        
        self.log('train_loss', {"loss": weighted_loss,
                                'velocity_loss': losses[0],
                                'duration_loss': losses[1],
                                'ioi_loss': losses[2],
                                }, on_epoch=True, prog_bar=False, on_step=False)
        
        self.log('train_acc', {"accuracy": accuracy,
                                'velocity_acc': acces[0],
                                'duration_acc': acces[1],
                                'ioi_acc': acces[2]
                                }, on_epoch=True, prog_bar=False, on_step=False)
        
        return
    
    def validation_step(self, batch, batch_idx):
        inputs, masks, targets, performers = batch
        inputs = inputs[..., [0,1,2,3,4,6]]
        targets = targets[..., [0,1,2,3,4,6]]
        inputs_masked = self.create_masks(inputs)
        outputs = self(inputs_masked, masks, performers)
        acces = self.compute_accuracy(outputs, targets, self.loss_mask)
        losses = self.compute_loss(outputs, targets, self.loss_mask)
        weighted_loss = (self.params[0] * losses[0] \
                        + self.params[1] * losses[1] \
                        + self.params[2] * losses[2])/torch.sum(torch.tensor(self.params))
        
        accuracy = (acces[0] + acces[1] + acces[2])/3
        
        self.log('val_loss', weighted_loss, on_epoch=True, prog_bar=True, on_step=False, sync_dist=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True, on_step=False, sync_dist=True)
        return 

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_seq_len', type=int, default=MAX_LEN, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=128)      # hidden state
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--log_path', type=str, default="wb_logs/S2PBert")
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument("--cuda_devices", nargs='+', default=["4","0"], help="CUDA device ids")
    args = parser.parse_args()

    return args

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.cuda_devices)
    
    # Choose training data
    data_path = os.path.join("data/s2p_data_bee_dev_ioi_new.npz")
    data = np.load(data_path)
    
    # Set logger and checkpoint paths.
    logger = WandbLogger(project="S2P", name="s2p_dev_gradnorm_ioi_final", log_model=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_last=True) 
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    configuration = BertConfig(
                                max_position_embeddings=args.max_seq_len,
                                position_embedding_type='relative_key_query',
                                hidden_size=args.hs,
                                num_hidden_layers=4,
                                precision = 16,
                                num_attention_heads=4,
                                intermediate_size=128,
                                )
    
    torch.set_float32_matmul_precision('high')
    
    bertmodel = MidiBert(configuration)    
    mymodel = MidiBertLM(bertmodel)
    model = MyLightningModule(mymodel)
    
    if args.ckpt_path != None:
        model = model.load_from_checkpoint(args.ckpt_path)
    
    print(data['x_train'].shape, data['mask_train'].shape, data['y_train'].shape)
    
    print("loading data ...")
    datamodule = DataModule(data)
    # trainer = pl.Trainer(gpus=1, fast_dev_run=True)
    # trainer.fit(model, datamodule)
    
    print("initiating moodel ...")
    
    trainer = pl.Trainer(max_epochs=args.epochs, 
                         logger=logger,
                         gpus=2,
                         strategy="ddp",
                         enable_progress_bar=True,
                         log_every_n_steps=10,
                         callbacks=[checkpoint_callback, lr_monitor])
    
    print("start training moodel ...")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    args = get_args()
    train(args)

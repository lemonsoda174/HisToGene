import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import HisToGene
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HER2ST, ViT_SKIN


fold = 5
tag = '-htg_her2st_785_32_cv'

dataset = ViT_HER2ST(train=True, fold=fold, ratio=4)
train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
model = HisToGene(n_layers=8, n_genes=785, learning_rate=1e-5)
trainer = pl.Trainer(accelerator='gpu', max_epochs=100)
trainer.fit(model, train_loader)
trainer.save_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt")
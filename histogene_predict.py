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

def normal_pred():
    #normal histogene prediction
    model = HisToGene.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt",n_layers=8, n_genes=785, learning_rate=1e-5)
    device = torch.device("cuda")
    dataset = ViT_HER2ST(train=False,sr=False,fold=fold)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4)
    adata_pred, adata_truth = model_predict(model, test_loader, attention=False, device = device)
    adata_pred = comp_tsne_km(adata_pred,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    print(adata_pred)

    #visualize results
    print(sc.pl.spatial(adata_pred, img=None, color='kmeans', spot_size=112))
    print(sc.pl.spatial(adata_pred, img=None, color='FASN', spot_size=112, color_map='magma'))


def super_pred():
    #super-resolution
    model = HisToGene.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt",n_layers=8, n_genes=785, learning_rate=1e-5)
    dataset_sr = ViT_HER2ST(train=False,sr=True,fold=fold)
    test_loader_sr = DataLoader(dataset_sr, batch_size=1, num_workers=2)
    adata_sr = sr_predict(model, test_loader_sr, attention=False, device = torch.device('cuda'))
    adata_sr = comp_tsne_km(adata_sr,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_sr.var_names = g
    sc.pp.scale(adata_sr)
    print(adata_sr)

    #visualize results
    print(sc.pl.spatial(adata_sr, img=None, color='kmeans', spot_size=56))
    print(sc.pl.spatial(adata_sr, img=None, color='FASN', spot_size=56, color_map='magma'))

super_pred()
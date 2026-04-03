import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime
from tqdm import tqdm

from Model.CPI import *
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics
import sklearn
import warnings
from dataset.Dataset import *
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./CellChem.yaml', os.path.join(model_checkpoints_folder, 'CellChem.yaml'))

def KLLoss(x,y):
    x_log = F.log_softmax(x,dim=-1)
    y = F.softmax(y,dim=-1)
    kl = torch.nn.KLDivLoss(reduction='sum')
    loss = kl(x_log, y)
    return loss
        
class SmilesnCmap(object):
    def __init__(self, i, train_loader,valid_loader, config):
        self.config = config
        self.device = self._get_device()
        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('ckpt', dir_name)
        self.i = i
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device

    def evaluate(self, true,pred_label,pred_score):
        ACC = sklearn.metrics.accuracy_score(true, pred_label, normalize=True, sample_weight=None)
        precision = sklearn.metrics.precision_score(true, pred_label)
        recall = sklearn.metrics.recall_score(true, pred_label)
        f1 = sklearn.metrics.f1_score(true, pred_label)
        auc = sklearn.metrics.roc_auc_score(true, pred_score)
        mcc = sklearn.metrics.matthews_corrcoef(true, pred_label)
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(true, pred_label).ravel()
        sensitivity = 1.0 * TP / (TP + FN)
        specificity = 1.0 * TN / (FP + TN)
        pr, re, thresholds = sklearn.metrics.precision_recall_curve(true, pred_score)
        AUPR = sklearn.metrics.auc(re, pr)
        return ACC, precision, recall, f1, auc, mcc, sensitivity, specificity, AUPR

    def _load_pre_trained_weights(self,model):
        try:
            state_dict = torch.load('./pertain_model_pt/with_perturb.pt')
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
        return model
    
    def train(self):
        protein_dim = 1280
        atom_dim = 256
        hid_dim = 640
        n_layers = 1
        n_heads = 8
        pf_dim = 256
        dropout = 0.1
        batch = 64
        lr = 1e-4
        weight_decay = 1e-4
        decay_interval = 5
        lr_decay = 1.0
        iteration = 300
        kernel_size = 7
        encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, self.device)
        config_file = './config_mg.yaml'
        config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        mol_encoder = GraphTransformer(**config["model"]).to(self.device)
        mol_encoder = self._load_pre_trained_weights(mol_encoder)
        decoder = Decoder(mol_encoder,atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,dropout, self.device)
        model = Predictor(encoder, decoder, self.device).to(self.device)
        print(model)

        """weight initialize"""
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': 1e-4}, {'params': bias_p, 'weight_decay': 0}], lr=5e-5)
        

        model_checkpoints_folder = './Model_CellChem_random_save/'#(e.g. Model_CellChem_random_save,Model_CellChem_scaffold_save)
        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_acc = 0
        
        for epoch_counter in range(self.config['epochs']):
            LOSS = 0.0
            #ACC = 0.0
            Predict_Label = []
            True_Label = []
            Predict_Scores = []
            counter = 0
            for (Protein,Mol,label,Protein_len, Mol_len) in tqdm(self.train_loader):
                optimizer.zero_grad()
                Protein = Protein.to(self.device)
                Mol = Mol.to(self.device)
                label = label.to(self.device)
                Protein_len = Protein_len.to(self.device)
                Mol_len = Mol_len.to(self.device)
                predict_label,predict_scores, loss = model(Mol, Protein, label, Mol_len, Protein_len)
                LOSS += loss.item()
                counter += 1
                predict_label = predict_label.tolist()
                predict_scores = predict_scores.tolist()
                label = label.tolist()
                Predict_Label.extend(predict_label)
                Predict_Scores.extend(predict_scores)
                True_Label.extend(label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1
            LOSS /= counter
            ACC, precision, recall, f1, auc, mcc, sensitivity, specificity, AUPR= self.evaluate(True_Label, Predict_Label,Predict_Scores)
            print('epoch:',epoch_counter,'train_loss:',LOSS,'accuracy:',ACC,'precision:',precision, 'recall:',recall, 
                    'f1:',f1, 'auc:',auc,'mcc:',mcc,'sensitivity:',sensitivity,'specificity:',specificity,'AUPR:', AUPR)

            valid_loss,ACC, precision, recall, f1, auc, mcc, sensitivity, specificity, AUPR = self._validate(model, self.valid_loader)
            print('validation:','epoch:',epoch_counter, 'valid_loss:',valid_loss,'valid_accuracy:',ACC,'precision:',precision, 'recall:',recall, 
                'f1:',f1, 'auc:',auc,'mcc:',mcc,'sensitivity:',sensitivity,'specificity:',specificity,'AUPR:', AUPR)

            if ACC > best_valid_acc:
                # save the model weights
                best_valid_acc = ACC
                torch.save( {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch_counter}, os.path.join(model_checkpoints_folder, '{}_fold_model.pth'.format(self.i)))
                valid_n_iter += 1

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()
            Predict_Label = []
            Predict_Scores = []
            True_Label = []
            valid_loss = 0.0
            counter = 0
            for (Protein,Mol,label,Protein_len, Mol_len) in tqdm(valid_loader):
                Protein = Protein.to(self.device)
                Mol = Mol.to(self.device)
                label = label.to(self.device)
                Protein_len = Protein_len.to(self.device)
                Mol_len = Mol_len.to(self.device)
                predict_label,predict_scores,loss = model(Mol ,Protein ,label,Mol_len,Protein_len)
                predict_label = predict_label.tolist()
                label = label.tolist()
                Predict_Label.extend(predict_label)
                Predict_Scores.extend(predict_scores)
                True_Label.extend(label)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
            ACC, precision, recall, f1, auc, mcc, sensitivity, specificity, AUPR= self.evaluate(True_Label, Predict_Label,Predict_Scores)
        model.train()
        return valid_loss,ACC, precision, recall, f1, auc, mcc, sensitivity, specificity, AUPR



def main():
    config = yaml.load(open("CellChem.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    from dataset.Dataset import MoleculeDatasetWrapper
    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset']).get_dataset()
    batch_size = 16
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    torch.manual_seed(2024)
    if torch.cuda.is_available():  
        torch.cuda.manual_seed_all(2024) 
    i = 0
    for train_index, val_index in kf.split(dataset):
        i = i+1
        print('*' * 25, 'No.', i , '-fold', '*' * 25)
        train_dataset = torch.utils.data.dataset.Subset(dataset, train_index)
        valid_dataset = torch.utils.data.dataset.Subset(dataset, val_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
        molclr = SmilesnCmap(i,train_loader, valid_loader, config)
        molclr.train()
        




if __name__ == "__main__":
    main()

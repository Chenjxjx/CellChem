from Model.CPI import *
from tqdm import tqdm
import yaml
import sklearn.metrics
import sklearn
import os

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
device = 'cuda'
encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
config_file = './config_mg.yaml'
config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
mol_encoder = GraphTransformer(**config["model"]).to('cuda:0')
decoder = Decoder(mol_encoder,atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,dropout, device)
model = Predictor(encoder, decoder, device).to(device)
def evaluate(true,pred_label,pred_score):
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
def _validate(model, valid_loader):
    # validation steps
    model.eval()
    with torch.no_grad():
        Predict_Label = []
        Predict_Scores = []
        True_Label = []
        valid_loss = 0.0
        counter = 0
        for bn, (Protein,Mol,label,Protein_len, Mol_len) in enumerate(valid_loader):
            Protein = Protein.to(device)
            Mol = Mol.to(device)
            label = label.to(device)
            Protein_len = Protein_len.to(device)
            Mol_len = Mol_len.to(device)
            predict_label,predict_scores,loss = model(Mol ,Protein ,label,Mol_len,Protein_len)
            predict_label = predict_label.tolist()
            label = label.tolist()
            Predict_Label.extend(predict_label)
            Predict_Scores.extend(predict_scores)
            True_Label.extend(label)
            valid_loss += loss.item()
            counter += 1
        valid_loss /= counter
        ACC, precision, recall, f1, auc, mcc, sensitivity, specificity, AUPR=evaluate(True_Label, Predict_Label,Predict_Scores)
    return valid_loss,ACC, precision, recall, f1, auc, mcc, sensitivity, specificity, AUPR,True_Label, Predict_Label,Predict_Scores
config = yaml.load(open("CellChem_test.yaml", "r"), Loader=yaml.FullLoader)  ##(eg:CellChem_test.yaml,CellChem_wo_test.yaml)
print(config)
from dataset.Dataset import MoleculeDatasetWrapper
dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset']).get_dataset()
from dataset.Dataset import *
test_loader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True)
A = []
P =[]
R =[]
F = []
AUC = []
MCC =[]
aupr = []
for i in range(1,6):
    print(i)
    path = './Model_CellChem_random_save/'+str(i)+'_fold_model.pth'  ####(eg:Model_CellChem_random_save,CellChem_wo_random_save,Model_CellChem_scaffold_save,CellChem_wo_scaffold_save)
    model.load_state_dict(torch.load(path)['net'])
    valid_loss,ACC, precision, recall, f1, auc, mcc, sensitivity, specificity, AUPR,True_Label,Predict_Label,Predict_Scores = _validate(model, test_loader)
    A.append(ACC)
    P.append(precision)
    R.append(recall)
    F.append(f1)
    AUC.append(auc)
    MCC.append(mcc)
    aupr.append(AUPR)
df = pd.DataFrame([A,P,R,F,AUC,aupr]).T
df.columns = ['accuracy','precision','recall','f1','auc','AUPR']
df.to_csv('./test_result/CellChem_test_random.csv')  #(eg:CellChem_test_random.csv,CellChem_wo_test_random.csv,CellChem_test_scaffold.csv,CellChem_wo_test_scaffold.csv)

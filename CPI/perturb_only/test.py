from Model.gin_autoencoder import SMILES_CMAP_CL
import yaml
import torch
from tqdm import tqdm
import sklearn.metrics
import sklearn
config = yaml.load(open("test.yaml", "r"), Loader=yaml.FullLoader)
device = config.get('gpu', 'cuda' if torch.cuda.is_available() else 'cpu')
model = SMILES_CMAP_CL(**config["model"]).to(device)
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
    with torch.no_grad():
        model.eval()
        Predict_Label = []
        Predict_Scores = []
        True_Label = []
        valid_loss = 0.0
        counter = 0
        for (CRISPR,Cmap,Protein,Mol,label) in tqdm(valid_loader):
            CRISPR = CRISPR.to(device)
            Cmap = Cmap.to(device)
            Protein = Protein.to(device)
            Mol = Mol.to(device)
            label = label.to(device)
            loss,predict_label,predicted_score= model(CRISPR,Cmap,Protein,Mol,label)
            predict_label = predict_label.tolist()
            label = label.tolist()
            Predict_Label.extend(predict_label)
            Predict_Scores.extend(predicted_score)
            True_Label.extend(label)
            valid_loss += loss.item()

            counter += 1
        valid_loss /= counter
        ACC, precision, recall, f1, auc, mcc, sensitivity, specificity, AUPR= evaluate(True_Label, Predict_Label,Predict_Scores)
    model.train()
    return valid_loss,ACC, precision, recall, f1, auc, mcc, sensitivity, specificity, AUPR,True_Label,Predict_Scores
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
T=[]
Pred=[]
for i in range(1,6):
    print(i)    
    path = './Model_save/'+str(i)+'_fold_model.pth'
    model.load_state_dict(torch.load(path))
    valid_loss,ACC, precision, recall, f1, auc, mcc, sensitivity, specificity, AUPR,True_Label,Predict_Scores = _validate(model, test_loader)
    A.append(ACC)
    P.append(precision)
    R.append(recall)
    F.append(f1)
    AUC.append(auc)
    MCC.append(mcc)
    aupr.append(AUPR)
    T.append(True_Label)
    Pred.append(Predict_Scores)
df = pd.DataFrame([A,P,R,F,AUC,aupr]).T
df.columns = ['accuracy','precision','recall','f1','auc','AUPR']
df.to_csv('./test_result/perturb.csv')

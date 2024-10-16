import torch 
import os
import soundfile as sf
import pandas  as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datasets import load_dataset, Audio, Dataset
from transformers import EncodecModel, AutoProcessor
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from audiocraft.modules.conditioners import ConditioningAttributes
from tqdm import tqdm


class SpeechCommandTransformer(torch.nn.Module):
    # initialize
    def __init__(self, feature_size, seq_length, num_classes, model_dim=256, nhead=4, num_layers=3, dropout=0.3):
        super(SpeechCommandTransformer, self).__init__()
        # Embedding layer
        self.embedding = torch.nn.Linear(feature_size, model_dim)
        # Positional encoding
        self.pos_encoder = torch.nn.Parameter(torch.randn(1, seq_length, model_dim))
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = torch.nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # Rearrange dimensions
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x += self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_layer(x)
        return x

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.gru = torch.nn.GRU(input_size=128, hidden_size=150, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = torch.nn.Linear(150, 50)  
        self.fc2 = torch.nn.Linear(50, 7)  

    def forward(self, x):
        x = x.transpose(1,2)
        _, h_n = self.gru(x)  
        x = self.fc1(h_n[-1])  
        x = torch.nn.functional.relu(x) 
        x = self.fc2(x)  
        return x


model_type = "conv_SC" # "conv_SC", "trans_SC", "gru_TESS", "conv_TESS"
dataset_name = "SC" # "SC" or "TESS"

if dataset_name=="SC":
    X_test_org = torch.load('valid_data_embed_SC_deq_x_audiogen.pt')
    y_test_org = torch.load('valid_data_embed_SC_deq_y_audiogen.pt')
elif dataset_name=="TESS":
    X_test_org = torch.load('audiogen_tess_X_val_embed_deq.pt').float()
    y_test_org = torch.load('audiogen_tess_y_val_embed_deq.pt').long()
X_test_org = X_test_org.to(device="cuda")
y_test_org = y_test_org.to(device="cuda")

if model_type == "conv_SC":
    model = torch.nn.Sequential( 
        torch.nn.Conv1d(128, 64, 5, stride=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(64, 32, 3, stride=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(32, 16, 3, stride=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(16, 16, 3, stride=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features = 640, out_features = 200), 
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_features = 200, out_features = 50), 
        torch.nn.Softmax(dim=1) 
    )
    model.load_state_dict(torch.load("models/SC_AudioXLM_conv"))

elif model_type == "trans_SC":
    model = SpeechCommandTransformer(feature_size=128, seq_length=50, num_classes=35)
    model.load_state_dict(torch.load("models/SC_AudioXLM_transformer.pth"))

elif model_type == "conv_TESS":
    model = torch.nn.Sequential( 
    torch.nn.Conv1d(128, 64, 5, stride=1),
    torch.nn.ReLU(),
    torch.nn.Conv1d(64, 32, 3, stride=1),
    torch.nn.ReLU(),
    torch.nn.Conv1d(32, 16, 3, stride=1),
    torch.nn.ReLU(),
    torch.nn.Conv1d(16, 16, 3, stride=1),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(in_features = 1440, out_features = 200), 
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(in_features = 200, out_features = 50), 
    torch.nn.Softmax(dim=1) 
    )
    model.load_state_dict(torch.load("models/TESS_AudioXLM_conv.pth"))

elif model_type == "gru_TESS":
    model = RNN()
    model.load_state_dict(torch.load("models/TESS_AudioXLM_gru.pth"))

model = model.to(device="cuda")



n_sample = 500 #number of random samples to run validation
shuffled_list = np.arange(len(X_test_org))
np.random.seed(0)
np.random.shuffle(shuffled_list)
shuffle_idxs = shuffled_list[:n_sample]
X_test = X_test_org[shuffle_idxs]
y_test = y_test_org[shuffle_idxs]

 
model.eval()
with torch.no_grad(): 
    y_pred = model(X_test_org) 
    _, predicted = torch.max(y_pred, dim=1) 
    accuracy = (predicted == y_test_org).float().mean() 
    print(f'Test Accuracy whole: {accuracy.item():.4f}')

with torch.no_grad(): 
    y_pred = model(X_test) 
    _, predicted = torch.max(y_pred, dim=1) 
    accuracy = (predicted == y_test).float().mean() 
    print(f'Test Accuracy sample: {accuracy.item():.4f}')

y_test_m = predicted.clone()
print(y_test[:10])
print(predicted[:10])

if dataset_name == "SC":
    audio_dur = 1
    n_feats_max = 6400
    codes_len = 200
    codes_dim = 50
elif dataset_name == "TESS":
    audio_dur = 2
    n_feats_max = 12800
    codes_len = 400
    codes_dim = 100

uselm = 1 #to enable ALM usage
onlyimp = True
descriptions = [None] #for audiogen text condition
i = 0
n_steps = 50
method = "featatt" #random, featatt: feature attribution (IG) is ours

# AudioGen loading
model_ag = AudioGen.get_pretrained('facebook/audiogen-medium')
model_ag.set_generation_params(duration=audio_dur)
encodec_model = model_ag.compression_model

# feature attribution method
integrated_gradients = IntegratedGradients(model) # feature attribution method


accs_all = []
for n_feats in [640, 1280, 2560, 3840, 5120, 5760, 6400]:#[9600, 8600, 7600, 5600, 3600, 1600, 0]: #[0, 1600, 3600, 5600, 7600, 8600, 9600]
    
    if dataset_name=="TESS":
        n_feats = n_feats * 2
    accs_feat = [n_feats]

    for seed in range(5):

        X_test = X_test_org.clone()[shuffle_idxs]
        y_test = y_test_org.clone()[shuffle_idxs]

        rand_list = np.arange(n_feats_max)
        np.random.shuffle(rand_list)
        rand_idxs = rand_list[:n_feats]
        rand_idxs_row = rand_idxs//codes_dim
        rand_idxs_col = rand_idxs%codes_dim

        for i in tqdm(range(n_sample)):
            with torch.no_grad(): 
                sample_embed = torch.unsqueeze(X_test[i].clone(), 0)

                xid = y_test_m[i] 
                attributions_ig = integrated_gradients.attribute(sample_embed, target=xid, n_steps=n_steps)
                attributions_ig = attributions_ig[0].detach().cpu().numpy()

                if method == "featatt":
                    w_i = np.unravel_index(np.argsort(attributions_ig, axis=None), attributions_ig.shape)
                    w_i = (w_i[0][:n_feats], w_i[1][:n_feats])
                elif method == "random":
                    w_i = (rand_idxs_row, rand_idxs_col)

                deq_embed_org = sample_embed.clone()
                codes = encodec_model.quantizer.encode(sample_embed)
                sample_embed[0][w_i] = 0  #removing the least important n_feats elements, so keeping most important 9600-n_feats
                
                if uselm == 0:
                    X_test[i] = sample_embed[0]

                elif uselm == 1:
                    codes_onlyimp = encodec_model.quantizer.encode(sample_embed) #128->4 (deq->quant)
                    n_code_feats = int(codes_len * (n_feats/n_feats_max))
                    dists = torch.abs(codes_onlyimp-codes) * -1 # to find max changed codes and make them -1
                    dists = dists[0].cpu()
                    code_i = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
                    code_i = (code_i[0][:n_code_feats], code_i[1][:n_code_feats])
                    codes_onlyimp = codes.clone()
                    codes_onlyimp[0][code_i] = -1 #unknown token for AudioGen
                    codes_onlyimp_0 = codes_onlyimp.clone()
                    codes_onlyimp_0[0][code_i] = 0
                    
                    attributes = [ConditioningAttributes(text={'description': description}) for description in descriptions]

                    #Complete the explanation with AuidoGen ALM by using the modified version
                    tokens = model_ag._generate_tokens_AudioXLM(attributes, None, progress=False, gen_mode= "audioXLM", prompt_tokens_onlyimp=codes_onlyimp)

                    deq_embed = encodec_model.quantizer.decode(tokens) #deq_embed.shape:  torch.Size([1, 128, 150])
                    deq_embed_org[0][w_i] = deq_embed[0][w_i] #getting the original values from deq version to prevent enc/dec info loss
                    X_test[i] = deq_embed_org
        
        with torch.no_grad(): 
            y_pred = model(X_test) 
            _, predicted = torch.max(y_pred, dim=1) 
            accuracy = (predicted == y_test_m).float().mean() 
            print(f'Test Accuracy after important feature deletion: {accuracy.item():.4f}')

        accs_feat.append(accuracy.item())
    accs_feat = np.array(accs_feat)
    mean_acc = np.mean(accs_feat[1:])
    std_acc = np.std(accs_feat[1:])
    accs_feat = np.append(accs_feat, [mean_acc])
    accs_feat = np.append(accs_feat, [std_acc])
    accs_all.append(accs_feat)
accs_all = np.array(accs_all)
print(accs_all)

np.savetxt("path.csv", accs_all, delimiter=",", fmt='%1.4f')

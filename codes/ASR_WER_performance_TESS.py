import torch 
import torchaudio
from torchaudio.utils import download_asset
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
from torcheval.metrics.functional import word_error_rate
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

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank
    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


model_type = "gru_TESS" # "gru_TESS", "conv_TESS"

X_test_org = torch.load('valid_data_embed_SC_deq_x_audiogen.pt')
y_test_org = torch.load('valid_data_embed_SC_deq_y_audiogen.pt')
X_test_org = X_test_org.to(device="cuda")
y_test_org = y_test_org.to(device="cuda")

if model_type == "conv_TESS":
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

X_test = X_test_org.clone()
y_test = y_test_org.clone()

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



model_asr = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
labels_asr=bundle.get_labels()
labels_asr = list(labels_asr)
labels_asr[1] = " "
decoder = GreedyCTCDecoder(labels_asr)

#create original ASR scripts with full features
target_asr = []
for i in tqdm(range(len(X_test))):
    sample_embed = torch.unsqueeze(X_test[i].clone(), 0)
    codes = encodec_model.quantizer.encode(sample_embed)
    gen_audio = model_ag.compression_model.decode(codes, None)
    emission, _ = model_asr(gen_audio[0].cpu())
    transcript = decoder(emission[0])
    target_asr.append(transcript)
print("target_asr[0:5]: ", target_asr[0:5])
print("len(target_asr): ", len(target_asr))



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
for n_feats in [0, 640, 1280, 2560, 3840, 5120, 5760, 6400]:#[9600, 8600, 7600, 5600, 3600, 1600, 0]: #[0, 1600, 3600, 5600, 7600, 8600, 9600]
    n_feats = n_feats * 2
    accs_feat = [n_feats]
    n_feats_rem = n_feats_max - n_feats

    for seed in range(5):
        pred_asr = []
        X_test = X_test_org.clone()
        y_test = y_test_org.clone()

        rand_list = np.arange(n_feats_max)
        np.random.shuffle(rand_list)
        rand_idxs = rand_list[:n_feats]
        rand_idxs_row = rand_idxs//codes_dim
        rand_idxs_col = rand_idxs%codes_dim

        model.train()
        for i in tqdm(range(len(X_test))):
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
                sample_embed[0][w_i] = 0 


                if uselm == 0:
                    X_test[i] = sample_embed[0]
                    codes_onlyimp = encodec_model.quantizer.encode(sample_embed)

                    n_code_feats = int(codes_len * (n_feats/n_feats_max))
                    dists = torch.abs(codes_onlyimp-codes) * -1 # to find max changed codes and make them -1
                    dists = dists[0].cpu()
                    code_i = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
                    code_i = (code_i[0][:n_code_feats], code_i[1][:n_code_feats])
                    codes_onlyimp_0 = codes.clone()
                    codes_onlyimp_0[0][code_i] = 0
                    tokens = codes_onlyimp_0

                elif uselm == 1:
                    codes_onlyimp = encodec_model.quantizer.encode(sample_embed) #128->4 (deq->quant)


                    n_code_feats = int(codes_len * (n_feats/n_feats_max))
                    dists = torch.abs(codes_onlyimp-codes) * -1 # to find max changed codes and make them -1
                    dists = dists[0].cpu()
                    code_i = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
                    code_i = (code_i[0][:n_code_feats], code_i[1][:n_code_feats])
                    codes_onlyimp = codes.clone()
                    codes_onlyimp[0][code_i] = -1
                    codes_onlyimp_0 = codes_onlyimp.clone()
                    codes_onlyimp_0[0][code_i] = 0
                    
                    attributes = [ConditioningAttributes(text={'description': description}) for description in descriptions]
                    tokens = model_ag._generate_tokens_AudioXLM(attributes, None, progress=False, gen_mode= "audioXLM", prompt_tokens_onlyimp=codes_onlyimp)
                        
                    deq_embed = encodec_model.quantizer.decode(tokens) 
                    deq_embed_org[0][w_i] = deq_embed[0][w_i] #getting the original values from deq version to prevent enc/dec info loss
                    X_test[i] = deq_embed_org

                    
                gen_audio = model_ag.compression_model.decode(tokens, None)
                emission, _ = model_asr(gen_audio[0].cpu())
                decoder = GreedyCTCDecoder(labels=labels_asr)
                transcript = decoder(emission[0])
                pred_asr.append(transcript)

        
        accuracy_wer = word_error_rate(pred_asr, target_asr)
        print(f'Test Accuracy after important feature deletion: {accuracy_wer.item():.4f}')

        accs_feat.append(accuracy_wer.item())
    accs_feat = np.array(accs_feat)
    mean_acc = np.mean(accs_feat[1:])
    std_acc = np.std(accs_feat[1:])
    accs_feat = np.append(accs_feat, [mean_acc])
    accs_feat = np.append(accs_feat, [std_acc])
    accs_all.append(accs_feat)
accs_all = np.array(accs_all)
print(accs_all)

np.savetxt("path.csv", accs_all, delimiter=",", fmt='%1.4f')
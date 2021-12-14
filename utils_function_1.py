import pandas as pd
import numpy as np
import os
import re
import random
import torch
import torchaudio
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import itertools
import datetime


from IPython.display import Audio
from rich.progress import track
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def calculate_MFCC(data, n_mels):
    '''
    This function load the audio files from the server and calculates the MFCC
    
    inputs:
    - data: dataframe containing the path of the file and the sex of the speaker for that file
    - n_mels: # of mel-bands
    
    output:
    - dataframe with path, sex and the calculated mfcc (tensor) for each audio file
    ''' 
    
    melkwargs={'n_fft': 1024,
              'hop_length': 512, 
              'n_mels': n_mels}
    mfcc = []
    for i, audio_path in track(enumerate(data.path), total=data.shape[0]):
        audio_tensor, sr = torchaudio.load(audio_path)
        audio_mfcc = (torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mels, log_mels=True, melkwargs=melkwargs)(audio_tensor[0, :].t()))
        mfcc.append(audio_mfcc)
    return pd.DataFrame({'path': data.path, 'SEX': data.SEX, 'mfcc': mfcc})

###########################################################################

def calculate_band_metrics(data, n_mels):
    '''
    This function load the audio files from the server, calculates the MFCC and computes the mean over the time dimension
    for each mel-band
    
    inputs:
    - data: dataframe containing the path of the file and the sex of the speaker for that file
    - n_mels: # of mel-bands
    
    output:
    - dataframe where each row represents an audio sample and the columns represent the mel-band.
      moreover, the last two columns are represented by the path and the sex of the speaker
    ''' 
    melkwargs={'n_fft': 1024,
              'hop_length': 512, 
              'n_mels': n_mels}
    
    summary_band = pd.DataFrame()
    for i, audio_path in track(enumerate(data.path), total=data.shape[0]):
        audio_tensor, sr = torchaudio.load(audio_path)
        audio_mfcc = (torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mels, log_mels=True, melkwargs=melkwargs)(audio_tensor[0, :].t()))
        sample_mean = torch.mean(audio_mfcc, 1)
        summary_band=pd.concat([summary_band, pd.DataFrame(sample_mean.numpy())], axis = 1, ignore_index=True)
    result = summary_band.transpose()
    result['path'] = data.path.values

    result['SEX'] = data.SEX.values
    return result

#############################################################################

def split_train_validation_test(data, perc_train, perc_validation):
    
    """
    This function splits the data into train, validation and test.
    Moreover this function pays attention to do not put the same speaker in both train and test  
    """
    
    df_male = data[data.SEX == 'M'].sort_values('ID', ignore_index=True, ascending=False)
    df_female = data[data.SEX == 'F'].sort_values('ID', ignore_index=True, ascending=False)

    #divide train test:
    speakers_train_male = df_male.loc[:df_male.shape[0]*perc_train, 'ID'].unique()
    speakers_train_female = df_female.loc[:df_female.shape[0]*perc_train, 'ID'].unique()
    
    df_train_male = df_male[df_male.ID.isin(speakers_train_male)]
    df_train_female = df_female[df_female.ID.isin(speakers_train_female)]
    
    df_test_male = df_male[~df_male.ID.isin(speakers_train_male)]
    df_test_female = df_female[~df_female.ID.isin(speakers_train_female)]
    
    df_train = pd.concat([df_train_male, df_train_female], axis=0, ignore_index=True)
    df_test = pd.concat([df_test_male, df_test_female], axis=0, ignore_index=True)
    
    # in the validation is not a problem if we have some speaker of the train, thus we can use the sklearn.model_selection.train_test_split
    # to extract from the train the validation dataset
    df_train_final, df_validation = train_test_split(df_train, test_size=perc_validation, stratify=df_train.SEX)
    
    return df_train_final, df_validation, df_test

#############################################################################

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, file_name=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    n_samples = np.sum(cm)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 8))
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    thresh = 0.8 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout(pad=2.0)
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nnumber of samples={:0.4f}\naccuracy={:0.4f}; misclass={:0.4f}'
                                            .format(n_samples, accuracy, misclass))
    plt.show()
    plt.close()

#############################################################################

def scale_embedding(data_to_scale, data_to_fit):
    '''
    this function fits the standard scaler with data_to_fit and transforms data_to_scale
    
    returns a dataframe with the scaled embeddings and the information regarding the path of the audio file
    and the sex of the speaker
    '''
    scaler = StandardScaler()
    print(scaler.fit(data_to_fit.iloc[:, :64]))
    df_mfcc_scaled_metrics = pd.DataFrame(scaler.transform(data_to_scale.iloc[:, :64]))
    df_mfcc_scaled_metrics['path'] = data_to_scale.path.values
    df_mfcc_scaled_metrics['SEX'] = data_to_scale.SEX.values
    
    return df_mfcc_scaled_metrics

#############################################################################

def preliminary_plot(sample):
    
    """
    This function plots 3 different charts for an input sample:
    - Raw signal (waveform)
    - Frequency-Magnitude diagram (until Nyquist frequency)
    - Spectrogram
    """
    
    audio_tensor, sr = torchaudio.load(sample.path)
    
    # plot Frequency-Magnitude until the Nynquist frequency 
    mag = torch.abs(torch.fft.fft(audio_tensor[0,:]))
    x_axis = np.linspace(0, sr/2, len(audio_tensor[0,:])//2)
    y_axis = mag[mag.shape[0]//2]

    fig, axs = plt.subplots(1,3, figsize=(20,5), dpi=100)
    axs[0].set_title('Raw signal '+ sample.SEX)
    axs[0].plot(audio_tensor.numpy().transpose())
    axs[0].set_xlabel("nr. sample")
    axs[0].set_ylabel("Amplitude")
    
    axs[1].plot(x_axis, mag[:int(mag.shape[0]/2)])
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Magnitude')
    axs[1].set_title(torch.mean(mag[:int(mag.shape[0]/2)]))
    
    spectrogram = torchaudio.transforms.Spectrogram(n_fft = 1024, hop_length = 512)(audio_tensor[0,:]).numpy()
    im = axs[2].imshow(spectrogram, cmap = "magma", aspect = "auto")
    axs[2].set_title("Spectrogram "+ sample.SEX + ' ' + str(np.mean(spectrogram)))
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Frequency")
    fig.colorbar(im)

#############################################################################

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
#############################################################################

def calculate_mfcc_segment(data, n_mels, len_sample):
    '''
    This function load the audio files from the server, calculates the MFCC and splits it into chuncks of len_sample duration
    
    inputs:
    - data: dataframe containing the path of the file and the sex of the speaker for that file
    - n_mels: # of mel-bands
    - len_sample: len of each chuncks
    
    output:
    - dataframe composed by 3 columns: mfcc, path and sex of the speaker
    ''' 
    melkwargs={'n_fft': 1024,
              'hop_length': 512, 
              'n_mels': n_mels}
    frames_per_sample = int(len_sample * 16000/512 + 1)
    audio_segmented_mfcc = []
    label = []
    id_audio = []
    for i, audio_path in track(enumerate(data.path), total=data.shape[0]):
        audio_tensor, sr = torchaudio.load(audio_path)
        audio_mfcc = (torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mels, log_mels=True, melkwargs=melkwargs)(audio_tensor[0, :].t()))
        for start_frame in range(0, audio_mfcc.shape[1]-frames_per_sample+1, frames_per_sample):
            id_audio.append(data.path[i])
            label.append(data.SEX[i])
            audio_segmented_mfcc.append(audio_mfcc[ :, start_frame:start_frame+frames_per_sample])
    result = pd.DataFrame({'mfcc': audio_segmented_mfcc, 'SEX': label, 'path': id_audio})
    
    return result

#############################################################################

def training_validation_linear(model, df_train_emb, df_validation_emb, batch_size, epochs, name_dir, tensorboard_report):
    
    x_train = df_train_emb[list(range(0,64))].values
    y_train = df_train_emb.SEX
    x_validation = df_validation_emb[list(range(0,64))].values
    y_validation = df_validation_emb.SEX
    
    # we need to convert the target label into [0,1] 
    # 1 = M
    # 0 = F
    y_train_norm = (y_train == 'M').astype(int)
    y_validation_norm = (y_validation == 'M').astype(int)

    train_data = CustomDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train_norm))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    validation_data = CustomDataset(torch.FloatTensor(x_validation), torch.FloatTensor(y_validation_norm))
    validation_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=1, shuffle=False)
    
    
    time_now = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")

    train_val_log_dir = 'runs/'+ name_dir+'/batch_size_'+ str(batch_size)+ 'n_layer_' +str(len(list(model.modules()))-1)+\
    'learning_rate'+ str(optimizer.param_groups[0]["lr"]) +  "_" + time_now  
    writer = SummaryWriter(train_val_log_dir)
    
    
    for epoch in track(range(epochs), total=epochs):
        
        # TRAIN:
        epoch_loss_train = []
        model = model.train()
        for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(x_batch)
                #print(y_pred)
                loss = criterion((y_pred), y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()

                epoch_loss_train.append(loss.item())
                # print("Epoch : {}/{}, loss_train = {:.6f}".format(epoch, epochs, epoch_loss/len(train_loader)), end="\r")
        
        # VALIDATION
        y_pred_list = []
        epoch_loss_validation = []
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in validation_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_validation_pred = model(x_batch)
                loss = criterion((y_validation_pred), y_batch.unsqueeze(1))
                epoch_loss_validation.append(loss.item())
                
                y_pred_tag = torch.round(y_validation_pred)
                y_pred_list.append(y_pred_tag.cpu().numpy())
            
            # TENSORBOARD
            if (tensorboard_report == True):
                writer.add_scalars('Loss/train and validation/batch size = '+str(batch_size),\
                                   {"train_mean": np.mean(epoch_loss_train),\
                                    "validation_mean": np.mean(epoch_loss_validation)}, epoch)

    return model

#############################################################################

def testing_linear(df_test_emb, model):
    
    x_test = df_test_emb[list(range(0,64))].values
    y_test = df_test_emb.SEX
    y_test_norm = (y_test == 'M').astype(int)
    
    test_data = CustomDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test_norm))
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            y_test_pred = model(x_batch)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    
    return pd.DataFrame({'y_true': y_test_norm, 'y_pred': y_pred_list})

#############################################################################

def training_validation_CNN(model, df_train_mfcc, df_validation_mfcc, batch_size, epochs, name_dir, tensorboard_report):
    
    x_train = df_train_mfcc.mfcc
    y_train = df_train_mfcc.SEX
    x_validation = df_validation_mfcc.mfcc
    y_validation = df_validation_mfcc.SEX
    
    # we need to convert the target label into [0,1] 
    # 1 = M
    # 0 = F
    y_train_norm = (y_train == 'M').astype(int)
    y_validation_norm = (y_validation == 'M').astype(int)

    train_data = CustomDataset(x_train, torch.FloatTensor(y_train_norm))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    validation_data = CustomDataset(x_validation, torch.FloatTensor(y_validation_norm))
    validation_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=1, shuffle=False)
    
    
    time_now = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")

    train_val_log_dir = 'runs/'+ name_dir+'/batch_size_'+ str(batch_size)+ 'n_layer_' +str(len(list(model.modules()))-1)+\
    'learning_rate'+ str(optimizer.param_groups[0]["lr"]) +  "_" + time_now  
    writer = SummaryWriter(train_val_log_dir)
    
    
    for epoch in track(range(epochs), total=epochs):
        
        # TRAIN:
        epoch_loss_train = []
        model = model.train()
        for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(torch.unsqueeze(x_batch, 1))
                loss = criterion((y_pred), y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()

                epoch_loss_train.append(loss.item())
                # print("Epoch : {}/{}, loss_train = {:.6f}".format(epoch, epochs, epoch_loss/len(train_loader)), end="\r")
        
        # VALIDATION
        y_pred_list = []
        epoch_loss_validation = []
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in validation_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_validation_pred = model(torch.unsqueeze(x_batch, 1))
                loss = criterion((y_validation_pred), y_batch.unsqueeze(1))
                epoch_loss_validation.append(loss.item())
                
                y_pred_tag = torch.round(y_validation_pred)
                y_pred_list.append(y_pred_tag.cpu().numpy())
            
            # TENSORBOARD
            if (tensorboard_report == True):
                writer.add_scalars('Loss/train and validation/batch size = '+str(batch_size),\
                                   {"train_mean": np.mean(epoch_loss_train),\
                                    "validation_mean": np.mean(epoch_loss_validation)}, epoch)

    return model

#############################################################################

def testing_CNN(df_test_mfcc, model):
    
    x_test = df_test_mfcc.mfcc
    y_test = df_test_mfcc.SEX
    y_test_norm = (y_test == 'M').astype(int)
    
    test_data = CustomDataset(x_test, torch.FloatTensor(y_test_norm))
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            y_test_pred = model(torch.unsqueeze(x_batch, 1))
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    
    return pd.DataFrame({'y_true': y_test_norm, 'y_pred': y_pred_list, 'path': df_test_mfcc.path})

#############################################################################


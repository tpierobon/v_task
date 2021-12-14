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
import fairseq


from IPython.display import Audio
from rich.progress import track
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


def calculate_raw_signal_segment(data, len_sample):
    '''
    This function load the audio files from the server and divides it into different audio segments with duration = len_sample
    
    inputs:
    - data: dataframe containing the path of the files, the sex of the speaker and the ID of the speaker
    
    output:
    - dataframe where each row represents an audio sample and the columns are the following:
        - raw_signal: tensor 
        - ID: int
        - SEX: str
        - path: str
    
    NOTE: if the duration of the original audio track is not divisible by len_sample, then the function will exclude the last audio chunk. 
    We can accept this also because we have noticed that most of the audio files are characterized by a pause at the beginning and at the end 
    (the speaker doesn't start immediately to talk)
    ''' 
    
    frames_per_segment = int(len_sample * 16000)
    audio_segmented = []
    label = []
    path_audio = []
    id_audio = []
    for i, audio_path in track(enumerate(data.path), total=data.shape[0]):
        audio_tensor, sr = torchaudio.load(audio_path)
        for start_frame in range(0, audio_tensor.shape[1]-frames_per_segment+1, frames_per_segment):
            path_audio.append(data.path[i])
            id_audio.append(data.ID[i])
            label.append(data.SEX[i])
            audio_segmented.append(audio_tensor[ :, start_frame:start_frame+frames_per_segment])
    result = pd.DataFrame({'raw_signal': audio_segmented, 'ID': id_audio, 'SEX': label, 'path': path_audio})
    
    return result

########################################################################

def extract_w2v_features_reshape(model, device, data):
    
    """
    this function applies the w2v2.0 to the input audio, extracting the features. After this, the function reshapes the audio tensor
    from a shape (m,n) to a shape (m*n)
    
    input:
    - w2v model
    - device in which the model will be run
    - dataframe with the audio in a tensor format, the path of the audio, the sex and the ID of the speaker
    
    output: Dataframe in which the first 512 columns represent the different w2v bands and each row represent an audio segment. 
    Moreover, in the last 3 columns the path, the Sex and the ID of the audio segment are reported 
    """

    model.eval()
    list_features = []
    for i, audio in track(enumerate(data.raw_signal), total=data.shape[0]):
        z = model.feature_extractor(audio.to(device))
        c = model.feature_aggregator(z)
        list_features.append(torch.squeeze(c.detach()).cpu().view(-1).numpy())
        
    df_result = pd.DataFrame(np.vstack(list_features))
    df_result['path'] = data.path.values
    df_result['SEX'] = data.SEX.values
    df_result['ID'] = data.ID.values
    
    return df_result

########################################################################

def extract_w2v_mean(model, device, data):
    
    """
    this function applies the w2v2.0 to the input audio, extracting the features. After this, the function cumputes the mean of each w2v band
    
    input:
    - w2v model
    - device in which the model will be run
    - dataframe with the audio in a tensor format, the path of the audio, the sex and the ID of the speaker
    
    output: Dataframe in which the first 512 columns represent the different w2v bands and each row represent an audio segment. 
    Moreover, in the last 3 columns the path, the Sex and the ID of the audio segment are reported
    """

    model.eval()
    list_features = []
    for i, audio in track(enumerate(data.raw_signal), total=data.shape[0]):
        z = model.feature_extractor(audio.to(device))
        c = model.feature_aggregator(z)
        list_features.append(torch.mean(torch.squeeze(c.detach()), axis=1).cpu().numpy())
        #print(torch.mean(torch.squeeze(c.detach()), axis=1).cpu().numpy().shape)
        #break
    df_result = pd.DataFrame(np.vstack(list_features))
    df_result['path'] = data.path.values
    df_result['SEX'] = data.SEX.values
    df_result['ID'] = data.ID.values
    
    
    return df_result

########################################################################

def scale_embedding(data_to_scale, data_to_fit):
    '''
    This function fits the standard scaler with data_to_fit and transforms data_to_scale
    
    returns a dataframe with the scaled embeddings and the information regarding the path of the audio file
    and the sex of the speaker
    '''
    scaler = StandardScaler()
    print(scaler.fit(data_to_fit.iloc[:, :512]))
    df_mfcc_scaled_metrics = pd.DataFrame(scaler.transform(data_to_scale.iloc[:, :512]))
    df_mfcc_scaled_metrics['path'] = data_to_scale.path.values
    df_mfcc_scaled_metrics['ID'] = data_to_scale.ID.values
    
    return df_mfcc_scaled_metrics

########################################################################

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

########################################################################

def training_validation(model, df_train_emb, df_validation_emb, batch_size, epochs, name_dir, tensorboard_report, class2idx):
    
    x_train = df_train_emb[list(range(0,512))].values
    y_train = df_train_emb.ID
    x_validation = df_validation_emb[list(range(0,512))].values
    y_validation = df_validation_emb.ID
    
    # Encode The Output Variable
   
    encoded_y_train = y_train.replace(class2idx).to_numpy()
    encoded_y_validation = y_validation.replace(class2idx).to_numpy()

    train_data = CustomDataset(torch.from_numpy(x_train).float(), torch.from_numpy(encoded_y_train).long())
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    validation_data = CustomDataset(torch.from_numpy(x_validation).float(), torch.from_numpy(encoded_y_validation).long())
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
                
                loss = criterion((y_pred), y_batch)
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
                loss = criterion((y_validation_pred), y_batch)
                epoch_loss_validation.append(loss.item())
                
                y_pred_softmax = torch.softmax(y_validation_pred, dim = 1)
                
                _, y_pred_tag = torch.max(y_pred_softmax, dim = 1) 
                y_pred_list.append(y_pred_tag.cpu().numpy())
            
            # TENSORBOARD
            if (tensorboard_report == True):
                writer.add_scalars('Loss/train and validation/batch size = '+str(batch_size),\
                                   {"train_mean": np.mean(epoch_loss_train),\
                                    "validation_mean": np.mean(epoch_loss_validation)}, epoch)

    return model

########################################################################

def testing(df_test_emb, model, class2idx):
    
    x_test = df_test_emb[list(range(0,512))].values
    y_test = df_test_emb.ID
    encoded_y_test = y_test.replace(class2idx).to_numpy()
    test_data = CustomDataset(torch.from_numpy(x_test).float(), torch.from_numpy(encoded_y_test).long())
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            y_test_pred = model(x_batch)
            y_pred_softmax = torch.softmax(y_test_pred, dim = 1)
            _, y_pred_tag = torch.max(y_pred_softmax, dim = 1) 
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    
    return pd.DataFrame({'y_true': encoded_y_test, 'y_pred': y_pred_list})

########################################################################

def training_validation_gru(model, df_train_mfcc, df_validation_mfcc, batch_size, epochs, name_dir, tensorboard_report):
    
    x_train = df_train_mfcc.w2v_features
    y_train = df_train_mfcc.SEX
    x_validation = df_validation_mfcc.w2v_features
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
        for x_batch, _ in train_loader:
                x_batch = x_batch.to(device)
                optimizer.zero_grad()
                x_pred, hn = model(x_batch)
                #print(y_pred)
                loss = criterion(x_pred, x_batch)
                loss.backward()
                optimizer.step()

                epoch_loss_train.append(loss.item())
                # print("Epoch : {}/{}, loss_train = {:.6f}".format(epoch, epochs, epoch_loss/len(train_loader)), end="\r")
        
        # VALIDATION
        compressed = []
        epoch_loss_validation = []
        model.eval()
        with torch.no_grad():
            for x_batch, _ in validation_loader:
                x_batch = x_batch.to(device)
                x_validation_pred, last_state = model(x_batch)
                loss = criterion((x_validation_pred), x_batch)
                epoch_loss_validation.append(loss.item())
                compressed.append(torch.squeeze(last_state).t().cpu())
            
            # TENSORBOARD
            if (tensorboard_report == True):
                writer.add_scalars('Loss/train and validation/batch size = '+str(batch_size),\
                                   {"train_mean": np.mean(epoch_loss_train),\
                                    "validation_mean": np.mean(epoch_loss_validation)}, epoch)

    return model

########################################################################

def calculate_embedding_gru(data, model, criterion, device):
    
    x_validation = data.w2v_features
    y_validation = data.SEX
    
    # we need to convert the target label into [0,1] 
    # 1 = M
    # 0 = F
    y_validation_norm = (y_validation == 'M').astype(int)

    validation_data = CustomDataset(x_validation, torch.FloatTensor(y_validation_norm))
    validation_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=1, shuffle=False)
    
    # VALIDATION
    compressed = []
    epoch_loss_validation = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in track(validation_loader, total=len(validation_loader)):
            x_batch = x_batch.to(device)
            x_validation_pred, last_state = model(x_batch)
            loss = criterion((x_validation_pred), x_batch)
            epoch_loss_validation.append(loss.item())
            compressed.append(torch.squeeze(last_state).t().cpu())
        result = pd.DataFrame(np.vstack(compressed))
        result['SEX'] = data.SEX.values
        result['path'] = data.path.values


    return result

########################################################################

def extract_w2v_features_gru(model, device, data):

    model.eval()
    list_features = []
    df_result= pd.DataFrame()
    for i, audio in track(enumerate(data.raw_signal), total=data.shape[0]):
        z = model.feature_extractor(audio.to(device))
        c = model.feature_aggregator(z)
        list_features.append(torch.squeeze(c.detach()).cpu().numpy().transpose())
        
    df_result['w2v_features'] = list_features
    df_result['path'] = data.path.values
    df_result['SEX'] = data.SEX.values
    
    return df_result

########################################################################


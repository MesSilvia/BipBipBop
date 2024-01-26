import torch  #used to create tensors to store values
import torch.nn as nn# used to make weights and biases part of the neural network
import torch.nn.functional as F # activatiom functions
from torch.optim import SGD # stochastic gradient descent
import lightning as L # Training easier to code
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import TensorDataset, DataLoader #for large datasets !
from torch.optim import Adam
#import pytorch_lightning as pl
import pdb
#import matplotlib.pyplot as plt
import math
#For importing midi files

import pretty_midi
import utils
import tensorboard



DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE_S = "cuda" if torch.cuda.is_available() else "cpu"


class model_test(L.LightningModule): 
    def __init__(self, lstm_hu): 
        super().__init__() 
        #mother class constructor
        self.save_hyperparameters()
        #Saving hyperparameters to checkpoint 
        self.lstm_hu = lstm_hu
        self.lstm_pitch = nn.LSTM(input_size=3, hidden_size = lstm_hu, batch_first = True)
       

        self.linear_pitch = nn.Linear(in_features = lstm_hu, out_features = 128)
        self.linear_duration = nn.Linear(in_features = lstm_hu, out_features = 1)
        self.linear_step = nn.Linear(in_features = lstm_hu, out_features = 1)

        self.hp = None
        self.cp = None

        #130 Outputs: 128 for sparse Crossentropy for pitch prediction, the remaining 2 for duration and step. 
        #They will have obviously different losses ! 
        #Expected input : L x N x H
    

    def configure_optimizers(self): 
        #Describes the method we want to use to optimize neural network 
        return Adam(self.parameters(), lr = 0.0005)

    def initialize_hidden_state(self, batch_size): 
        self.hp = torch.zeros(1, batch_size, self.lstm_hu, device=DEVICE)
        self.cp = torch.zeros(1, batch_size, self.lstm_hu, device=DEVICE)
         
    def forward(self, input): 
        #Builds the neural network structure: describes what happens to the input while passing though all the layers
        #Visto che batch first True, ci si aspetta l'input come batch_sizeXsequence_lengthXfeatures_length
        #Se batch_first è falso, allora sequence_lengthXbatch_sizeXfeatures_length

        
        if self.hp is None or self.cp is None : 
            self.initialize_hidden_state(input.shape[0])
        
        input_pitch_lstm = input


        #Forwarding through lstm 
        output_pitch_lstm, (self.hp, self.cp) = self.lstm_pitch(input_pitch_lstm, ((self.hp).detach(), (self.cp).detach()))

        #Forwarding through linear layers
        out_pitch = self.linear_pitch(output_pitch_lstm[:,-1,:])
        out_duration = self.linear_duration(output_pitch_lstm[:,-1,:])
        out_step = self.linear_step(output_pitch_lstm[:,-1, :])


    
   
        

        out = torch.cat((out_pitch, out_duration, out_step), dim = 1)
        
        return out
    
    def training_step(self, batch, batch_idx): 
        
        print("INSIDE TRAINING STEP\n")
          
        data, labels = batch
 
        out = self.forward(data)
        out_pitch = out[:, 0:128]        
        out_duration = torch.unsqueeze(out[:, -2], 1)        
        out_step  = torch.unsqueeze(out[:,-1], 1)
 
        #LOSSES
        processed_pitch_target =  labels[:,0].to(torch.int64)

        loss_pitch = nn.CrossEntropyLoss()(out_pitch, processed_pitch_target)
        loss_duration = utils.mse_with_positive_pressure(labels[:,1], torch.squeeze(torch.squeeze(out_duration, 0),1))
        loss_step = utils.mse_with_positive_pressure(labels[:,2], torch.squeeze(torch.squeeze(out_step, 0),1))

        print("Crossentropy loss", loss_pitch)
        print("MSE Duration", loss_duration)
        print("MSE Step", loss_step)
        

        #output is : 100*50*130 : 100, batch size, 50 * 130 is 50: sequence length, 130 prediction of features 
        
        loss = 0.05*loss_pitch + 1.0*loss_duration + 1.0*loss_step
       
        self.log("Pitch Loss (Sparse CE): ", loss_pitch )
        self.log("Duration Loss (MSE): ", loss_duration)
        self.log("Step Loss (MSE): ", loss_step)
        self.log("Total_LOSS", loss)
        

        return loss
       
    def validation_step(self, batch, batch_idx): 
        
        #implicitally takes a batch of validation data pairs and it is expected to compute the validation loss.
        #a validation step is like a training step, but at the end the weights and biases aren't updated. 
        #it is used to check the generalization abilities of the network on unseen data 

        print("VALIDATION LOOP")
        data, labels = batch
        #Implicitally takes a batch of training data pairs from data loader and the index of the batch.
        #It is expected to return the loss between the predicted and expected output. 
        #The predicted output is what comes out of the forward function

        #self calls forward! 
        out = self.forward(data)
        out_pitch = out[:, 0:128]
        
        print(out_pitch.shape)
        out_duration = torch.unsqueeze(out[:, -2], 1)
        print(out_duration.shape)
        out_step  = torch.unsqueeze(out[:,-1], 1)
        print(out_step.shape)
        
     
        #LOSSES
        
        processed_pitch_target =  labels[:,0].to(torch.int64)
        val_loss_pitch= nn.CrossEntropyLoss()(out_pitch, processed_pitch_target)
        val_loss_duration = utils.mse_with_positive_pressure(labels[:,1], torch.squeeze(torch.squeeze(out_duration, 0),1))
        val_loss_step = utils.mse_with_positive_pressure(labels[:,2], torch.squeeze(torch.squeeze(out_step, 0),1))
    
        val_loss = 0.05*val_loss_pitch + 1*val_loss_duration + 1*val_loss_step
        self.log("TOTAL_Val_Loss", val_loss)


        return val_loss


#Class definition ends
    
#Main
if __name__ == "__main__":
    

    print(torch.cuda.is_available())

  
        ## DEPTH ## 
    #DEPTH = 25
    #DEPTH = 50
    #DEPTH = 100
    DEPTH = 150
   
        ##Unprocessed midi data
    
    #Dataset path 
    dataset_path = "C:\\Users\\Sisso\\Desktop\\Midi_Improviser\\python\\maestro-v2.0.0"
    midi_seq = utils.MaestroToTensor(dataset_path)
  
    #Tensor of the shape : NXL, N = nb of midi features (pitch [0], duration [1], step[2]), L = nb of midi messages analyzed. 
    #need to be divided in sequences to be used in lstm
    
    dataset, targets, datalen = utils.datasetCreation(midi_seq, DEPTH)
    dataset = dataset.to(device=DEVICE)
    targets = targets.to(device=DEVICE)
    #Batch size
    bs = 256
    train_len = math.floor(dataset.shape[0]*0.8)
    val_len = math.floor(dataset.shape[0]*0.2)
    
    
    
    train_data= dataset[: train_len -(train_len%bs)]

    
    train_labels = targets[:train_len -(train_len%bs)]
    


    val_data = dataset[train_len -(train_len%bs): (train_len -(train_len%bs) + (val_len - val_len%bs))]
    
    val_labels = targets[train_len -(train_len%bs): (train_len -(train_len%bs) + (val_len - val_len%bs))]


    


    math.floor(train_labels.shape[0]/bs)
    #Since we’re using lightning, we have to combine the training inputs and outputs into a TensorDataset: 
    train_set = TensorDataset(train_data, train_labels)
    val_set = TensorDataset(val_data, val_labels)
    #and then we use tensorDataset to create a dataloader: 
    train_dataloader = DataLoader(train_set, batch_size = bs, shuffle = True)
    val_dataloader = DataLoader(val_set, batch_size = bs, shuffle = True)

    
    checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{Total_LOSS: .4f}',
    monitor='Total_LOSS',
    save_top_k=3,
    mode='min',
    )

    
    
    trainer = Trainer(accelerator = DEVICE_S , max_epochs=60, log_every_n_steps = 1000, callbacks = [checkpoint_callback])
    midi_lstm = model_test(lstm_hu = 128)
    midi_lstm.to(DEVICE)
    trainer.fit(midi_lstm, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
 




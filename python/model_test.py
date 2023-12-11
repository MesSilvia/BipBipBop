import torch  #used to create tensors to store values
import torch.nn as nn# used to make weights and biases part of the neural network
import torch.nn.functional as F # activatiom functions
from torch.optim import SGD # stochastic gradient descent
import lightning as L # Training easier to code!!
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import TensorDataset, DataLoader #for large datasets !
from torch.optim import Adam
import pytorch_lightning as pl
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
        super().__init__() #mother class constructor
        self.save_hyperparameters()
        #Saving hyperparameters to checkpoint !!!! 
        self.lstm_hu = lstm_hu
        self.lstm_pitch = nn.LSTM(input_size=3, hidden_size = lstm_hu, batch_first = True)
        #self.lstm_pitch = nn.LSTM(input_size=1, hidden_size = lstm_hu, batch_first = True)
        #self.lstm_pitch = nn.LSTM(input_size=1, hidden_size = lstm_hu, batch_first = True)
        #self.lstm_duration = nn.LSTM(input_size=1, hidden_size = lstm_hu, batch_first = True)
        #self.lstm_step = nn.LSTM(input_size=1, hidden_size = lstm_hu, batch_first = True)


        self.linear_pitch = nn.Linear(in_features = lstm_hu, out_features = 128)
        self.linear_duration = nn.Linear(in_features = lstm_hu, out_features = 1)
        self.linear_step = nn.Linear(in_features = lstm_hu, out_features = 1)

        self.train_match = 0
        self.val_match = 0


        self.hp = None
        self.cp = None

        #self.hd = None
        #self.cd = None

        
        #self.hs = None
        #self.cs = None


        #130 Outputs: 128 for sparse Crossentropy for pitch prediction, the remaining 2 for duration and step. 
        #They will have obviously different losses ! 
        #Expected input : L x N x H
    

    def configure_optimizers(self): 
        #Describes the method we want to use to optimize neural network 
        return Adam(self.parameters(), lr = 0.005)

    def initialize_hidden_state(self, batch_size): 
        self.hp = torch.zeros(1, batch_size, self.lstm_hu, device=DEVICE)
        self.cp = torch.zeros(1, batch_size, self.lstm_hu, device=DEVICE)
        #self.hd = torch.zeros(1, batch_size, self.lstm_hu, device=DEVICE)
        #self.cd = torch.zeros(1, batch_size, self.lstm_hu, device=DEVICE)
        #self.hs = torch.zeros(1, batch_size, self.lstm_hu, device=DEVICE)
        #self.cs = torch.zeros(1, batch_size, self.lstm_hu, device=DEVICE)
    
    def forward(self, input): 

        #print("Network input shape", input.shape)
        #Visto che batch first True, ci si aspetta l'input come batch_sizeXsequence_lengthXfeatures_length
        #Se batch_first è falso, allora sequence_lengthXbatch_sizeXfeatures_length
        #Debug, press c to continue
       
        #A third dimension must be added to each input to match the lstm expected input NxLxInputSize (1)
        # zeros((1, b_s,lstm_hu))

        
        if self.hp is None or self.cp is None : 
            self.initialize_hidden_state(input.shape[0])
        
        input_pitch_lstm = input

        #input_pitch_lstm = torch.unsqueeze(input[:,:,0],2)
        #input_duration_lstm = torch.unsqueeze(input[:,:,1],2)
        #input_step_lstm = torch.unsqueeze(input[:,:,2],2)
        #print("input pitch lstm shape", input_pitch_lstm.shape)
        #print("input pitch ltsm ", input_pitch_lstm)

        output_pitch_lstm, (self.hp, self.cp) = self.lstm_pitch(input_pitch_lstm, ((self.hp).detach(), (self.cp).detach()))

        #output_duration_lstm, (self.hd, self.cd) = self.lstm_duration(input_duration_lstm, ((self.hd).detach(), (self.cd).detach()))
        #output_step_lstm, (self.hs, self.cs) = self.lstm_step(input_step_lstm, ((self.hs).detach(), (self.cs).detach()))    

        out_pitch = self.linear_pitch(output_pitch_lstm[:,-1,:])
        out_duration = self.linear_duration(output_pitch_lstm[:,-1,:])
        out_step = self.linear_step(output_pitch_lstm[:,-1, :])


        # out_pitch = self.linear_pitch(output_pitch_lstm[:,-1,:])
        #out_duration = self.linear_duration(output_duration_lstm[:,-1,:])
        #out_step = self.linear_step(output_step_lstm[:,-1, :])

   
        #Builds the neural network structure: describes what happens to the input while passing though all the layers
        #Here, it is simply for the moment, output = lstm(input)
        out = torch.cat((out_pitch, out_duration, out_step), dim = 1)
        
        return out
        #return out_pitch
    
    def training_step(self, batch, batch_idx): 
        
        print("INSIDE TRAINING STEP\n")
          
        #To do : normalize pitches, un-normalize them when computing loss wrt to true labels (they should be integers). Forse no, 
        #all'uscita dell'ultimo layer, escono dei logits 128 long. la sparse cross entropy confronta la versione one hot encoded del label i.e. pdf, 
        #con il tensore 128 long di logits. Forse anzi è assolutamente necessario che venga diviso per 128.......... all'inizio direttamente !!! 
        #è necessario per le funzioni di attivazione probabilmente. 
        #anche se forse c'è la softmax implicita non si è capito.....

        data, labels = batch
 
        #print("data shape", data.shape)
        #print("labels shape", labels.shape)
        #print("Batch index ", batch_idx)

        #Implicitally takes a batch of training data pairs from data loader and the index of the batch.
        #It is expected to return the loss between the predicted and expected output. 
        #The predicted output is what comes out of the forward function
        
        out = self.forward(data)
        out_pitch = out[:, 0:128]        
        out_duration = torch.unsqueeze(out[:, -2], 1)        
        out_step  = torch.unsqueeze(out[:,-1], 1)
        
        #out_pitch = self.forward(data)
        #print("batch index", batch_idx, "\n")
        #print("pirch branch output shape: ", out_pitch.shape ,"\n")
        #print("pitch branch output:", out_pitch, "\n")
        #print("Higher logit in pitch prediction", torch.argmax(out_pitch[-1,:]))
        
        #print("out pitch shape", out_pitch.shape)# batch_size *128
        #print("out pitch content: ", out_pitch[0:3, :])
        
        #print("out duration shape", out_duration.shape) batch_size * 1
        #print("out step shape", out_step.shape) batch_size * 1
        #LOSSES
        processed_pitch_target =  labels[:,0].to(torch.int64)
        #print("target shape", processed_pitch_target.shape)
        #print("target ", processed_pitch_target)
        #print("pitch branch output shape: , out_pitch.shape ,"\n")

        #print("OUT DURATION ")
        #print(out_duration.shape)
        #print(out_duration)

        #print("LABELS DURATION")
        #print(torch.unsqueeze(labels[:,1], 1))
        #print(torch.unsqueeze(labels[:,1], 1))

        loss_pitch = nn.CrossEntropyLoss()(out_pitch, processed_pitch_target)
        loss_duration = utils.mse_with_positive_pressure(labels[:,1], torch.squeeze(torch.squeeze(out_duration, 0),1))
        #loss_duration = nn.MSELoss()(out_duration ,torch.unsqueeze(labels[:,1], 1))
        #loss_step = nn.MSELoss()(out_step ,torch.unsqueeze(labels[:,2], 1))
        loss_step = utils.mse_with_positive_pressure(labels[:,2], torch.squeeze(torch.squeeze(out_step, 0),1))

        print("Crossentropy loss....", loss_pitch)
        print("MSE Duration", loss_duration)
        print("MSE Step", loss_step)
        
        #Yeah??!?!?!?!?
        #self.train_match = self.Matches(processed_pitch_target, out_pitch, batch_idx, self.train_match)
        #output is : 100*50*130 : 100, batch size, 50 * 130 is 50: sequence length, 130 prediction of features 
        
        loss = 0.05*loss_pitch + 1.0*loss_duration + 1.0*loss_step
       
        self.log("Pitch Loss (Sparse CE): ", loss_pitch )
        self.log("Duration Loss (MSE): ", loss_duration)
        self.log("Step Loss (MSE): ", loss_step)
        self.log("Total_LOSS", loss)
        
        #self.log("Train Accuracy", self.train_match)
        #last element

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
        #out_pitch= self(data)
        out = self.forward(data)
        out_pitch = out[:, 0:128]
        
        print(out_pitch.shape)
        out_duration = torch.unsqueeze(out[:, -2], 1)
        print(out_duration.shape)
        out_step  = torch.unsqueeze(out[:,-1], 1)
        print(out_step.shape)
        
        #print("out pitch shape", out_pitch.shape)
        #print("out duration shape", out_duration.shape)
        #print("out step shape", out_step.shape)
        #LOSSES
        #print("dimension of one column of labels", labels[:,0].shape)
        processed_pitch_target =  labels[:,0].to(torch.int64)
        #print("Processed pitch target", processed_pitch_target)
        val_loss_pitch= nn.CrossEntropyLoss()(out_pitch, processed_pitch_target)
        val_loss_duration = utils.mse_with_positive_pressure(labels[:,1], torch.squeeze(torch.squeeze(out_duration, 0),1))
        val_loss_step = utils.mse_with_positive_pressure(labels[:,2], torch.squeeze(torch.squeeze(out_step, 0),1))
        print("VALIDATION MEASURES:\n")
        print("VALIDATION Crossentropy loss", val_loss_pitch)
        print("VALIDATION MSE Duration", val_loss_duration)
        print("VALIDATION MSE Step", val_loss_step)
        #To have logs of the accuracy for tensorboard, 
        #def training_epoch_end(self,outputs):
        #  the function is called after every epoch is completed
        #Must be implemented.
        #outputs is a python list containing the batch_dictionary 
        #from each batch for the given epoch stacked up against each other

        #self.val_match = self.Matches(processed_pitch_target, out_pitch, batch_idx, self.val_match)
        val_loss = 0.05*val_loss_pitch + 1*val_loss_duration + 1*val_loss_step
        #self.log("Val loss pitch", val_loss_pitch)
        #self.log("Val loss duration", val_loss_duration)
        #self.log("Val loss step", val_loss_step)
        self.log("TOTAL_Val_Loss", val_loss)
        
        
        #self.log("Val Matches x Epoch", self.val_match)

        return val_loss

 
        #val_loss = 0

        

    def Matches(self, batch_labels, batch_prediction, batch_idx, match): 
        print("Prediction accuracy! ")

        if(batch_idx==0):
            match=0
        
        for i in range(0, len(batch_labels)): 

            #print("Sample num", i, "Batch num:", batch_idx, "\n")
            #print("Prediction:", torch.argmax(batch_prediction[i]))
            #print("True label: ", batch_labels[i], "\n")
            #QUesta è una zstronzata
            
            if(batch_labels[i] == torch.argmax(batch_prediction[i])):
                print("Match !")    
                match = match + 1
        #print("End accuracy computation")
        
        return match


#Class definition ends
#Main
if __name__ == "__main__":
    

    print(torch.cuda.is_available())

  
        ## DEPTH ## 
    #DEPTH = 25
    #DEPTH = 50
    #DEPTH = 100
    DEPTH = 150
   
        ##
    midi_seq = utils.MaestroToTensor("C:\\Users\\Sisso\\Desktop\\Midi_Improviser\\python\\maestro-v3.0.0\\2017")
    #unprocessed midi data
    #Tensor of the shape : NXL, N = nb of midi features (pitch [0], duration [1], step[2]), L = nb of midi messages analyzed. 
    #need to be divided in sequences to be used in lstm
    #print(midi_seq[0][0:10])
    



    dataset, targets, datalen = utils.datasetCreation(midi_seq, DEPTH)
    print(dataset.shape[0])
    
    dataset = dataset.to(device=DEVICE)
    targets = targets.to(device=DEVICE)
    # print(targets[0:10, : ])
    #print(dataset[0:10, :, 1])
    bs = 64
    train_len = math.floor(dataset.shape[0]*0.8)
    val_len = math.floor(dataset.shape[0]*0.2)
    #len: 21980
    
    #train, test 


    #train data/labels : 10000
    train_data= dataset[: train_len -(train_len%bs)]
    print(train_data.shape)
    
    train_labels = targets[:train_len -(train_len%bs)]
    #val data/labels : 50000
    print("train labels main")
    print(train_labels) 


    val_data = dataset[train_len -(train_len%bs): (train_len -(train_len%bs) + (val_len - val_len%bs))]
    print(val_data.shape)
    val_labels = targets[train_len -(train_len%bs): (train_len -(train_len%bs) + (val_len - val_len%bs))]
    #test data/labels : circa 60000
    #test_data = dataset[15000:]
    #test_labels = targets[15000:]



    
    '''
      
    print("Train data: ", train_data.shape, "Train labels : ", train_labels.shape)

    print("Val data: ", val_data.shape, "Val labels : ", val_labels.shape)
    
    print("Test data: ", test_data.shape, "Test labels : ", test_labels.shape)

 
    Train data:  torch.Size([10000, 50, 3]) Train labels :  torch.Size([10000, 3])
    Val data:  torch.Size([5000, 50, 3]) Val labels :  torch.Size([5000, 3])
    Test data:  torch.Size([6980, 50, 3]) Test labels :  torch.Size([6980, 3])
    
    
    '''


        ## BATCH SIZE 

        ##
    math.floor(train_labels.shape[0]/bs)
    #Since we’re using lightning, we have to combine the training inputs and outputs into a TensorDataset: 
    train_set = TensorDataset(train_data, train_labels)
    val_set = TensorDataset(val_data, val_labels)
    #and then we use tensorDataset to create a dataloader: 
    train_dataloader = DataLoader(train_set, batch_size = bs, shuffle = True)
    val_dataloader = DataLoader(val_set, batch_size = bs, shuffle = True)

    print("Train and Val set uploaded correctly")

    
    
    checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/v4/',
    filename='model-{epoch:02d}-{Total_LOSS: .4f}',
    monitor='Total_LOSS',
    save_top_k=3,
    mode='min',
    )

    trainer = Trainer(accelerator = DEVICE_S , max_epochs=60, log_every_n_steps = 300, callbacks = [checkpoint_callback])
    midi_lstm = model_test(lstm_hu = 128)
    midi_lstm.to(DEVICE)
    trainer.fit(midi_lstm, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
 




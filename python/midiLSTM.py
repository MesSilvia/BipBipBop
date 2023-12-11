import utils

from math import ceil, floor
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mchmm as mc
import pandas as pd
import json
from json import JSONEncoder
import random

import lightning as L 
from torch.utils.data import TensorDataset, DataLoader
import tensorboard

class midiLSTM(L.LightningModule): 
    def __init__(self, DEPTH):

        super().__init__()
        
        self.lstm = nn.LSTM(input_size=1, hidden_size = 127, batch_first = True)
        self.fc = nn.Linear(in_features=32, out_features = 127)
        self.notes = torch.empty(127) 


        #input_size refers to the number of features, variables we have in the training data
        #in our case we have only one feature (size of input), that is 
        #the value for a company
        #hidden_size : number of output we want 


    def forward(self, input): 
    #turns the tensor into a row tensor into a col tensor
        print("input inside forward", input)
        input = torch.transpose(input,0,1)
        input = input.unsqueeze(-1)
        lstm_out,    temp = self.lstm(input)

    ## NOTA BENE : lstm_out contains the short term memory values for each lstm we've unrolled. 
    #in this case it means it has 4 values, because we unrolled lstm four times for each of the input (day 1 - day 4)
        sequence_representation = lstm_out[-1]
        #logits = self.fc(sequence_representation)
        probabilities = torch.softmax(sequence_representation/(0.4), dim = 1)
        
        #print('probabilities shape', probabilities.shape )
        print("probabilities shape", probabilities.shape)
        
       
        #probabilities = torch.transpose(probabilities, 0, 1)

        return probabilities


    def configure_optimizers(self): 
        return Adam(self.parameters(), lr = 0.001)
        #lr very big 


    def training_step(self, batch, batch_idx): 
        print('print batch_idx', batch_idx)
        #training step is equal to the handmade version: 
        input_i, label_i = batch
        print('label_i', label_i)
        print('label_i shape', label_i.shape)

        output_i= self.forward(input_i)

        print('output_i', output_i)
        print('output_i shape', output_i.shape)
        
        #output_i = torch.transpose(output_i, 0, 1)
        loss = nn.CrossEntropyLoss()(output_i.view(-1,127 ), label_i.view(-1,127))
        #Batching is automatically done by pytorch 
       
        
        self.log("train loss", loss)

        return loss
    

    def ListOfIntToString(self, lint): 
        rep_str = []
        for i in range(0, len(lint)): 
            to_append = str(lint[i]).replace(".0", "")
            rep_str.append(to_append)
            

        return rep_str

  
    def textToList(self, s): 
        notesview = []
        notes = []
        lengths = []
        pauses = []
        for i in range(0, len(s)): 
            note, length, pause = s[i].split(':')

            notes.append(float(note))
            lengths.append(float(length))
            pauses.append(float(pause))

        return notes, lengths, pauses
    

    def defInputTensors(self, depth, tensor):

        tensor_list = []
        cnt = 0 
        

        for i in range(0, len(tensor)-(depth-1)):
            sub_list = []
            while(cnt<depth): 
                sub_list.append(tensor[i+cnt])
                cnt = cnt + 1 
            tensor_list.append(sub_list)
            cnt = 0 

        return tensor_list


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred= self(x)


        loss = nn.CrossEntropyLoss()(y_pred.view(-1,127 ), y.view(-1,127))
        self.log("Val loss", loss)
        
        return {"val_loss": loss}


    def getProbForSequence(self, notes): 

        prob = torch.zeros(127)
        prob.fill_(100)

        for i in range(0, 127, 1): 

            for note in notes: 

                if (note==(i+0.0)): 
                    print('check:',note, 'i:', i)
                    prob[i] = prob[i] - 20
 
        print("Prob shape", prob.shape)
        print("prob", prob) 
        return prob
    
    def nOrderMarkovChainMatrix(self, notes, depth): 
        list_dict = []
        #Splitting the notes list in sequences of length depth
        dict_sequence_next = {"sequence:"}
        sequences_list = []
        next_note_list = []
        cnt = 0 
         

        for i in range(0, len(notes)-(depth-1)):
            sub_list = ''
            while(cnt<depth): 

                if((cnt == 0)): 
                    sub_list = notes[i+cnt]
                else: 
                    sub_list = sub_list +'-'+ notes[i+cnt]
                cnt = cnt + 1 

            if(i < (len(notes)-depth)): 
                next_note_list.append( notes[depth+i])
            
            sequences_list.append(sub_list)
            cnt = 0 
        
        #We can ignore the last sequence, since we don't know what's next
        sequences_list = sequences_list[0:-1]
        print('Sequences list', sequences_list)
        print('Nextnote list', next_note_list)

        list_un = [sequences_list, next_note_list]
        unique_sequences = list(set(sequences_list))
        unique_notes = list(set(notes))
        
        print("Unique values", unique_sequences)
        
        for i in range(0, len(unique_sequences)): 
            list_dict.append( { "seq" : unique_sequences[i], "val" : torch.zeros(size = (1, 127)) } )     
        print("list_dict", list_dict)
        

        for i in range(0, len(list_dict)): 
            for j in range(0, len(sequences_list)): 
                if(list_dict[i]["seq"] == sequences_list[j]): 
                    list_dict[i]["val"][0][int(next_note_list[j])] = list_dict[i]["val"][0][int(next_note_list[j])] + 1 

            sum = np.sum(list_dict[i]["val"].numpy())
            print('sum', sum)
            list_dict[i]["val"] = torch.div(list_dict[i]["val"], sum)
            print("inside loop", list_dict[i]["val"])
                
              

        out_list = torch.zeros(len(sequences_list), 127)
        for i in range(0, len(sequences_list)): 

            for j in range(0, len(list_dict)): 

                if(list_dict[j]["seq"]==sequences_list[i]): 

                    print('shape list_dict [0] : ',  list_dict[j]["val"].shape)
                    print('shape out_list[i] : ',  out_list[i].shape) 
                    out_list[i][:] = list_dict[j]["val"]
                    #out_list.append(list_dict[j]["val"])
            

        print("out_list", out_list)
        input()
        return list_dict, out_list, sequences_list
        #for i in range(0, len(sequences_list)): 



            
    def save_for_rtneural(self, outfile):
        ## used for saving 
        class EncodeTensor(JSONEncoder):
            def default(self, obj):
                if isinstance(obj, torch.Tensor):
                    return obj.cpu().detach().numpy().tolist()
                return super(json.NpEncoder, self).default(obj)
            
        with open(outfile, 'w') as json_file:
            json.dump(self.state_dict(), json_file,cls=EncodeTensor)



    def addRandomicity(self, list_c, depth):
        #Copia per reference ! 
        #ext = list
        cnt = 0 
        ext = list(list_c)
        for i in range(0, len(list_c), depth):
            print('i', i)
            if(i != 0):  
                ext.insert(i + cnt, float(random.randint(50, 100)))
                cnt =+ 1
            print(ext)
            
        
        return ext





    def tensor_slide(self, val, tensor): 

        t_val = torch.tensor(val[0][0])
        for i in range(tensor.size(1) - 1):
            
            tensor[0][i] = tensor[0][i+1]

    # Insert the new value at the end
        tensor[0][-1] =t_val

        return tensor
                       


if __name__ == "__main__":
    #TO DO - PLUGINEDITOR / PROCESSOR 
    #OCCURRANCES USED FOR TRAINING! FIXED, BUT THEN CAN BE USER DEFINED 
    #HE CAN SELECT BETWEEN 1 AND THE LENGTH OF TRAINING SEQUENCE 
    #TEXT EDITOR 

    
    client_socket = utils.ServerToClientSocketSetup()

    message = utils.receivingFromClient(client_socket)

    

    DEPTH = int(message); 
    print("Depth:", DEPTH)  
    #RECEIVING LENGTH OF SEQUENCE FROM JUCE
    message = utils.receivingFromClient(client_socket)
  

    lgt = int(message); 
    print("Length", lgt)  

   

    model = midiLSTM(DEPTH)
    contents = utils.textProcessing()
    print(contents)
    




    input()
    
    notes, lengths, pauses = model.textToList(contents)
    ext = model.addRandomicity(notes, DEPTH)
    print("Original notes: ", notes)
    print("Notes with randomicity:", ext)

    #notes = ext
    print("Notes to be used: ", notes)
    input()
    #Extending input to random combination of notes of the sequence

    #input_ex = []

    #for i in range(0, len(notes)): 
    #for i in range(0, 3): 
    #    input_ex.extend(notes)
    #    random.shuffle(notes)
        

    #print("input_ex", input_ex)
    #input()
    model.notes = model.getProbForSequence(notes)
    notesview = model.ListOfIntToString(notes)
    
    scaler = MinMaxScaler((-100 , 100))

    notes_2d = []
    #need matrix like form
    notes_2d.append(notes)

    print("notes_2d:", notes_2d)
    #need a col per feature vector
    notes_array = np.array(notes_2d)
    notes_array = np.transpose(notes_2d)
    print("shape, length", notes_array.shape, len(notes_array)) 
    scaler.fit(notes_array)
    notes_std = scaler.transform(notes_array)
    print("Std notes: ", notes_std,)

    #To build dataset, need a list 
    print("Std notes to list",list(np.transpose(notes_std)[0]))



    notes_std_toproc = list(np.transpose(notes_std)[0])



    


    print("notes over 127", notesview)

 
    
    print("notes normalized: ", notes_std)
    
    print("lengths: ", lengths)

    print("pauses: " , pauses)

    
    print("Training communication finished")
    #Cancellare contenuto text quando train è off, dopo che è stato salvato
    

   

    utils.textErase()


    tensor_list = model.defInputTensors(DEPTH, notes_std_toproc)
    
    print("---------------------------------------")
    print("For depth = ", DEPTH) 
    print(tensor_list)
    
    viewlist = model.defInputTensors(DEPTH, notesview)


    labels = []
    labelsview = []

    out_dict, prob_tensors, sequences= model.nOrderMarkovChainMatrix(notesview, DEPTH)


    for i in range(1, len(tensor_list)):
        print("Writing labels")

        labelsview.append(viewlist[i][DEPTH-1])
        labels.append(tensor_list[i][DEPTH-1])
    

    

    #Adding for every batch size one training data that is randomly(?) generated (sequence + note?)
    batch_size = 3
    print("Nb of batches:" , floor((len(tensor_list)-1)/batch_size))

    inputs = (torch.tensor(tensor_list[:-1])).float()
    

    X_train, X_val, y_train, y_val = train_test_split(inputs, prob_tensors, test_size=0.5, random_state=33, shuffle = False)
    
    print("train_test_split")
    

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    
    print("Dataset created. Continue with training? ")

    
    trainer = L.Trainer(max_epochs=1500, log_every_n_steps = 2)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    data_to_send = "0"
    client_socket.send(data_to_send.encode())


    
    prec = 0 
    
    model.save_for_rtneural("model.json")
    print("saved for rtNeural")

    for i in range(0, len(tensor_list) - 1): 

        print("After OPTIMIZATION! ..")
        print("\nComparaison btw observed and predicted values 0")
        print("Depth:", DEPTH, "Sequence:", viewlist[i], " : Observed = ", labelsview[i]," Predicted=")

        tensor_input = torch.unsqueeze(torch.tensor(tensor_list[i]), 1)
        input_i = torch.transpose(tensor_input,0,1)
        print("input_i shape", input_i.shape)
        print("input_i", input_i)
        result = model(input_i.float()).detach()
        print("result matrix ", result)
        print("result shape", result.shape)

        t_result = torch.argmax(result)
        print("Predicted", t_result.item(), 'Observed', labelsview[i])

        if(str(t_result.item()) == labelsview[i]): 
            print("Great!")
            prec = prec + 1

        prec_p = prec / (len(tensor_list)-1)

        print("Precision percentage: ", prec_p*100 )


    
   
    tensor_input = torch.unsqueeze(torch.tensor(tensor_list[-1]), 1)
    input_i = torch.transpose(tensor_input,0,1)
      #while play is on

#FIFO Stack !!!! A..AB...ABC...BCD.... A is the first in and the first out. 
    message = client_socket.recv(1024).decode()




    print(f"Received message: {message}")
    cnt = 0 
    while(message == "1" and cnt < lgt) :
        result = model(input_i.float()).detach()

        print("RESULT MATRIX", result)
        sum =  torch.sum(result[0])       
        print(" Is the sum of the result probabilities equal to one?")
        print("sum: ", sum)
        sorted, indexes = torch.sort(result, dim = 1, descending = True)
        print("First most probable note:", indexes)
        #print("Second most probable note:", indexes[1])
        t_result = torch.argmax(result)
        client_socket.send(str(t_result.item()).encode())
        input_i = model.tensor_slide(scaler.transform([[t_result.item()]]), input_i)
        cnt = cnt + 1
     
        

        


    #Fino a quando play è on, faccio questo 
    


    client_socket.close()
    server_socket.close()
    


        


    
 


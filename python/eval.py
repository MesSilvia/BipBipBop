import os
import subprocess
import sys


from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import numpy as np
import torch
import utils
from model_test import model_test



DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    print("Python original folder")
    plugin_path = os.getcwd()

    dir_text = plugin_path + "\\Train.txt"

    os.chdir("../")
    
    cur_dir = os.getcwd()
    

    #Communication logic with Juce Client - Python Server
    


  
   
    
    client_socket = utils.ServerToClientSocketSetup()
    a = True
    while a: 
        

        #Receiving depth: 

        message = utils.receivingFromClient(client_socket)
        open("Python\\Sequence.txt", "w").close() 
        DEPTH = int(message); 
        print("Depth:", DEPTH)  
        check = "1" 
        utils.sendToClient(client_socket, check)


        #Receiving Length:
        message = utils.receivingFromClient(client_socket)
        LENGTH = int(message); 
        print("Length", LENGTH)  
        utils.sendToClient(client_socket, check)


        message = utils.receivingFromClient(client_socket)
        TEMPERATURE = float(int(message)/100)
        print("Temperature", TEMPERATURE)
        utils.sendToClient(client_socket, check)

        input_sequence_pitch, input_sequence_dur, input_sequence_step = utils.textProcessing(dir_text)

   
        #128 hu, 100 batchsize, depth: 100

        checkpoint_path = cur_dir + "\\Python\\checkpoints\\v4\\dpt" + str(DEPTH) + ".ckpt"

        #v1
        #dpt25Total_LOSS= 0.1950
        #dpt50Total_LOSS= 0.2218
        #dpt100Total_LOSS= 0.1981
        #dpt150Total_LOSS= 0.1997

        #v2
        #dpt25 Total_LOSS= 0.1953
        #dpt 50 total_loss = 0.2186


        #v3
        #dpt25 Total_LOSS= 0.2166.ckpt bs= 64
        #dpt50 Total_loss = 0.1912 bs=64
        #dpt100 Total_LOSS= 0.2101 bs = 128


        #v4
        # dpt 25 model-epoch=14-Total_LOSS= 0.2012
        # dpt 50 model-epoch=05-Total_LOSS= 0.2051
        # dpt 100 model-epoch=13-Total_LOSS= 0.1939
        # dpt 150 model-epoch=05-Total_LOSS= 0.2214
        scaler = load("Python\\scaler.joblib")
        depth = scaler.n_features_in_
        #Deptch from juce sceglie quale scaler prendere IMPO !!! Depth modello scelto e scaler devono corrispondere 


        #To transform the pitches of a sequence in the same way we did in the train: 
        #the scaler expectes a 1*100 (depth) feature vector to transform
        #Se we take the pitch, and repeat it 1*depth times, creating a row vector
        #For a sequence, we need sequence_len*depth
        #depth is retrieved from scaler features description


        #To encapsulate, passing the input from juce!!!     
        #@Input_definition:
        #input new note ! implement fifo ..... 
        #input_sequence_pitch = np.array([[20],[20],[20],[20],[20],[20],[20],[31],[42],[22],[33],[62],[64],[64],[61],[72],[61],[20],[20],[20]])
        #print("input sequence shape", input_sequence_pitch.shape)
        #input_sequence_dur = torch.tensor(lengths, dtype = torch.float64)
        #input_sequence_step = torch.tensor(pauses, dtype = torch.float64)

    
        #OK!!! ADESSO UNSQUIZZI TUTTO E FIDI IL MODELLO !!!!!!!! 
        #RETURN model_input

        model_input = utils.modelInputDef(input_sequence_pitch, input_sequence_dur, input_sequence_step, depth, scaler)
        myModel = model_test.load_from_checkpoint(checkpoint_path, map_location = DEVICE)
        myModel = myModel.to(DEVICE)
        original_len = model_input.shape[1]
        new_input = torch.zeros((1, original_len, 3))
        txt = ""
        print("New input, sequence generation !!!")
        for i in range(0,LENGTH): 

            if i == 0 : 
                out= myModel(model_input)
                o1 = out[:, 0:128]        
                o2 = torch.unsqueeze(out[:, -2], 1)        
                o3  = torch.unsqueeze(out[:,-1], 1)
                #print(o1.shape)            
                #print("Distribution pitch output \n", o1)
                predicted_pitch_midi = torch.multinomial(torch.softmax(o1/TEMPERATURE, dim=1),1)

                print("Output pitch: ", predicted_pitch_midi, "\n Output Duration : ", o2.detach(), "\n Output Step", o3.detach())
                pitch_transformed = utils.pitchTransform(predicted_pitch_midi, scaler, depth)

                predicted_pitch = torch.unsqueeze(torch.unsqueeze(pitch_transformed,0),0)
                new_input = utils.updateInputFixedLen(model_input, predicted_pitch, o2.detach(), o3.detach())
                txt = str(predicted_pitch_midi[0][0].item()) + ":" + str(o2[0][0].item()) + ":" + str(o3[0][0].item()) + "\n"
                #print("I = 0 !")
                #print(new_input.shape)
                #print(new_input)
            else: 
                out= myModel(model_input)
                o1 = out[:, 0:128]        
                o2 = torch.unsqueeze(out[:, -2], 1)        
                o3  = torch.unsqueeze(out[:,-1], 1)
                predicted_pitch_midi = torch.multinomial(torch.softmax(o1/TEMPERATURE, dim=1),1)
                print("Output pitch: ", predicted_pitch_midi, "\n Output Duration : ", o2.detach(), "\n Output Step", o3.detach())
                pitch_transformed = utils.pitchTransform(predicted_pitch_midi, scaler, depth)

                predicted_pitch = torch.unsqueeze(torch.unsqueeze(pitch_transformed,0),0)
                new_input = utils.updateInputFixedLen(new_input, predicted_pitch, o2.detach(), o3.detach())
                txt += str(predicted_pitch_midi[0][0].item()) + ":" + str(o2[0][0].item()) + ":" + str(o3[0][0].item()) + "\n"
                #print(new_input.shape)
                #print(new_input)


            #Item get value of a scalar tensor     




        with open("Python\\Sequence.txt", "w") as file: 
            file.write(txt)

        #Send zero when logic is finished and sequence is ready to be played
        utils.sendToClient(client_socket, "0")
        #open("Sequence.txt", "w").close()
        
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
import muspy


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    
    #This script is called directly from the plugin so the current path isn't the one of python folder
    #But the one containing the .sln file. 
    plugin_path = os.getcwd()
    dir_text = plugin_path + "\\Collection.txt"
    #Collection.txt is filled everytime the user clicks on continuation settings, insert values, and starts playing
    #The notes played along with the duration and the step get written here. 
    #This is retrieved by textProcessing and turned into sliding sequences and fed to the model 
    os.chdir("../")
    cur_dir = os.getcwd()
    

    #Communication logic with Juce Client - Python Server setup
    client_socket = utils.ServerToClientSocketSetup()

    #RETRIEVING SAMPLE RATE FROM JUCE
    message = utils.receivingFromClient(client_socket)
    SR =  int(message)
    check = "2" 
    utils.sendToClient(client_socket, check)
    
    

    a = True
    while a: 
        
        #Receiving depth: 
        message = utils.receivingFromClient(client_socket)
        open("Python\\Sequence.txt", "w").close()
        DEPTH = int(message); 
        #Synchronization step 
        check = "1" 
        utils.sendToClient(client_socket, check)

        #Receiving Length:
        message = utils.receivingFromClient(client_socket)
        LENGTH = int(message); 

        utils.sendToClient(client_socket, check)

        #Receiving Temperature
        message = utils.receivingFromClient(client_socket)
        TEMPERATURE = float(int(message)/100)

        utils.sendToClient(client_socket, check)

        input_sequence_pitch, input_sequence_dur, input_sequence_step, notes_mus = utils.textProcessing(dir_text, SR)

   
        #128 hu, 64 batchsize, depth: varia

        checkpoint_path = cur_dir + "\\Python\\checkpoints\\dpt" + str(DEPTH) + ".ckpt"
        #DEPTH 25:  model-epoch=21-Total_LOSS= 0.2079
        #model-epoch=33-Total_LOSS= 0.2201
        #DEPTH 50:  model-epoch=06-Total_LOSS= 0.2272
        #DEPTH 100: model-epoch=04-Total_LOSS= 0.2130
        #DEPTH 150: model-epoch=20-Total_LOSS= 0.2040




        scaler = load("Python\\scaler" + str(DEPTH) + ".joblib")
        


        #To transform the pitches of a sequence in the same way we did in the train: 
        #the scaler expectes a 1*100 (depth) feature vector to transform
        #Se we take the pitch, and repeat it 1*depth times, creating a row vector
        #For a sequence, we need sequence_len*depth
          

        model_input = utils.modelInputDef(input_sequence_pitch, input_sequence_dur, input_sequence_step, DEPTH, scaler)
        myModel = model_test.load_from_checkpoint(checkpoint_path, map_location = DEVICE)
        myModel = myModel.to(DEVICE)
        original_len = model_input.shape[1]
        new_input = torch.zeros((1, original_len, 3))
        txt = ""
       
    
        for i in range(0,LENGTH): 

            if i == 0 : 
                out= myModel(model_input)
                o1 = out[:, 0:128]        

                o1 = utils.logits_masking(o1)                
                o2 = torch.unsqueeze(out[:, -2], 1)        
                o3  = torch.unsqueeze(out[:,-1], 1)

                predicted_pitch_midi = torch.multinomial(torch.softmax(o1/TEMPERATURE, dim=1),1)
                pitch_transformed = utils.pitchTransform(predicted_pitch_midi, scaler, DEPTH)

                predicted_pitch = torch.unsqueeze(torch.unsqueeze(pitch_transformed,0),0)
                new_input = utils.updateInputFixedLen(model_input, predicted_pitch, o2.detach(), o3.detach())
                txt = str(predicted_pitch_midi[0][0].item()) + ":" + str(o2[0][0].item()) + ":" + str(o3[0][0].item()) + "\n"
                n = muspy.Note( int(notes_mus[-1].time + SR*o3[0][0].item()), int(predicted_pitch_midi[0][0].item()), int(SR*o2[0][0].item()) )
                notes_mus.append(n)
            else: 
                out= myModel(new_input)
                o1 = out[:, 0:128]        
                o1 = utils.logits_masking(o1)
                o2 = torch.unsqueeze(out[:, -2], 1)        
                o3  = torch.unsqueeze(out[:,-1], 1)               
                predicted_pitch_midi = torch.multinomial(torch.softmax(o1/TEMPERATURE, dim=1),1)
                pitch_transformed = utils.pitchTransform(predicted_pitch_midi, scaler, DEPTH)
                predicted_pitch = torch.unsqueeze(torch.unsqueeze(pitch_transformed,0),0)
                new_input = utils.updateInputFixedLen(new_input, predicted_pitch, o2.detach(), o3.detach())
                txt += str(predicted_pitch_midi[0][0].item()) + ":" + str(o2[0][0].item()) + ":" + str(o3[0][0].item()) + "\n"
                n = muspy.Note( int(notes_mus[-1].time + SR*o3[0][0].item()), int(predicted_pitch_midi[0][0].item()), int(SR*o2[0][0].item()) )
                notes_mus.append(n)
                #Item() get value of a scalar tensor
      

            

        infer = muspy.Track(notes = notes_mus)
        music_track = muspy.Music(tracks = [infer])
        print("Pitch class entropy of result", muspy.metrics.pitch_class_entropy(music_track))
        print("Scale consistency of result", muspy.metrics.scale_consistency(music_track))

        with open("Python\\Sequence.txt", "w") as file: 
            file.write(txt)

        #Sends zero when logic is finished and sequence is ready to be played
        utils.sendToClient(client_socket, "0")

        
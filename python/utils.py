import socket
import os #Deal with directories
import pretty_midi #Deal with midis
import torch #Deal with tensors
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def ServerToClientSocketSetup(): 
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", 12345))
    print("Communication Client (C++) to Server (Python, ME) established.\n")
    server_socket.listen()
    #Accept() is a blocking method, waiting for client to connect and communicate 
    client_socket, addr = server_socket.accept()
    return client_socket



def receivingFromClient(client_socket): 
   
    
    # If there is no data available to be received, it will block and wait until data is received 
    print("Receiving from client:\n")
    message = client_socket.recv(1024).decode()
    print(f"Received message: {message}")
    return message



def sendToClient(client_socket, data_to_send): 
    client_socket.send(data_to_send.encode())


def textProcessing(path):
    #try to open text for training 

    print("From text containing Juce's input midi:")
    substring1 = '\x00'
    substring2= '\n'
    notes = []
    lengths = []
    pauses = []

    print("Text path: ", path)
    with open(path, 'r') as f:
        contents = f.readlines()
        print("Rows content: ", contents)
    for i in range(0, len(contents)): 
        contents[i] = contents[i].replace(substring1, '')
        contents[i] = contents[i].replace(substring2, '')
        contents[i] = contents[i].replace(' ', '')


    for i in range(0, len(contents)): 
        note, length, pause = contents[i].split(':')

        notes.append(float(note))
        lengths.append(float(length))
        pauses.append(float(pause))

    print("Erasing train content after reading has finished")

    open(path, "w").close()
    notes = np.asarray(notes)
    lengths = torch.as_tensor(lengths)
    pauses = torch.as_tensor(pauses)
    #notes, lengths, pauses : depth*1 array - like structures
    return notes, lengths, pauses
        

def MaestroToTensor(files): 

    directory_contents = os.listdir(files)



    notes_arr = []
    duration = []
    step = []
    
    for elem in directory_contents:
        
        midi_file = pretty_midi.PrettyMIDI(files + "\\" + elem)
        instrument = midi_file.instruments[0]
        #instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        #print("Instrument name: ", instrument_name)

        notes = instrument.notes
        #Sorting notes based on start time. 
        sorted_notes = sorted(notes,  key=lambda n: n.start)
        previous_start = 0 
        for n in sorted_notes :
            #Pitch extraction from each message
            notes_arr.append(n.pitch)
            #Duration of each note in seconds
            d = n.end - n.start 
            duration.append(d)
 
            if(n != instrument.notes[0]): 
                p = n.start - previous_start
                step.append(p)
            else: 
                #if first note, note on difference is 0. 
                step.append(0)
        
            previous_start = n.start    

    tensor_input = torch.tensor((notes_arr, duration, step))

    return tensor_input
    #Tensor of the shape : NXL, N = nb of midi features (pitch [0], duration [1], step[2]), L = nb of midi messages analyzed. 
  

def datasetCreation(tensor, depth): 
    #Nb of sequences: total number of cols - the depth 
    
    nb_sequences = tensor.shape[1]-depth

    train = torch.empty([nb_sequences, depth, 3])
    target = torch.empty([nb_sequences, 3])

    for i in range(0,tensor.shape[1]-depth): 
        
        data_row = tensor[:, i : i + depth]         
        
      
        data_row = torch.transpose(data_row, 0, 1)  
        train[i, :, :] = data_row 
        target[i, :] = tensor[:, i + depth]

    
        #Forse giusto? buh da verificare... puntini puntini....!!!
    scaler = MinMaxScaler((0, 1))
    scaler.fit(train[:,:,0])
    train[:,:,0] =  torch.from_numpy(scaler.transform(train[:,:,0]))
    dump(scaler, "scaler.joblib")

    
    return train, target, nb_sequences



def modelInputDef(input_sequence_pitch, input_sequence_dur, input_sequence_step, depth, scaler):
    #From 3 array like structures (issued from text written by juce). Each array like float contains pitch (np array), duration (tensor), step (tensor)
    #Function that creates an input sequence for the model of the shape 1xdepthx3

    input_sequence_pitch = np.expand_dims(input_sequence_pitch,1)

    model_input = torch.zeros((len(input_sequence_pitch), 3), device = DEVICE)

    in_2d = np.zeros((len(input_sequence_pitch), depth))
    for i in range(0,in_2d.shape[0]): 
        in_2d[i,:].fill(input_sequence_pitch[i,0])
    #           each row is scaler.n_features_in_ long!! 
    #in_2d is [[50, 50, 50 ..], [51, 51, 51....]....]
    new_in = scaler.transform(in_2d)
    print(new_in.shape)
    pitch_in_seq =torch.from_numpy(new_in[:, 0])
    print(pitch_in_seq.shape)
    print(model_input[:,0].shape)
    model_input[:,0] = pitch_in_seq 
    model_input[:,1] = input_sequence_dur
    model_input[:,2] = input_sequence_step
    print(model_input.shape)
    print(model_input)
    test = torch.unsqueeze(model_input, 0)
    return test



def updateInput(model_in, new_pitch, new_dur, new_step):
    #iteratively adds an entry to the three dimensions of the input tensor
    #Function discarded. 
    original_len = model_in[0,:].shape[0]

    updated_input = torch.empty(1, original_len + 1, 3, device = DEVICE)
    updated_input[:,0:original_len, :] = model_in
    #updated_input = torch.empty(1, model_in[0,:].shape[0], 3)
    #updated_input[0, :, 0] = tensor_slide(new_pitch, model_in[0, :, 0])
    #updated_input[0, :, 1] = tensor_slide(new_dur, model_in[0, :, 1])
    #updated_input[0, :, 2] = tensor_slide(new_step, model_in[0, :, 2])
    updated_input[0, -1, 0] = new_pitch
    updated_input[0, -1, 1] = new_dur
    updated_input[0, -1, 2] = new_step
        
        
    return updated_input



def updateInputFixedLen(model_in, new_pitch, new_dur, new_step):
    
    #Mantains the input to the model to the same initial length
    
    for i in range(model_in.shape[1] - 1):
         
        model_in[0][i][0] = model_in[0][i+1][0]
        model_in[0][i][1] = model_in[0][i+1][1]
        model_in[0][i][2] = model_in[0][i+1][2]

# Insert the new value at the end
    model_in[0][-1][0] = new_pitch
    model_in[0][-1][1] = new_dur
    model_in[0][-1][2] = new_step

    
    print(model_in)
    
        
    return model_in



def pitchTransform(val, scaler, depth): 
    #Used to re-transform the predicted pitch to the original range (0-127)

    in_2d = np.zeros((1, depth))
    in_2d[0,:].fill(val)    
    new_in = scaler.transform(in_2d)
    print("Scaled pitch", new_in[0][0])
    return torch.tensor(new_in[0,0], dtype = torch.float64)


# Computes a penalty for negative predictions.
def mse_with_positive_pressure(y_true, y_pred):
   
    mse = (y_true - y_pred)**2
  
    positive_pressure =10 * torch.relu(-y_pred)

    return torch.mean(mse + positive_pressure)



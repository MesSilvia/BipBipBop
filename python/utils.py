import socket
import os #Deal with directories
import pretty_midi #Deal with midis
import torch #Deal with tensors
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import muspy

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


def textProcessing(path, sr):
    #try to open text for training 

    substring1 = '\x00'
    substring2= '\n'
    notes = []
    lengths = []
    pauses = []
    notes_muspy = []

    with open(path, 'r') as f:
        contents = f.readlines()

    for i in range(0, len(contents)): 
        contents[i] = contents[i].replace(substring1, '')
        contents[i] = contents[i].replace(substring2, '')
        contents[i] = contents[i].replace(' ', '')

    for i in range(0, len(contents)): 
        
        note, length, pause = contents[i].split(':')
        notes.append(float(note))
        lengths.append(float(length))
        pauses.append(float(pause))
        if(i==0): 
            n = muspy.Note(0, int(note), int(float(length)*sr))
        else: 
            n = muspy.Note(((pauses[i-1] + float(pause))*sr), int(note), int(float(length)*sr))
        notes_muspy.append(n)
        

    open(path, "w").close()
    notes = np.asarray(notes)
    lengths = torch.as_tensor(lengths)
    pauses = torch.as_tensor(pauses)
    return notes, lengths, pauses, notes_muspy
        

def MaestroToTensor(files): 

    directory_contents = os.listdir(files)
    notes_arr = []
    duration = []
    step = []
    pce = [] #Pitch class entropy
    sc = [] #Scale consistency
    for folder in directory_contents:
        #For all folders in the directory 
        for elem in os.listdir(files + "\\" + folder)[:50]:
            #For all (up to 10th) files in each folder             
            midi_file = pretty_midi.PrettyMIDI(files + "\\" + folder + "\\" + elem)
            music_obj = muspy.read(files + "\\" + folder + "\\" + elem)
            pce.append(muspy.metrics.pitch_class_entropy(music_obj))
            sc.append(muspy.metrics.scale_consistency(music_obj))
            instrument = midi_file.instruments[0]
           
            notes = instrument.notes
           
            sorted_notes = sorted(notes,  key=lambda n: n.start)
            previous_start = 0 
            for n in sorted_notes :
              
                notes_arr.append(n.pitch)
                d = n.end - n.start 
                duration.append(d)
    
                if(n != instrument.notes[0]): 
                    p = n.start - previous_start
                    step.append(p)
                else:  
                    step.append(0)

                previous_start = n.start   
            
   
    tensor_input = torch.tensor((notes_arr, duration, step))
    print("Dataset average pitch class entropy", sum(pce)/len(pce))
    print("Dataset average scale consistency", sum(sc)/len(sc))
    
    return tensor_input
  

def datasetCreation(tensor, depth): 

    
    nb_sequences = tensor.shape[1]-depth

    train = torch.empty([nb_sequences, depth, 3])
    target = torch.empty([nb_sequences, 3])

    for i in range(0,tensor.shape[1]-depth): 
        
        data_row = tensor[:, i : i + depth]         
        
      
        data_row = torch.transpose(data_row, 0, 1)  
        train[i, :, :] = data_row 
        target[i, :] = tensor[:, i + depth]

    

    scaler = MinMaxScaler((0, 1))
    scaler.fit(train[:,:,0])
    train[:,:,0] =  torch.from_numpy(scaler.transform(train[:,:,0]))
    dump(scaler, "scaler" + str(depth) + ".joblib")

    return train, target, nb_sequences

#Function to correctly apply scaler object. Scaler waits for 2-d input, wide as the selected depth. 
def scaling(array_like, scaler, depth): 
    in_2d = np.zeros((len(array_like), depth))

    for i in range(0,in_2d.shape[0]): 
        in_2d[i,:].fill(array_like[i,0])
    
    new_in = scaler.transform(in_2d)

    scaled =torch.from_numpy(new_in[:, 0])
    return scaled 


def modelInputDef(input_sequence_pitch, input_sequence_dur, input_sequence_step, depth, scaler):
    #From 3 array like structures (issued from text written by juce). Each array like float contains pitch (np array), duration (tensor), step (tensor)
    #Function that creates an input sequence for the model of the shape 1xdepthx3

    input_sequence_pitch = np.expand_dims(input_sequence_pitch,1)

    model_input = torch.zeros((len(input_sequence_pitch), 3), device = DEVICE)

    in_2d = np.zeros((len(input_sequence_pitch), depth))
    for i in range(0,in_2d.shape[0]): 
        in_2d[i,:].fill(input_sequence_pitch[i,0])
  
    new_in = scaler.transform(in_2d)

    pitch_in_seq =torch.from_numpy(new_in[:, 0])

    model_input[:,0] = pitch_in_seq 
    model_input[:,1] = input_sequence_dur
    model_input[:,2] = input_sequence_step
 
    test = torch.unsqueeze(model_input, 0)
    return test


    #Function discarded. Input dinamycally grows. 
    #iteratively adds an entry to the three dimensions of the input tensor
def updateInput(model_in, new_pitch, new_dur, new_step):


    original_len = model_in[0,:].shape[0]

    updated_input = torch.empty(1, original_len + 1, 3, device = DEVICE)
    updated_input[:,0:original_len, :] = model_in
    updated_input[0, -1, 0] = new_pitch
    updated_input[0, -1, 1] = new_dur
    updated_input[0, -1, 2] = new_step
        
        
    return updated_input



def updateInputFixedLen(model_in, new_pitch, new_dur, new_step):
    
    #Mantains the input to the model to the same initial length
    #And adds the output as last element of the input, discarding the first value (FIFO)
    
    for i in range(model_in.shape[1] - 1):
         
        model_in[0][i][0] = model_in[0][i+1][0]
        model_in[0][i][1] = model_in[0][i+1][1]
        model_in[0][i][2] = model_in[0][i+1][2]

# Insert the new value at the end
    model_in[0][-1][0] = new_pitch
    model_in[0][-1][1] = new_dur
    model_in[0][-1][2] = new_step


    
        
    return model_in



def pitchTransform(val, scaler, depth): 
    #Scaling again only pdf sampling outcome, to be fed back to the model
    in_2d = np.zeros((1, depth))
    in_2d[0,:].fill(val)    
    new_in = scaler.transform(in_2d)

    return torch.tensor(new_in[0,0], dtype = torch.float64)


# Computes a penalty for negative predictions.
def mse_with_positive_pressure(y_true, y_pred):
   
    mse = (y_true - y_pred)**2
  
    positive_pressure =10 * torch.relu(-y_pred)

    return torch.mean(mse + positive_pressure)

#Discarded
def try_prob_conditioning(in_seq, p_out, scaler, depth): 

    expanded_input_sequence = np.expand_dims((in_seq.cpu()).numpy() ,1)

    in_2d = np.zeros((expanded_input_sequence.shape[0], depth))
    for i in range(0,in_2d.shape[0]): 
       in_2d[i,:].fill(expanded_input_sequence[i,0])

    new_in = scaler.inverse_transform(in_2d)

    pitch_in_seq =torch.from_numpy(new_in[:, 0])
    print(pitch_in_seq)
    input_prob = torch.zeros((1, 128), device = DEVICE )
    input_prob = torch.add(input_prob, 1)

    for index in range(0, input_prob.shape[1]): 
        for elem in pitch_in_seq: 
            if index == int(elem): 
                input_prob[0][index] = 8

    print(input_prob)
    print(input_prob*p_out)
    return input_prob * p_out


#In this function, logits that have negative values are set to a big negative value ( -100 ). This ensures that when 
#logits tensor will be passed to softmax, uncommon values become almost impossible when sampling the pdf. 
#Generation process stability benefits a lot from this process, preventing unlikely values (such as very low notes) to be
#sampled from output probability density function and thus re-entering in the model as input. 
def logits_masking(tens): 

    for index in range(0, tens.shape[1]): 
        if tens[0][index].item()  < 0: 
            tens[0][index] = -100
    
    return tens
        

/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <climits>
#include <algorithm>
#include <iterator>

//==============================================================================
NewProjectAudioProcessor::NewProjectAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
     )
   
#endif
{


    //SOCKET COMMUNICATION

    if (WSAStartup(MAKEWORD(2, 2), &Wsa) != 0) {
        DBG("Failed to initialize Winsock");
    }


    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(server_port);
    serverAddr.sin_addr.s_addr = inet_addr(server_ip);

    // Create socket
    clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == INVALID_SOCKET) {
        WSACleanup();
        DBG("Failed to create socket");
    }

    //Connect to server


    for (auto i = 0; i < 127; ++i) {
        noteOffTimes[i] = 0;
        noteOnTimes[i] = 0;
        noteOnTimes_pb[i] = 0; 
        noteOffTimes_pb[i] = 0;
    }


    for (auto i = 0; i < 127; ++i) {
        upToDate[i] = false; 

    }

    CollectionText.open("Collection.txt");
    CollectionText.close(); 

}

NewProjectAudioProcessor::~NewProjectAudioProcessor()
{
    CollectionText.open("Collection.txt");
    CollectionText.close();
  
}

//==============================================================================
const juce::String NewProjectAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool NewProjectAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool NewProjectAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool NewProjectAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double NewProjectAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int NewProjectAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int NewProjectAudioProcessor::getCurrentProgram()
{
    return 0;
}

void NewProjectAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String NewProjectAudioProcessor::getProgramName (int index)
{
    return {};
}

void NewProjectAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

//==============================================================================
void NewProjectAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{

    synthAudioSource.prepareToPlay(samplesPerBlock, sampleRate);
    sr = sampleRate; 

    connect_sock();


    sendSocketClient(sr);
    if (receiveFromServer() == "2") {
    }

    // Use this method as the place to do any pre-playback
    // initialisation that you need..
}

void NewProjectAudioProcessor::releaseResources()
{

    close_connection(); 
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool NewProjectAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void NewProjectAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();


    if (midiToProcess.getNumEvents() > 0) {
        midiMessages.addEvents(midiToProcess, midiToProcess.getFirstEventTime(), midiToProcess.getLastEventTime() + 1, 0);
        midiToProcess.clear();
    }


    

  


    if (play) {
        filled = true; 
      
        for (int i = 0; i < notes_to_playback.size(); i++) {

            if (noteOn_times_pb_vect[i] != 0 && noteOn_times_pb_vect[i] >= buffernumsamples){
                
                noteOn_times_pb_vect[i] = noteOn_times_pb_vect[i] - buffernumsamples;

                
                if (noteOn_times_pb_vect[i] < buffernumsamples) {
                    
                    juce::MidiMessage msg1 = juce::MidiMessage::noteOn(1, notes_to_playback[i], 0.5f);
                    midiMessages.addEvent(msg1, 0);
                    noteOn_times_pb_vect[i] = 0;
                    kbdState.noteOn(1, notes_to_playback[i], 0.5);
                    

                }

            }

            if (noteOff_times_pb_vect[i] != 0 && (noteOff_times_pb_vect[i] >= buffernumsamples)) {
            
                noteOff_times_pb_vect[i] = noteOff_times_pb_vect[i] - buffernumsamples;

                if (noteOff_times_pb_vect[i] < buffernumsamples) {

                    juce::MidiMessage msg1 = juce::MidiMessage::noteOff(1, notes_to_playback[i], 0.5f);
                    midiMessages.addEvent(msg1, 0);
                    noteOff_times_pb_vect[i] = 0;
                    kbdState.noteOff(1, notes_to_playback[i], 0.5);

                }

            }

        }

    }
    else if (!play && filled) {

        for (int i = 0; i < 127; i++) {
        
            kbdState.noteOff(1, i, 0.5);         
        }
        midiMessages.clear(); 
        filled = false;
    }

    synthAudioSource.setMidiBuffer(midiMessages);

    // In case we have more outputs than inputs, this code clears any output
    // channels that didn't contain input data, (because these aren't
    // guaranteed to be empty - they may contain garbage).
    // This is here to avoid people getting sc
    // reaming feedback
    // when they first compile a plugin, but obviously you don't need to keep
    // this code if your algorithm always overwrites all the output channels.
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // This is the place where you'd normally do the guts of your plugin's
    // audio processing...
    // Make sure to reset the state if your inner loop is processing
    // the samples and the outer loop is handling the channels.
    // Alternatively, you can process the samples with the channels
    // interleaved by keeping the same state.
   


    auto audiochannelbuf = juce::AudioSourceChannelInfo(&buffer, 0, int(buffer.getNumSamples()));

    synthAudioSource.getNextAudioBlock(audiochannelbuf);

 

    for (const auto metadata : midiMessages) {
        
        auto message = metadata.getMessage();



        if (message.isNoteOn()) {
            if (collection) {

                noteOnTimes[message.getNoteNumber()] = elapsedSamples;

                if (!new_noteon) {

                    previous_noteon = elapsedSamples;                 
                    new_noteon = true;
                    distance_noteon = 0;
                }



            }            

        }
        if (message.isNoteOff()) {

            if (collection) {
                upToDate[message.getNoteNumber()] = true;
                noteOffTimes[message.getNoteNumber()] = elapsedSamples;
            }
           
        }
    }

    if (collection) {

        for (int i = 0; i < 127; i++) {
            if ( (upToDate[i]) ){
                if (noteOnTimes[i] == getMinArray(noteOnTimes, 127)) {

                    distance_noteon = noteOnTimes[i] - previous_noteon;                   
                    previous_noteon = noteOnTimes[i];
                    int diff = noteOffTimes[i] - noteOnTimes[i];
                    CollectionText << i << ":" << (1.0 / sr) * diff << ":" << (1.0 / sr) * distance_noteon << endl;
                    upToDate[i] = false;
                    noteOnTimes[i] = 0;
                    noteOffTimes[i] = 0;

                }                
            }
        }

    }

    buffernumsamples = buffer.getNumSamples(); 
   
    //Buffer 480 samples, 10 ms
    elapsedSamples += buffer.getNumSamples();


}






//==============================================================================
bool NewProjectAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* NewProjectAudioProcessor::createEditor()
{
    return new NewProjectAudioProcessorEditor (*this);
}

//==============================================================================
void NewProjectAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
}

void NewProjectAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
}

void NewProjectAudioProcessor::addMidi(juce::MidiMessage msg, int sampleOffset)
{


    midiToProcess.addEvent(msg, sampleOffset);
    

}

void NewProjectAudioProcessor::startMidiCollection()
{
    CollectionText.open("Collection.txt");
    
    collection = true;
     
}

void NewProjectAudioProcessor::stopMidiCollection()
{
    collection = false; 
    new_noteon = false; 

    CollectionText.close();
    
    fillPlayBackMidi();

    //Collecting notes from python inference. 

    
  
}

void NewProjectAudioProcessor::startPlay()
{
    play = true; 

}

void NewProjectAudioProcessor::stopPlay()
{
    play = false; 
    notes_to_playback.clear(); 
    noteOn_times_pb_vect.clear(); 
    noteOff_times_pb_vect.clear();
}



void NewProjectAudioProcessor::fillPlayBackMidi()
{
   //Reading the txt file provided by python inference


    std::vector<unsigned long> length_to_playback{};
    std::vector<unsigned long> step_to_playback{};
 
    DBG("READING FROM TXT GENERATED BY PYTHON INFERENCE");

    std::string filename = "../Python/Sequence.txt";
    std::ifstream inputFile(filename);

    std::string token; 
    std::string line; 

    

    int idx = 0; 
    while (std::getline(inputFile, line)) {
        std::istringstream ss(line); 

        while (std::getline(ss, token, ':')) {
            if (idx == 0) {
                notes_to_playback.push_back(std::stoi(token));
            }
            else if (idx == 1) {
                length_to_playback.push_back(unsigned long(abs(sr*std::stof(token))));
            }
            else if (idx == 2) {
                step_to_playback.push_back(unsigned long(abs(sr*std::stof(token))));
            }
            else {
                DBG("ERROR");
            }
            idx++;
        }

        idx = 0; 

    }

    unsigned long new_start = 0; 
    for (int i = 0; i < notes_to_playback.size(); i++) {
        
        if (i == 0) {
            noteOn_times_pb_vect.push_back(new_start); 
            noteOff_times_pb_vect.push_back(length_to_playback[i]); 
        }
        else {            
            new_start = new_start + step_to_playback[i]; 
            noteOn_times_pb_vect.push_back(new_start); 
            noteOff_times_pb_vect.push_back(new_start + length_to_playback[i]); 

        }

    } 

}

void NewProjectAudioProcessor::setDepth(int d)
{
    depth = d; 
}

unsigned long  NewProjectAudioProcessor::getMinArray(unsigned long arr[], int size)
{
    //min of array excluding zeros

    unsigned long minimum = ULONG_MAX; 
    for (int i = 0; i < size; i++) {
        if(arr[i] < minimum && arr[i] != 0){
            minimum = arr[i];
        } 
    }

    return minimum;

}



void NewProjectAudioProcessor::sendSocketClient(int m)
{

    DBG("SENDING TO PYTHON"); 

    DBG("INCOMING VALUE TO SEND:" << m); 
  

    std::string d = std::to_string(m);
    const char* message = d.c_str(); 
    send(clientSocket, message, strlen(message), 0);


}

std::string NewProjectAudioProcessor::receiveFromServer()
{

    // Receive data from the server
    DBG("RECEIVING FROM PYTHON"); 
    char buffer[1024] = {0};
    int bytes_received = recv(clientSocket, buffer, sizeof(buffer), 0);

    if (bytes_received == SOCKET_ERROR) {
        DBG("Error while receiving data");
    }
    else {
        buffer[bytes_received] = '\0';
        DBG("Received: " << buffer);
    }


    return std::string(buffer);




}



void NewProjectAudioProcessor::setMessage(juce::String mes)
{
    message = mes; 
}



juce::String NewProjectAudioProcessor::getMessage()
{
    return message; 
}

void NewProjectAudioProcessor::connect_sock()
{
    if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        DBG("Connection failed");
        closesocket(clientSocket);
        WSACleanup();
    }

}

void NewProjectAudioProcessor::close_connection()
{
    closesocket(clientSocket);
    WSACleanup();

}

int NewProjectAudioProcessor::getSr()
{
    return sr; 
}

void NewProjectAudioProcessor::setSequenceLen(int s)
{
    sequence_len = s; 
}

bool NewProjectAudioProcessor::getPlayState()
{
    return play; 

}

void NewProjectAudioProcessor::setModelTemperature(int t)
{
    temperature = t; 
    
}


//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NewProjectAudioProcessor();
}

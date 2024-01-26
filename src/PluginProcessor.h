/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

//#include <JuceHeader.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include "SynthAudioSource.cpp"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <winsock2.h>
#include <filesystem>
#include <cstring>
#include <cstdlib>
#pragma comment(lib, "ws2_32.lib")



using namespace std;


class python_communication {
public: 
    python_communication() {
        std::filesystem::path venv{ "..\\python\\.venv" };
      
        auto run = []() {
            
            const char* command_run = "python ..\\python\\run.py";
            //const char* command = "python ..\\python\\eval.py";
            int returnCodeRun = std::system(command_run);
            DBG("RETURN CODE ENV" << returnCodeRun); 
            DBG(std::filesystem::current_path().string());

        };



        std::thread thread_object_run(run);

         //SERVER (PYTHON SIDE) ->MUST<- BE CREATED BEFORE CLIENT OBVIOUSLY
         //THIS THREAD SHOULD SLEEP UNTIL WE ARE SURE THAT PYTHON 
         //CORRECTLY SET UP COMMUNICATION SERVER SIDE!!! 



        if (std::filesystem::exists(venv)) {
            //Time needed for communication set up 
            std::this_thread::sleep_for(std::chrono::seconds(20));
        }
        else {
            //If the directory doesn't exists, it means that the venv should be created 
            //And all packages installed. It may require some time. To be generous 
            //we make the thread sleep for 2 mins and 20 secs. 
            std::this_thread::sleep_for(std::chrono::seconds(140));
        }

        
        thread_object_run.detach();
       
        DBG(" -------------- COMMUNICATION WITH PYTHON SETUP OK -------------------");
    }

};  


//==============================================================================
/**
*/
class NewProjectAudioProcessor  : public juce::AudioProcessor
                            #if JucePlugin_Enable_ARA
                             , public juce::AudioProcessorARAExtension
                            #endif
{
public:
    //==============================================================================
    NewProjectAudioProcessor();
    ~NewProjectAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    /** add some midi to be played at the sent sample offset*/
    void addMidi(juce::MidiMessage msg, int sampleOffset);


    //PlayBack
    void startPlay();
    void stopPlay();
    void fillPlayBackMidi();
    
    //Collection
    unsigned long getMinArray(unsigned long arr[], int size);
    void startMidiCollection(); 
    void stopMidiCollection(); 



    //Socked-based Communication
    std::string receiveFromServer();
    void setMessage(juce::String mes); 
    juce::String getMessage();
    void sendSocketClient(int m);
    void connect_sock(); 
    void close_connection(); 
    //SR retrieving to be called in plugin editor to exchanging it with Python 
    int getSr(); 

    //Hold internal representation of data exchanged with python for time modelling 
    void setSequenceLen(int s);
    bool getPlayState(); 
    void setModelTemperature(int t); 
    void setDepth(int d); 

    juce::MidiKeyboardState kbdState;


  


private:
    //==============================================================================

    python_communication p;  


    juce::MidiBuffer midiToProcess;
    unsigned long buffernumsamples = 0; 
    //Synth
    int sr = 0;

    //juce to python 
    bool collection = false; 

    //playback
    bool play = false; 
    
    std::vector<int> notes_to_playback{};
    std::vector<unsigned long> noteOn_times_pb_vect{};
    std::vector<unsigned long> noteOff_times_pb_vect{};
    unsigned long noteOffTimes_pb[127];
    unsigned long noteOnTimes_pb[127];
    int sequence_len = 0; 
    int temperature = 0; 
    bool filled = false;
    //time modeling 
    unsigned long noteOffTimes[127];
    unsigned long noteOnTimes[127];
    unsigned long elapsedSamples;
    bool upToDate[127];

    unsigned long previous_noteon = 0; 
    unsigned long distance_noteon = 0; 
    bool new_noteon = false; 
    
    //NN 
    ofstream CollectionText;
    int depth = 0; 

  

    //Polyphony
    SynthAudioSource synthAudioSource;


    //Messages: 
    juce::String message; 
    //Communication 
    const char* server_ip = "127.0.0.1";
    const int server_port = 12345;
    sockaddr_in serverAddr;
    SOCKET clientSocket; 
    WSADATA Wsa; 

    
    //Note On/Off Playback visualization 
    //juce::AudioProcessorValueTreeState parameters;
    int curr_note_on = 0; 
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NewProjectAudioProcessor)
};

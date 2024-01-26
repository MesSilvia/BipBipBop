/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"


//==============================================================================
NewProjectAudioProcessorEditor::NewProjectAudioProcessorEditor (NewProjectAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p), miniPianoKbd{p.kbdState, juce::MidiKeyboardComponent::horizontalKeyboard}, depthWindow{juce::String("Set Depth"), juce::String("Set depth"), juce::AlertWindow::NoIcon, this}
{

    setSize (700, 400);
    //KBD can be played with keyboard
    miniPianoKbd.setWantsKeyboardFocus(true);
  
    //Listeners
    (p.kbdState).addListener(this);
    trainButton.addListener(this); 
    playButton.addListener(this); 
    temperature_slider.addListener(this);


    //Modal loop, alert window
    depthWindow.setMessage("Set depth of the model,\n length (in number of notes)\n of the sequence to generate\n and temperature parameter:");
    depthWindow.addComboBox("Depth", s, "Depth");
    depthWindow.addTextEditor("Length", "0", "Length");
    depthWindow.addButton("Ok", 1); 


    temperature_label.setJustificationType(juce::Justification::centred);
    temperature_label.setText("Temperature", juce::NotificationType::dontSendNotification);
    depthWindow.addCustomComponent(&temperature_label);

    temperature_slider.setRange(1.0, 3.0, 0.01);
    temperature_slider.setColour(juce::Slider::backgroundColourId, juce::Colours::antiquewhite);
    depthWindow.addCustomComponent(&temperature_slider); 



    //Text
    //message.setText(std::filesystem::current_path().string(), juce::NotificationType::dontSendNotification);
    message.setText("Bip Bip Bop:\n a neural midi continuator", juce::NotificationType::dontSendNotification); 
    message.setJustificationType(juce::Justification::centred); 

    
 
    trainButton.setButtonText("Continuation Settings"); 
    playButton.setButtonText("Play"); 

    //Buttons toggling logic: 
    trainButton.setClickingTogglesState(true); 
    
    //Graphics
    trainButton.setColour(juce::TextButton::buttonOnColourId, juce::Colours::green);
    playButton.setColour(juce::TextButton::buttonOnColourId, juce::Colours::red);


    //Visibility
    addAndMakeVisible(miniPianoKbd);
    addAndMakeVisible(trainButton);
    addAndMakeVisible(playButton);
    addAndMakeVisible(message); 
    depthWindow.addAndMakeVisible(temperature_slider);
    depthWindow.addAndMakeVisible(temperature_label);

    //NEW - SR COMMUNICATION 




}

NewProjectAudioProcessorEditor::~NewProjectAudioProcessorEditor()
{
}

//==============================================================================
void NewProjectAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    g.fillAll(juce::Colours::transparentBlack); 
    g.setColour (juce::Colours::white);
    //Graphics
    auto centralArea = getLocalBounds().toFloat().reduced(10.0f);
    g.drawRoundedRectangle(centralArea, 5.0f, 3.0f);
    auto centralAreaForMessage = getLocalBounds().toFloat().reduced(160.0f); 
    g.drawRoundedRectangle(centralAreaForMessage, 5.0f, 3.0f);

    g.setFont (15.0f);
    miniPianoKbd.grabKeyboardFocus();


}

void NewProjectAudioProcessorEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..

    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
    float rowHeight = getHeight() / 5;
    float colWidth = getWidth() / 3;
    float row = 4;
    float offset = 20; 

    miniPianoKbd.setBounds(0+ offset, rowHeight * row, getWidth()-2*offset, rowHeight-offset);
    trainButton.setBounds(0+offset, 0+offset, colWidth-offset, rowHeight-offset); 
    playButton.setBounds(getWidth() - colWidth, 0 + offset, colWidth - offset, rowHeight - offset); 
    message.setBounds(getWidth() / 2 - colWidth/2 , getHeight() / 2 - rowHeight/2, colWidth, rowHeight);
    temperature_label.setBounds(0, getHeight() / 2 + 3 * getHeight() / 10, depthWindow.getWidth(), getHeight() / 20);
    temperature_slider.setBounds( 0, getHeight()/2 + getHeight()/6 , depthWindow.getWidth(), getHeight() / 10);
    
}


void NewProjectAudioProcessorEditor::sliderValueChanged(juce::Slider* slider)
{
}

void NewProjectAudioProcessorEditor::buttonClicked(juce::Button* btn)
{


    //Toggling logic
    if (btn == &trainButton) {
        
        if (trainButton.getToggleState() && !ready_train && !ready_play) {
            //State 1
            trainButton.setClickingTogglesState(false); 
            playButton.setClickingTogglesState(false);
            while (lgt <= 0) {
                int res = depthWindow.runModalLoop();
                if (res == 1) {
                    juce::String depth = (depthWindow.getComboBoxComponent("Depth"))->getText();  //penso che è juce::String
                    dpt = depth.juce::String::getIntValue();
                    juce::String length = depthWindow.getTextEditorContents("Length");  //penso che è juce::String
                    
                    lgt = length.juce::String::getIntValue();
                    message.setText("Insert a valid value for length !", juce::dontSendNotification);

                    
                    
                    juce::Slider* mySlider= dynamic_cast<juce::Slider*> (depthWindow.getCustomComponent(1)); 
                    temperature = mySlider->getValue()*100; 

                    depthWindow.exitModalState();
                    audioProcessor.setDepth(dpt);
                    audioProcessor.setSequenceLen(lgt); 
                    audioProcessor.setModelTemperature(temperature); 
                    depthWindow.setVisible(false);
                }
            }
            audioProcessor.startMidiCollection();            
            message.setText("Collecting midi messages\n as input to the model...", juce::dontSendNotification);
            ready_train = true;
           
        }else if (trainButton.getToggleState() && ready_train && !ready_play) {         
            //State 2
            std::ifstream CollectionInFile("Collection.txt");
            CollectionInFile.seekg(0, std::ios::end);
            if (!CollectionInFile.tellg() == 0) {
                trainButton.setToggleState(false, true);
                //trainButton.setClickingTogglesState(true);
            }
            else {
                message.setText("Please play something to continue and click Continuation Settings button. ", juce::dontSendNotification);
            }
            //Forced by program
        }else if (!trainButton.getToggleState() && ready_train && !ready_play) {
            //State 3

            
            //Settings have been correctly inserted and the user has played something. 
            //Communicates settings and input notes to python for inference. 
            audioProcessor.sendSocketClient(dpt);

            //Blocking operation, to be sure that python correctly received the settings. 
            if (audioProcessor.receiveFromServer() == "1") {
            }
            audioProcessor.sendSocketClient(lgt);
            if (audioProcessor.receiveFromServer() == "1") {
            }
            audioProcessor.sendSocketClient(temperature);
            if (audioProcessor.receiveFromServer() == "1") {
            }
            
            //client.rcv is a blocking operation. If message is 0, python is done with inference.
            //Messages can be sent back to Juce for playing the midi continuation. 
            if (audioProcessor.receiveFromServer() == "0") {
                audioProcessor.stopMidiCollection();
            }            
            //Here we need to know when training is over from python, so that we can go on with inference
            playButton.setClickingTogglesState(true);
            ready_train = false; 
            ready_play = true; 
            message.setText("Press play to hear the continuation.", juce::dontSendNotification);

        }
        else if (!trainButton.getToggleState() && !ready_train && ready_play) {
            //State 4
            message.setText("To continue with another collection, hear the existing continuation first (Click Play)!", juce::dontSendNotification);
        }

    }






    if (btn == &playButton) {

        if (playButton.getToggleState() && ready_play && !ready_train) {
            //State 5
            //trainButton.setClickingTogglesState(false);
            audioProcessor.startPlay();
            message.setText("Playing...", juce::dontSendNotification);
            //sendSocketClient();
        }
        else if (!playButton.getToggleState() && !ready_train && !ready_play) {
            //State 6
            message.setText("Please, provide some notes as model input first following instructions (Click Continuation Settings).", juce::dontSendNotification);
        }
        else if (!playButton.getToggleState() && ready_train && !ready_play) {
            //State 7
            message.setText("Please, play something to finish training and click again Continuation Settings.", juce::dontSendNotification);
        }

        else if (!playButton.getToggleState() && ready_play) {
            //State 8 
            audioProcessor.stopPlay();
            trainButton.setClickingTogglesState(true);
            playButton.setClickingTogglesState(false);
            ready_play = false; 
            lgt = 0; 
            message.setText("Done! Hope you liked it. Provide new continuation settings to proceede with a new midi continuation. ", juce::dontSendNotification);
        }
    }
}

void NewProjectAudioProcessorEditor::handleNoteOn(juce::MidiKeyboardState* source, int midiChannel, int midiNoteNumber, float velocity)
{

    if (!audioProcessor.getPlayState()) {
        juce::MidiMessage msg1 = juce::MidiMessage::noteOn(midiChannel, midiNoteNumber, velocity);
        audioProcessor.addMidi(msg1, 0);

    }
    

}

void NewProjectAudioProcessorEditor::handleNoteOff(juce::MidiKeyboardState* source, int midiChannel, int midiNoteNumber, float velocity)
{

    if (!audioProcessor.getPlayState()) {
        juce::MidiMessage msg2 = juce::MidiMessage::noteOff(midiChannel, midiNoteNumber, velocity);
        audioProcessor.addMidi(msg2, 0);

    }
    
}



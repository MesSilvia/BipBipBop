/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

//#include <JuceHeader.h>
#include "PluginProcessor.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_utils/juce_audio_utils.h>


//

//==============================================================================
/**
*/
class NewProjectAudioProcessorEditor  : public juce::AudioProcessorEditor, 
                                        //listen to buttons
                                        public juce::Button::Listener, 
                                        //listen to sliders
                                        public juce::Slider::Listener, 
                                        //listen to piano keyboard 
                                        public juce::MidiKeyboardState::Listener
                                        //label for messages
                              
{
public:
    NewProjectAudioProcessorEditor (NewProjectAudioProcessor&);
    ~NewProjectAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;



    void sliderValueChanged(juce::Slider* slider) override;
    void buttonClicked(juce::Button* btn) override;

    void handleNoteOn(juce::MidiKeyboardState* source, int midiChannel, int midiNoteNumber, float velocity) override;
    // from MidiKeyboardState
    void handleNoteOff(juce::MidiKeyboardState* source, int midiChannel, int midiNoteNumber, float velocity) override;





private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.


    //juce::MidiKeyboardState kbdState;
    juce::MidiKeyboardComponent miniPianoKbd;
    juce::TextButton trainButton;
    juce::TextButton playButton;
    juce::Label message; 
    juce::Label temperature_label; 
    juce::AlertWindow depthWindow; 
    juce::StringArray s = juce::StringArray({ "25", "50", "100", "150" });
    juce::Slider temperature_slider;
    int dpt = 0; 
    int lgt = 0; 
    int temperature = 0; 

    bool ready_train = false;
    bool ready_play = false; 


    // This reference is provideas a quick way for your editor to
    // access the processor object that created it.
    NewProjectAudioProcessor& audioProcessor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NewProjectAudioProcessorEditor)
};

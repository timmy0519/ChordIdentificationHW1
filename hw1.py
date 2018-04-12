# -*- coding: utf-8 -*-
"""
Created on Fri Mar 02 00:43:05 2018
Example code 3: chromagram and basic chord recognition

@author: lisu
"""
import numpy as np
import librosa.feature
import scipy.io.wavfile as wav
import scipy.signal
import matplotlib 
import matplotlib.pyplot as plt

font = {'family' : 'sans-serif', 'sans-serif':'Arial', 'size'   : 18}
matplotlib.rc('font', **font)

# Generate major chord templates
Major_template = np.array([[1,0,0,0,1,0,0,1,0,0,0,0]])/np.sqrt(3.0)
# Generate monor chord templates
Minor_template = np.array([[1,0,0,1,0,0,0,1,0,0,0,0]])/np.sqrt(3.0)

Template = Major_template
for i in range(11):
    Template = np.append(Template, np.roll(Major_template, i+1), axis=0)    
for i in range(12):
    Template = np.append(Template, np.roll(Minor_template, i), axis=0)
 
#for debug
Key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',\
           'G', 'G#', 'A', 'A#', 'B',\
           'c', 'c#', 'd', 'd#', 'e', 'f', 'f#',\
           'g', 'g#', 'a', 'a#', 'b'] 
           
#manually change path to each genre and test each song           
for fileLen in range(100):
    path = 'genres/country/country.00{0:03}.au'.format(fileLen)
    x, fs = librosa.load(path, sr=None)
    ansKey = int(open('gtzan_key-master/gtzan_key/genres/country/country.00{0:03}.lerch.txt'.format(fileLen),'r').read())

    
    if x.dtype != 'float32': # deal with the case of integer-valued data
        x = np.float32(x/32767.)
    if x.ndim > 1: # I want to deal only with single-channel signal now
        x = np.mean(x, axis = 1)
    
    Chroma = librosa.feature.chroma_stft(y=x, sr=fs)
    Chroma = Chroma/np.sum(np.abs(Chroma)**2, axis=0)**(1./2)
    
    #Q1###
    GAMA = 100
    Chroma = np.log10(1+GAMA *np.abs(Chroma))
    #spectral smoothing
    Len = 10
    for i in range(Chroma.shape[1]):
        Chroma[:,i] = np.sum(Chroma[:,i- int(Len/2):int(i+Len/2)],axis = 1)/Len
    
    
    sumChroma =np.sum(Chroma,axis = 1)
    tonic = np.argmax(sumChroma)
    #substract mean x - x bar?
    sumChroma -= np.sum(sumChroma) /sumChroma.shape[0]
    
    MajorCof = np.dot(Template[tonic],sumChroma,)\
     / np.sqrt(np.sum(np.multiply(sumChroma,sumChroma)))
     
    MinorCof = np.dot(Template[tonic+12],sumChroma,)\
     / np.sqrt(np.sum(np.multiply(sumChroma,sumChroma)))
    if(MajorCof>MinorCof):
        print(tonic,Key[tonic]+" Major")
    else:
         print((tonic+12),Key[tonic+12]+" Minor")
         
    if(int(ansKey)>11):
        print('Ans: ',Key[int(ansKey)]+' Minor')
    else:
        print('Ans: ',Key[int(ansKey)]+' Major')
""" example 3 -> each frame's chord?  but in the homework's tutorial doesn't use this method
    Result = np.dot(Template, Chroma)
    #Result ->which chord match the most in every frame
    Chord = Result.argmax(axis=0)
    Chord = scipy.signal.medfilt(Chord, kernel_size = 21) # median filter as post-processing
    
    
    t = np.arange(Chord.shape[0])*512.0/fs # from frame index to time

plt.figure(1,figsize = (25,15))
plt.subplot(211)
plt.pcolormesh(t, range(13), Chroma, cmap='Purples')
plt.subplot(212)
plt.plot(t, Chord)
plt.xlim((0, t[-1]))
plt.xlabel("Time (s)")

plt.figure(2,figsize = (25,15))
plt.plot(t, Chord)
plt.xlim((0, t[-1]))
plt.yticks(range(24), ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#',\
           'G', 'G#', 'A', 'A#', 'B',\
           'c', 'c#', 'd', 'd#', 'e', 'f', 'f#',\
           'g', 'g#', 'a', 'a#', 'b'))
plt.xlabel("Time (s)")
"""
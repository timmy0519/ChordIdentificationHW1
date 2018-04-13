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
from collections import defaultdict

        
font = {'family' : 'sans-serif', 'sans-serif':'Arial', 'size'   : 18}
matplotlib.rc('font', **font)

Major_template = np.array([[1,0,0,0,6,1,0,3,0,1,0,0]])/np.sqrt(6.0)
# Generate monor chord templates
Minor_template = np.array([[1,0,0,6,0,1,0,3,1,0,0,0]])/np.sqrt(6.0)

Template = Major_template
for i in range(11):
    Template = np.append(Template, np.roll(Major_template, i+1), axis=0)    
for i in range(12):
    Template = np.append(Template, np.roll(Minor_template, i), axis=0)
    
def CalculateCof(tonic):
    MajorCof = np.dot(Template[tonic],sumChroma,)\
    / np.sum(sumChroma**2)**1/2
     
    MinorCof = np.dot(Template[(tonic+12)%24],sumChroma,)\
    / np.sum(sumChroma**2)**1/2
    return MajorCof,MinorCof 

#for debug
Key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',\
           'G', 'G#', 'A', 'A#', 'B',\
           'c', 'c#', 'd', 'd#', 'e', 'f', 'f#',\
           'g', 'g#', 'a', 'a#', 'b'] 
           
#manually change path to each genre and test each song
GENRE = 'country'           
for fileLen in range(100):
    path = 'genres/{0}/{1}.00{2:03}.au'.format(GENRE,GENRE,fileLen)
    x, fs = librosa.load(path, sr=None)
    ansKey = int(open('gtzan_key-master/gtzan_key/genres/{0}/{1}.00{2:03}.lerch.txt'.format(GENRE,GENRE,fileLen),'r').read())

    
    if x.dtype != 'float32': # deal with the case of integer-valued data
        x = np.float32(x/32767.)
    if x.ndim > 1: # I want to deal only with single-channel signal now
        x = np.mean(x, axis = 1)
    
    Chroma = librosa.feature.chroma_stft(y=x, sr=fs)
    Chroma = Chroma/np.sum(np.abs(Chroma)**2, axis=0)**(1./2)
    
    #shift down 
    Chroma[-1],Chroma[0:-1] = Chroma[0]*0.8 + Chroma[1]*0.2 , Chroma[1:]*0.8 + np.vstack((Chroma[2:],Chroma[0]))*0.2
    
    #Q1###
    GAMA = 100
    Chroma = np.log10(1+GAMA *np.abs(Chroma))
    #spectral smoothing
    Len = 2
    for i in range(Chroma.shape[1]):
        Chroma[:,i] = np.sum(Chroma[:,i- int(Len/2):int(i+Len/2)],axis = 1)/Len
    
    sumChroma =np.sum(Chroma,axis = 1)
    tonic = np.argmax(sumChroma)

    #substract mean x - x bar?
    sumChroma -= np.sum(sumChroma) /sumChroma.shape[0]
    toneTest = [tonic,(tonic+9)%12,(tonic+3)%12,(tonic+7)%12]
    
    dictCof = dict()
    
    for t in toneTest:        
        MajorCof ,MinorCof = CalculateCof(t)
        
        if(MajorCof>MinorCof):
            dictCof[t]= [0,MajorCof]
        else:
            dictCof[t] = [1,MinorCof]
    tryTonic = [0,-1] #[tonic,val]
    for key,val in dictCof.items():
        if(val[1]>tryTonic[1]):
            tryTonic = [key,val[1]]
    if(dictCof[tryTonic[0]][0] == 0):
        print(tryTonic[0],Key[tryTonic[0]]+"  Major")
    else:
         print((tryTonic[0]+12),Key[tryTonic[0]+12]+" Minor")
         
    if(int(ansKey)>11):
        print('Ans: ',Key[int(ansKey)]+' Minor\n')
    else:
        print('Ans: ',Key[int(ansKey)]+' Major\n')
        

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
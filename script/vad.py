'''
Filename: /home/richard/workspace/audio_tool_kit/vad.py
Path: /home/richard/workspace/audio_tool_kit
Created Date: Sunday, June 23rd 2024, 1:11:04 pm
Author: richard

Copyright (c) 2024
'''
import numpy as np
class vad(object):
    def __init__(self, signal, fs, frameDur, nbits, noise_threshold=-50, step=.5):
        self._signal_len = len(signal)
        if len(np.shape(signal)) != 1:
            signal = signal[:,0].flatten()
        self._frame_size = int(frameDur * fs)
        self._fs = fs
        
        self._step = step # measured in dB
        INT_MIN = float(2**(-(nbits-1)))
        self._nframe = len(signal) // self._frame_size
        self._ref_rms = np.zeros(np.shape(signal))
        
        s = np.reshape(signal[0:self._nframe*self._frame_size], (-1, self._frame_size))
        ms = np.sqrt(np.mean(s**2, axis=1))
        ms[ms < INT_MIN] = INT_MIN
        self._ref_rms = 20*np.log10(ms)
        self._smoothed_ref_rms = self._ref_rms
        self._activity_in_samples = np.zeros(np.shape(signal))
        
        for frameIndex in range(1, self._nframe):
            if self._smoothed_ref_rms[frameIndex] <= self._smoothed_ref_rms[frameIndex-1]:
                self._smoothed_ref_rms[frameIndex] = self._smoothed_ref_rms[frameIndex-1] - self._step
        self.activity_mask = (self._smoothed_ref_rms >= noise_threshold).flatten()*1
        self.noise_mask = self._smoothed_ref_rms < noise_threshold
        for fdx in range(self._nframe):
            self._activity_in_samples[fdx*self._frame_size:(fdx+1)*self._frame_size] = self.activity_mask[fdx]
    
    def get_active_mask(self, level="frame"):
        if level == "frame":
            return self.activity_mask
        elif level == "sample":
            return self._activity_in_samples
            
            
        
        
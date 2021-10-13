import random as rn
import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import wave
import time
from numba import njit


@njit()  # Used for CUDA implementation for python
def MyConvolve(x, h):
    np.flip(h)  # flipping the kernel
    M = len(x)
    N = len(h)
    C = np.zeros(M + N - 1)  # initializing result array
    temp = np.zeros(N)
    x = np.append(x, temp)  # zero padding only the right side
    for k in range(M + N - 1):  # to fill C array
        i = 0
        while 0 <= k-i and i < N:  # sliding dot product
            C[k] += h[i]*x[k-i]
            i += 1
    return C

# function used to write the wav file
def getWav(data,fname):  # lenght of wave file varies depending on sampling rate
    sps = 16000  # sampling rate
    freq = 440.0
    temp = data
    waveform = np.sin(2*np.pi*temp*freq/sps)
    waveform2 = waveform*0.3
    waveform3 = np.int16(waveform2 * 32767)
    write(fname, sps, waveform3)

# function used for reading wav file
def getAudio(fname):  # filename as parameter
    a = wave.open(fname)
    samplesa = a.getnframes()  # Used for sampling
    auda = a.readframes(samplesa)  # reading the sampled data
    audio_as_np_int16a = np.frombuffer(auda, dtype=np.int16)  # transfer data to numpy int array
    audio_as_np_float32a = audio_as_np_int16a.astype(np.float32)   # convert to float
    max_int = 2**15
    A = audio_as_np_float32a / max_int  # normalize to [0,1]
    return A


def first():
    a = []
    while True:
        N = int(input("Enter length of sequence greater than 10: "))  # User input
        if N > 10:  # Input soundness control
            break
        print("Must be greater than 10")
    for i in range(N):
        ran = round(rn.uniform(0, 100), 2)  # fill A with N random floats
        a.append(ran)
    A = np.array(a)  # convert to numpy array
    B = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    print(MyConvolve(A, B))
    temp = np.convolve(A,B)  # Comparing results to python's built in function
    print(temp)


def second():
    A = getAudio("sample_audio.wav")  # read both files
    B = getAudio("pink_noise.wav")
    final = MyConvolve(A, B)  # compute convolution
    getWav(final, "pinkNoise_sampleAudio.wav")  # write new file

    num_samples = 16000
    whitenoise = np.random.normal(0, 1, size=num_samples)  # creating random white noise
    finalb = MyConvolve(A, whitenoise)
    getWav(finalb, "whiteNoise_sampleAudio.wav")


first()
start = time.time()
second()
print("--- %s seconds ---" % (time.time() - start))

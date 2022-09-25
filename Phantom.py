# Load data
import mat73

def loadMatData():
    data = mat73.loadmat('data/data.mat')['data']

    elementsLocations = data['elementsLocations'] # waveform units
    elementsDelays = data['elementsDelays']
    elementsApodizations = data['elementsApodizations']
    basePulse = data['basePulse']
    numChannels = int(data['numChannels'])
    nt = int(data['nt'])
    dt = data['dt'] * 1e6
    dx = data['dx']
    nx = int(data['nx'])
    L = data['L']
    trans = data['RcvData']
    return elementsLocations, elementsDelays, elementsApodizations, basePulse, numChannels, nt, dt, dx, nx, L, trans

# create simulator

# Create pulses
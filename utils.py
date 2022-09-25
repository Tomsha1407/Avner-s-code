import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import torch.nn as nn

import DAS

class Grid:
    def __init__(self, Lx, Lz, nx, nz, nt, T):
        # Space
        self.Lx = Lx
        self.Lz = Lz
        self.nx = nx
        self.nz = nz
        self.dx = self.Lx / self.nx
        self.dz = self.Lz / self.nz

        # Time
        self.T = T
        self.nt = nt
        self.dt = float(self.T) / float(self.nt)
        self.CFL = 0.5

        self.device = torch.device("cuda")
        self.dtype = torch.float32

        # Number of operation (same as the size of the kernel)
        self.nop = 5

        self.PML_size = 10
        self.PML_max_damping = 0.1

        self.plotDas = False
        self.stdNoise = 0.0003
        self.factor = 0.95

        self.reconstructVelocity = True
        self.reconstructDensity = True
        self.reconstructDamping = False
        self.reconstructBeta = False

        self.mask = torch.tensor(1.0, dtype=self.dtype, device=self.device)


class Prob:
    def __init__(self, numChannels, f0, grid):
        # The number of channels in the probe
        self.numChannels = numChannels

        # Contains the pulse of each emitter as a function of the time. An numChannels x nt matrix
        self.pulses = getPulses(numChannels, f0, grid)
        self.base_pulse = self.pulses[0]

        # The location of the elements in the probe. An numChannels x 2 matrix
        self.stepSize = 1
        self.loc = getElementsLocations(numChannels, grid, self.stepSize)

        # The dominant frequency of the elements (Hz)
        self.f0 = f0

        # Lateral properties of the prob
        self.sizeLateral = 16
        self.lateralStride = 4
        self.numLaterals = int((self.numChannels - self.sizeLateral + 1) / self.lateralStride)


class PhysicalModel:
    def __init__(self, attenuationFactor, amplificationFactor, c0):
        # Due to losses, the energy of the propagating wave dissipates according to this attenuation factor
        self.attenuationFactor = attenuationFactor

        # Since more distant emitters will emit weaker, we rectify by re-weighting (Not physical)
        self.amplificationFactor = amplificationFactor

        self.c0 = c0

        self.vmax_velocity = c0 + 55
        self.vmin_velocity = c0 - 55
        self.vmin_density = 0.8
        self.vmax_density = 1.2
        self.vmin_damping = -1e-6
        self.vmax_damping =  1e-6
        self.vmin_beta = nlp2beta('water') - 5
        self.vmax_beta = nlp2beta('water') + 5


class Properties:
    def __init__(self):
        pass


def circleMask(x0, z0, Sigma, amplitude, grid):
    """
    Define an ellipse centered at x0 and z0, with a value "amplitude".
    :param x0: x coordinate of the center
    :param z0: z coordinate of the center
    :param Sigma: The ellipse matrix
    :param amplitude: The amplitude of the mask
    :param grid
    :return: A mask with value "amplitude" inside the ellipse, and 0 outside of it.
    """
    z, x = np.ogrid[-grid.nz / 2: grid.nz / 2, -grid.nx / 2: grid.nx / 2]
    x = torch.from_numpy(x).to(grid.dtype).to(grid.device)
    z = torch.from_numpy(z).to(grid.dtype).to(grid.device)
    SigmaInv = np.linalg.inv(np.array(Sigma) @ np.array(Sigma))
    mask = ((x - x0) ** 2 * SigmaInv[1, 1] + (z - z0) ** 2 * SigmaInv[0, 0] \
           + (x - x0) * (z - z0) * (SigmaInv[0, 1] + SigmaInv[1, 0])) <= 1
    return amplitude * mask


def getGTProperties(grid, prob, physicalModel):
    """
    :return: The GT properties of the material.
    """
    SigmaLiver, SigmaFat, liverLocation, fatLocation = getMasks()
    GTVelocities = torch.zeros([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device) + physicalModel.c0
    maskT1 = circleMask(x0=liverLocation[0], z0=liverLocation[1], Sigma=SigmaLiver, amplitude=90, grid=grid) #circleMask(x0=25, z0=-grid.nz / 2 + 105, r=15, amplitude=-50, grid=grid)
    maskT2 = circleMask(x0=fatLocation[0], z0=fatLocation[1], Sigma=SigmaFat, amplitude=-30, grid=grid) # circleMask(x0=-25, z0=-grid.nz / 2 + 80, r=20, amplitude=50, grid=grid)
    GTVelocities += maskT1 + maskT2
    GTVelocities = torch.from_numpy(scipy.ndimage.gaussian_filter(GTVelocities.to(dtype=torch.float32).detach().cpu().numpy(), sigma=1.5)).to(grid.dtype).to(grid.device)

    GTDensities = torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    GTDensities += circleMask(x0=liverLocation[0], z0=liverLocation[1], Sigma=SigmaLiver, amplitude=0.06, grid=grid)
    GTDensities += circleMask(x0=fatLocation[0], z0=fatLocation[1], Sigma=SigmaFat, amplitude=-0.05, grid=grid)
    GTDensities = torch.from_numpy(scipy.ndimage.gaussian_filter(GTDensities.to(dtype=torch.float32).detach().cpu().numpy(), sigma=1.8)).to(grid.dtype).to(grid.device)

    GTDamping = initializeDamping(grid, prob, physicalModel)
    GTDamping += circleMask(x0=liverLocation[0], z0=liverLocation[1], Sigma=SigmaLiver, amplitude=tissueDamping('liver', grid, prob) - tissueDamping('water', grid, prob), grid=grid)
    GTDamping += circleMask(x0=fatLocation[0], z0=fatLocation[1], Sigma=SigmaFat, amplitude=tissueDamping('fat', grid, prob) - tissueDamping('water', grid, prob), grid=grid)

    GTBeta = nlp2beta('water') * torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    GTBeta += circleMask(x0=liverLocation[0], z0=liverLocation[1], Sigma=SigmaLiver, amplitude=nlp2beta('liver') - nlp2beta('water'), grid=grid)
    GTBeta += circleMask(x0=fatLocation[0], z0=fatLocation[1], Sigma=SigmaFat, amplitude=nlp2beta('fat') - nlp2beta('water'), grid=grid)

    properties = Properties()
    properties.velocity = GTVelocities
    properties.density = GTDensities
    properties.damping = GTDamping
    properties.beta = GTBeta
    return properties


def getRefProperties(grid, prob, physicalModel):
    """
    :return: The reference properties.
    """
    properties = Properties()
    properties.velocity = torch.zeros([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device) + physicalModel.c0
    properties.density = torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    properties.damping = initializeDamping(grid, prob, physicalModel)
    properties.beta = 5 * torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    return properties


def initializeProperties(trans, transRef, GTProperties, grid, prob, physicalModel, DASInitialization):
    """
    Initialize the properties.
    The velocity can be initialized using the corresponding DAS.
    :param trans: The channel data obtain. Will be used to compute the DAS image.
    :param transRef: The channel data obtain from the reference material. Will be used to compute the DAS image.
    :param GTProperties: The GT properties of the material.
    :param DASInitialization: True if initialize with DAS.
    :return: The initialization properties.
    """
    if not grid.reconstructVelocity:
        initialVelocities = GTProperties.velocity.clone()
    elif DASInitialization:
        das = DAS.DAS(trans - transRef, grid, prob, physicalModel).detach().cpu().numpy()
        # das = np.load('das.npy')
        das = np.exp(das / np.max(das))
        das = das / np.sum(das)
        das = das - np.mean(das)
        das = torch.from_numpy(das)

        das *= 100
        das += physicalModel.c0

        # grid.das = das

        initialVelocities = das / das.std() * 40 + physicalModel.c0 #physicalModel.c0 + torch.zeros([grid.nz, grid.nx]) + 0 * (torch.rand([grid.nz, grid.nx]) - 0.5)
    else:
        initialVelocities = torch.zeros([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device) + physicalModel.c0 + 0 * (torch.rand([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device) - 0.5)

    if not grid.reconstructDensity:
        initialDensities = GTProperties.density.clone()
    else:
        initialDensities = torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)

    if not grid.reconstructDamping:
        initialDamping = GTProperties.damping.clone()
    else:
        initialDamping = initializeDamping(grid, prob, physicalModel)

    if not grid.reconstructBeta:
        initialBeta = GTProperties.beta.clone()
    else:
        initialBeta = nlp2beta('water') * torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)

    properties = Properties()
    properties.velocity = initialVelocities
    properties.density = initialDensities
    properties.damping = initialDamping
    properties.beta = initialBeta
    return properties


def initializeDamping(grid, prob, physicalModel):
    """
    Initialize the damping, including the PML layer.
    :return: The initialized damping/attenuation.
    """
    damping = tissueDamping('water', grid, prob) * torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    line_damping = ((torch.arange(0, grid.PML_size) / grid.PML_size).pow(2) * grid.PML_max_damping).to(grid.dtype).to(grid.device)
    line_damping = torch.tile(line_damping, (grid.nz, 1))
    damping[:grid.PML_size, :] += line_damping.flip(0, 1).transpose(0, 1)
    damping[-grid.PML_size:, :] += line_damping.transpose(0, 1)
    damping[:, :grid.PML_size] += line_damping.flip(0, 1)
    damping[:, -grid.PML_size:] += line_damping
    return damping


def plotProperties(propertiesPred, GTProperties, plotLoss, grid, prob, physicalModel):
    addTitle = False
    if plotLoss or grid.plotDas:
        fig, axes = plt.subplots(2, 5)
    else:
        fig, axes = plt.subplots(2, 4)
    plt.ion()
    if addTitle: fig.suptitle('Backpropogating to the velocities')
    im = []
    if addTitle: axes[0, 0].set_title('GT Velocities')
    img = axes[0, 0].imshow(GTProperties.velocity.detach().cpu(), interpolation='nearest', vmin=physicalModel.vmin_velocity, vmax=physicalModel.vmax_velocity, animated=True, cmap=plt.cm.RdBu)
    if not plotLoss:
        cbar = fig.colorbar(img, ax=axes[0, 0])
        if addTitle: cbar.ax.set_ylabel('[m / s]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[1, 0].set_title('Estimated Velocities')
    im.append(axes[1, 0].imshow(propertiesPred.velocity.detach().cpu().numpy(), interpolation='nearest', vmin=physicalModel.vmin_velocity, vmax=physicalModel.vmax_velocity, animated=True, cmap=plt.cm.RdBu))
    if not plotLoss:
        cbar = fig.colorbar(im[-1], ax=axes[1, 0])
        if addTitle: cbar.ax.set_ylabel('[m / s]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[0, 1].set_title('GT Densities')
    img = axes[0, 1].imshow(GTProperties.density.detach().cpu(), interpolation='nearest', vmin=physicalModel.vmin_density, vmax=physicalModel.vmax_density, animated=True, cmap=plt.cm.RdBu)
    if not plotLoss:
        cbar = fig.colorbar(img, ax=axes[0, 1])
        if addTitle: cbar.ax.set_ylabel('[1000 * kg / m^3]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[1, 1].set_title('Estimated Densities')
    im.append(axes[1, 1].imshow(propertiesPred.density.detach().cpu().numpy(), interpolation='nearest', vmin=physicalModel.vmin_density, vmax=physicalModel.vmax_density, animated=True, cmap=plt.cm.RdBu))
    if not plotLoss:
        cbar = fig.colorbar(im[-1], ax=axes[1, 1])
        if addTitle: cbar.ax.set_ylabel('[1000 * kg / m^3]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[0, 2].set_title('GT Damping')
    img = axes[0, 2].imshow(GTProperties.damping.detach().cpu(), interpolation='nearest', vmin=physicalModel.vmin_damping, vmax=physicalModel.vmax_damping, animated=True, cmap=plt.cm.RdBu)
    if not plotLoss:
        cbar = fig.colorbar(img, ax=axes[0, 2])
        if addTitle: cbar.ax.set_ylabel('[1 / s]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[1, 2].set_title('Estimated Damping')
    im.append(axes[1, 2].imshow(propertiesPred.damping.detach().cpu().numpy(), interpolation='nearest', vmin=physicalModel.vmin_damping, vmax=physicalModel.vmax_damping, animated=True, cmap=plt.cm.RdBu))
    if not plotLoss:
        cbar = fig.colorbar(im[-1], ax=axes[1, 2])
        if addTitle: cbar.ax.set_ylabel('[1 / s]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[0, 3].set_title('GT nonlinearity')
    img = axes[0, 3].imshow(GTProperties.beta.detach().cpu(), interpolation='nearest', vmin=physicalModel.vmin_beta, vmax=physicalModel.vmax_beta, animated=True, cmap=plt.cm.RdBu)
    if not plotLoss:
        cbar = fig.colorbar(img, ax=axes[0, 3])
        if addTitle: cbar.ax.set_ylabel('', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[1, 3].set_title('Estimated nonlinearity')
    im.append(axes[1, 3].imshow(propertiesPred.beta.detach().cpu().numpy(), interpolation='nearest', vmin=physicalModel.vmin_beta, vmax=physicalModel.vmax_beta, animated=True, cmap=plt.cm.RdBu))
    if not plotLoss:
        cbar = fig.colorbar(im[-1], ax=axes[1, 3])
        if addTitle: cbar.ax.set_ylabel('', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if grid.plotDas:
        axes[1, 4].set_title('B-mode image')
        im.append(axes[1, 4].imshow(grid.das.detach().cpu().numpy(), interpolation='nearest', animated=True, cmap=plt.cm.binary))
        vmax_pressure = max([np.abs(prob.pulses.min().cpu()), np.abs(prob.pulses.max().cpu())])
        axes[0, 4].set_title('Signal on transducer')
        im.append(axes[0, 4].imshow(grid.channelData.detach().cpu().numpy(), interpolation='nearest', animated=True, vmin=-vmax_pressure, vmax=vmax_pressure, cmap=plt.cm.binary))
        cbar = fig.colorbar(im[-1], ax=axes[0, 4])
        cbar.ax.set_ylabel('Pressure [Pa]', rotation=270)

    for i in range(2):
        for j in range(4):
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)

    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, axes, im


def findCloseElement(x, z, locations):
    optLoc = locations[0]
    for loc in locations:
        if ((loc[0] - x)**2 + (loc[1] - z)**2) < ((optLoc[0] - x)**2 + (optLoc[1] - z)**2):
            optLoc = loc
    return optLoc[0], optLoc[1]


def getBasePulse(f0, offset, grid):
    """
    Define the pulse of the probe in focused beams.
    :param f0: Central frequency.
    :param offset: The time offset (used for focusing).
    :return: The base pulse.
    """
    ist = 30  # shifting of source time function
    pulseLength = 50
    dt = grid.dt.cpu().detach().numpy()
    src = np.empty(grid.nt + 1)
    t = np.arange(grid.nt) * dt
    for it in range(grid.nt):
        src[it] = np.exp(-1.0 * f0 ** 2 * ((it - ist) * dt + offset) ** 2)
    pulses_tmp = np.diff(src.copy()) / dt * ((t / dt) < (ist + pulseLength))
    pulse = torch.from_numpy(pulses_tmp).to(grid.dtype).to(grid.device)
    return 3e6 * pulse


def getPulses(numChannels, f0, grid):
    """
    Define the pulse of the probe if the beam is not focused.
    :param numChannels: Number of channel used in each pulse.
    :param f0: Central frequency.
    :return: The base pulse.
    """
    ist = 30  # shifting of source time function
    pulseLength = 50
    dt = grid.dt.cpu().detach().numpy()
    pulses_tmp = np.zeros([numChannels, grid.nt])
    src = np.empty(grid.nt + 1)
    t = np.arange(grid.nt) * dt
    for it in range(grid.nt):
        src[it] = np.exp(-1.0 * f0 ** 2 * ((it - ist) * dt) ** 2)
    for s in range(numChannels):
        # Take the first derivative
        pulses_tmp[s] = np.diff(src.copy()) / dt * ((t / dt) < (ist + pulseLength))

    pulses = torch.from_numpy(pulses_tmp).to(grid.dtype).to(grid.device)
    return 5e4 * pulses


def getElementsLocations(numChannels, grid, stepSize):
    """
    Define the location of the elements in space.
    :param numChannels: Number of channels in the transducer.
    :param stepSize: Number of pixels between each element.
    :return: The location of the elements in the transducer.
    """
    irx = np.array([i for i in
                    range(int(grid.nx / 2 - stepSize * numChannels / 2), int(grid.nx / 2 + stepSize * numChannels / 2),
                          stepSize)])
    irz = 25 * np.ones(numChannels)
    locations = np.array([irz, irx], dtype=int).T
    return locations


def initializeSimulator():
    # Initializing the geometry
    L = 50e-3
    grid = Grid(Lx=L, Lz=L, nx=100, nz=100, nt=240, T=0.00001)
    c0 = 1480.0  # velocity (can be an array) - 580
    grid.T = grid.Lz / c0
    grid.dt = torch.tensor(float(grid.T) / float(grid.nt), dtype=grid.dtype, device=grid.device)

    # Initializing the prob and the physical model
    numChannels = 80
    f0 = 3e6  # MHz pulse
    prob = Prob(numChannels=numChannels, f0=f0, grid=grid)
    physicalModel = PhysicalModel(attenuationFactor=1.0, amplificationFactor=1.0, c0=c0)
    return grid, prob, physicalModel


def saveProperties(name, properties, dir='Results'):
    np.save(dir + '/velocityPred' + name, properties.velocity.detach().cpu().numpy())
    np.save(dir + '/densityPred' + name, properties.density.detach().cpu().numpy())
    np.save(dir + '/dampingPred' + name, properties.damping.detach().cpu().numpy())
    np.save(dir + '/betaPred' + name, properties.beta.detach().cpu().numpy())


def nlp2beta(tissue):
    """
    Nonlinear parameter to beta.
    :param tissue: The name of the tissue.
    :return: The nonlinearity, beta, of the tissue.
    """
    NonLinearParam = {'water': 5.2, 'fat': 10, 'liver': 6.8}
    return 1 + NonLinearParam[tissue] / 2


def tissueDamping(tissue, grid, prob):
    """
    Attenuation of the tissue.
    :param tissue: The name of the tissue.
    :return: The attenuation of the tissue.
    """
    a = {'water': 0.002, 'fat': 0.6, 'liver': 0.9}
    b = {'water': 2.0  , 'fat': 1.0, 'liver': 1.1}
    return a[tissue] * (prob.f0 / 1e6) ** b[tissue] * grid.dt


def getNoise(grid, prob):
    """
    Define the noise in the channel data.
    :return: Return the i.i.d. normal noise.
    """
    mean = torch.zeros([grid.nt, prob.numChannels], dtype=grid.dtype, device=grid.device)
    std = grid.stdNoise * torch.ones([grid.nt, prob.numChannels], dtype=grid.dtype, device=grid.device)
    noise = []
    [noise.append(torch.normal(mean=mean, std=std).to(grid.device).to(grid.dtype).detach()) for _ in range(prob.numLaterals)]
    return noise


def RMSE(x, y):
    """
    Compute the normalized RMSE score.
    :param x: First tensor
    :param y: Second tensor
    :return: NRMSE
    """
    mseLoss = nn.MSELoss()
    RMSE = mseLoss(x, y).sqrt()
    NRMSE = RMSE / (x.max() - x.min())
    return NRMSE.item()


def CNR(x, maskObject, maskBackground):
    """
    Compute the CNR according to two regions: object and background.
    :param x: The signal.
    :param maskObject: Region of the object.
    :param maskBackground: Region of the background.
    :return: The CNR.
    """
    muObject = x[maskObject].mean()
    varObject = x[maskObject].std() ** 2
    muBg = x[maskBackground].mean()
    varBg = x[maskBackground].std() ** 2
    CNR = 2 * (muObject - muBg) ** 2 / (varObject + varBg)
    return CNR


def SNR(signal, std):
    """
    SNR.
    :param signal: The signal.
    :param std: The std of the normal noise.
    :return: The SNR.
    """
    ES2 = (signal - signal.mean()).pow(2).mean()
    return 10 * torch.log10(ES2 / (std**2)).item()


def getMasks():
    SigmaLiver = [[-4.0, 8.0],
                  [6.0,  -4.0]]
    SigmaFat   = [[7.0, 0.0],
                  [0.0,  7.0]]
    liverLocation = [18, -16]
    fatLocation = [-15, -16]
    return SigmaLiver, SigmaFat, liverLocation, fatLocation


def findMask(x, grid, thresh=0.4):
    x = x - x.mean()
    x = (x / x.abs().max()).abs()
    x = torch.from_numpy(scipy.ndimage.gaussian_filter(x.to(dtype=torch.float32).detach().cpu().numpy(), sigma=0.6)).to(grid.dtype).to(grid.device)
    mask = x > thresh
    return mask


def dict2prop(dict):
    properties = Properties()
    properties.velocity = dict['velocity']
    properties.density = dict['density']
    properties.damping = dict['damping']
    properties.beta = dict['beta']
    return properties


def loadProperties(name, dir='Results'):
    properties = Properties()
    properties.velocity = torch.from_numpy(np.load(dir + '/velocityPred' + name + '.npy', allow_pickle=True))
    properties.density = torch.from_numpy(np.load(dir + '/densityPred' + name + '.npy', allow_pickle=True))
    properties.damping = torch.from_numpy(np.load(dir + '/dampingPred' + name + '.npy', allow_pickle=True))
    properties.beta = torch.from_numpy(np.load(dir + '/betaPred' + name + '.npy', allow_pickle=True))
    return properties

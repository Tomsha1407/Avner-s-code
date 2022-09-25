import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gc
import torch.multiprocessing as mp

import utils


class SoundWave2DModel(torch.nn.Module):
    def __init__(self, properties, grid, prob, physicalModel):
        super(SoundWave2DModel, self).__init__()
        self.grid = grid
        self.prob = prob
        self.physicalModel = physicalModel

        self.velocity = torch.nn.Parameter(properties.velocity.detach().clone().to(grid.dtype).to(grid.device))
        self.density = torch.nn.Parameter(properties.density.detach().clone().to(grid.dtype).to(grid.device))
        self.damping = torch.nn.Parameter(properties.damping.detach().clone().to(grid.dtype).to(grid.device))
        self.beta = torch.nn.Parameter(properties.beta.detach().clone().to(grid.dtype).to(grid.device))

        if not grid.reconstructVelocity: self.velocity.requires_grad = False
        if not grid.reconstructDensity: self.density.requires_grad = False
        if not grid.reconstructDamping: self.damping.requires_grad = False
        if not grid.reconstructBeta: self.beta.requires_grad = False

        # Each iteration is obtained by a convolution with the Laplacian filter
        laplacian_kernel = torch.tensor([[0, 0, -1. / 12, 0, 0],
                                         [0, 0, 4. / 3, 0, 0],
                                         [-1. / 12, 4. / 3, -2 * 5. / 2, 4. / 3, -1. / 12],
                                         [0, 0, 4. / 3, 0, 0],
                                         [0, 0, -1. / 12, 0, 0]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_laplacian = nn.Conv2d(1, 1, kernel_size=(grid.nop, grid.nop),
                                             padding=(int((grid.nop - 1) / 2), int((grid.nop - 1) / 2)))
        self.cnn_layer_laplacian.weight = torch.nn.Parameter(
            laplacian_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_laplacian.weight.requires_grad = False
        self.cnn_layer_laplacian.bias = None
        self.laplacian = lambda x: 1 / self.grid.dz ** 2 * self.cnn_layer_laplacian(x)

        grad_z_kernel = 0.5 * torch.tensor([[0, -1, 0],
                                      [0, 0, 0],
                                      [0, 1, 0]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_grad_z = nn.Conv2d(1, 1, kernel_size=(grid.nop, grid.nop), padding=(1, 1))
        self.cnn_layer_grad_z.weight = torch.nn.Parameter(
            grad_z_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_grad_z.weight.requires_grad = False
        self.cnn_layer_grad_z.bias = None
        self.grad_z = lambda x: 1 / self.grid.dz * self.cnn_layer_grad_z(x)

        grad_x_kernel = 0.5 * torch.tensor([[0, 0, 0],
                                      [-1, 0, 1],
                                      [0, 0, 0]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_grad_x = nn.Conv2d(1, 1, kernel_size=(grid.nop, grid.nop), padding=(1, 1))
        self.cnn_layer_grad_x.weight = torch.nn.Parameter(
            grad_x_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_grad_x.weight.requires_grad = False
        self.cnn_layer_grad_x.bias = None
        self.grad_x = lambda x: 1 / self.grid.dz * self.cnn_layer_grad_x(x)

        diag1_kernel = 0.5 * torch.tensor([[1, 0, 0],
                                           [0, 0, 0],
                                           [0,  0, -1]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_diag1 = nn.Conv2d(1, 1, kernel_size=(grid.nop, grid.nop), padding=(1, 1))
        self.cnn_layer_diag1.weight = torch.nn.Parameter(diag1_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_diag1.weight.requires_grad = False
        self.cnn_layer_diag1.bias = None
        self.diag1 = lambda x: 1 / self.grid.dz * self.cnn_layer_diag1(x)

        diag2_kernel = 0.5 * torch.tensor([[0, 0, 1],
                                           [0, 0, 0],
                                           [-1,  0, 0]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_diag2 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1))
        self.cnn_layer_diag2.weight = torch.nn.Parameter(diag2_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_diag2.weight.requires_grad = False
        self.cnn_layer_diag2.bias = None
        self.diag2 = lambda x: 1 / self.grid.dz * self.cnn_layer_diag2(x)

        sobelx_kernel = 0.5 * torch.tensor([[1, 0, -1],
                                           [2, 0, -2],
                                           [1,  0, -1]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_sobelx = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1))
        self.cnn_layer_sobelx.weight = torch.nn.Parameter(sobelx_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_sobelx.weight.requires_grad = False
        self.cnn_layer_sobelx.bias = None
        self.sobelx = lambda x: 1 / self.grid.dz * self.cnn_layer_sobelx(x)

        sobelz_kernel = 0.5 * torch.tensor([[1, 2, 1],
                                           [0, 0, 0],
                                           [-1,  -2, -1]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_sobelz = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1))
        self.cnn_layer_sobelz.weight = torch.nn.Parameter(sobelz_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_sobelz.weight.requires_grad = False
        self.cnn_layer_sobelz.bias = None
        self.sobelz = lambda x: 1 / self.grid.dz * self.cnn_layer_sobelz(x)

        self.grad_dot_grad = lambda x, y: self.grad_x(1 / (x[None, None, :, :])) * self.grad_x(y) + self.grad_z(
            1/(x[None, None, :, :])) * self.grad_z(y)

        self.sobel  = lambda x: (self.sobelz(x[None, None, :, :]).abs()) #        self.sobel  = lambda x: (self.sobelz(x[None, None, :, :]).abs()) #
        self.robert = lambda x: (self.diag1(x[None, None, :, :]).abs() + self.diag2(x[None, None, :, :]).abs())

    def forward(self, NLAmodel, plot=False):
        p = torch.zeros([1, 1, self.grid.nz, self.grid.nx], requires_grad=True, dtype=self.grid.dtype, device=self.grid.device)
        pm1 = torch.zeros([1, 1, self.grid.nz, self.grid.nx], requires_grad=True, dtype=self.grid.dtype, device=self.grid.device)
        pm2 = torch.zeros([1, 1, self.grid.nz, self.grid.nx], requires_grad=True, dtype=self.grid.dtype, device=self.grid.device)
        trans = []
        ir = np.arange(self.prob.numChannels)

        K = lambda c, rho: c.pow(2) * rho
        NLA =  lambda beta, c, rho, pm1: (1 + 2 * beta / K(c, rho) * pm1)
        NLA2 = lambda beta, c, rho: (2 * beta / K(c, rho))

        for t in range(self.grid.nt):
            if NLAmodel:
                dt = self.grid.dt
                a = NLA(self.beta, self.velocity, self.density, pm1) + 2 * self.damping
                b = (2 * NLA(self.beta, self.velocity, self.density, pm1) - self.damping.pow(2))
                c = (2 * self.damping - NLA(self.beta, self.velocity, self.density, pm1))
                d = NLA2(self.beta, self.velocity, self.density)

                p = (b * pm1 + c * pm2 + d * (2 * pm1 * pm2 - pm1.pow(2) - pm2.pow(2)) +
                     dt.pow(2) * K(self.velocity, self.density) * (self.density * self.laplacian(p) + self.grad_dot_grad(self.density, p))
                     + dt.pow(2) * self.prob.pulse[t]) / a
            else: # Linear model
                dt = self.grid.dt
                a = 1 + 2 * self.damping
                b = 2 - self.damping.pow(2)
                c = 2 * self.damping - 1

                p = (b * pm1 + c * pm2 +
                     dt.pow(2) * K(self.velocity, self.density) * (self.density * self.laplacian(p) + self.grad_dot_grad(self.density, p))
                     + dt.pow(2) * self.prob.pulse[t]) / a

            pm2, pm1 = pm1, p

            # Save signal on transducers
            trans.append(p[0][0][self.prob.loc[ir, 0], self.prob.loc[ir, 1]])

            # if plot: yield p
        del p, pm1, pm2
        return torch.stack(trans)


def SoundWave2D(properties, grid, prob, physicalModel, plot=False):
    ir = np.arange(prob.numChannels)
    trans = torch.zeros([prob.numChannels, grid.nt], dtype=grid.dtype, device=grid.device)  # The signal on the transducers

    prob.pulse = getPulses(grid, prob, physicalModel)

    # Initialize animated plot
    if plot:
        fig, axes, im = prepareFigWave2D(properties, grid, prob, physicalModel)

    model = SoundWave2DModel(properties, grid, prob, physicalModel)
    if grid.device is torch.device("cuda"):
        model = model.cuda()

    if plot:
        ps = model.forward(NLAmodel=True, plot=True)
        for it, p in enumerate(ps):
            trans[ir, it] = p[0][0][prob.loc[ir, 0], prob.loc[ir, 1]]
            plotWave2D(p, properties, trans, it, plot, ir, fig, axes, im, grid, prob, physicalModel)
    return model.forward(NLAmodel=True)


def prepareFigWave2D(properties, grid, prob, physicalModel):
    fig = plt.figure(figsize=(10, 24))
    axes = []
    nrow = 1
    ncol = 5
    axes.append(plt.subplot2grid((nrow, ncol), (0, 0)))
    axes.append(plt.subplot2grid((nrow, ncol), (0, 1)))
    axes.append(plt.subplot2grid((nrow, ncol), (0, 2)))
    axes.append(plt.subplot2grid((nrow, ncol), (0, 3)))
    axes.append(plt.subplot2grid((nrow, ncol), (0, 4)))
    toPlot = torch.zeros([grid.nz, grid.nx])

    vmax_pressure = 2 * max([np.abs(prob.pulses.min().cpu()), np.abs(prob.pulses.max().cpu())])
    im = []
    im.append(axes[0].imshow(toPlot, interpolation='nearest', animated=True, #vmin=-vmax_pressure, vmax=vmax_pressure,
                             cmap=plt.cm.RdBu))
    cbar = fig.colorbar(im[0], ax=axes[0])
    cbar.ax.set_ylabel('Pressure [Pa]', rotation=270)
    im.append(axes[1].imshow(toPlot, interpolation='nearest', animated=True, vmin=physicalModel.vmin_velocity,
                             vmax=physicalModel.vmax_velocity, cmap=plt.cm.RdBu))
    cbar = fig.colorbar(im[1], ax=axes[1])
    cbar.ax.set_ylabel('[m / s]', rotation=270)
    im.append(axes[2].imshow(toPlot, interpolation='nearest', animated=True, vmin=physicalModel.vmin_density,
                             vmax=physicalModel.vmax_density, cmap=plt.cm.RdBu))
    cbar = fig.colorbar(im[2], ax=axes[2])
    cbar.ax.set_ylabel('[1000 * kg / m^3]', rotation=270)
    im.append(axes[3].imshow(toPlot, interpolation='nearest', animated=True, vmin=physicalModel.vmin_damping,
                             vmax=physicalModel.vmax_damping, cmap=plt.cm.RdBu))
    cbar = fig.colorbar(im[3], ax=axes[3])
    cbar.ax.set_ylabel('[1 / s]', rotation=270)
    im.append(axes[4].imshow(toPlot, interpolation='nearest', animated=True, vmin=physicalModel.vmin_beta,
                             vmax=physicalModel.vmax_beta, cmap=plt.cm.RdBu))
    cbar = fig.colorbar(im[4], ax=axes[4])
    cbar.ax.set_ylabel('', rotation=270)

    toPlot = properties.velocity.detach().cpu()
    im[1].set_data(toPlot)
    toPlot = properties.density.detach().cpu()
    im[2].set_data(toPlot)
    toPlot = properties.damping.detach().cpu()
    im[3].set_data(toPlot)
    toPlot = properties.beta.detach().cpu()
    im[4].set_data(toPlot)

    for s in range(prob.numChannels):
        axes[0].text(prob.loc[s, 1], prob.loc[s, 0], 'o')

    axes[0].set_title('Acoustic Wave')
    axes[1].set_title('Velocity')
    axes[2].set_title('Density')
    axes[3].set_title('Damping')
    axes[4].set_title('Beta')

    return fig, axes, im


def plotWave2D(p, properties, trans, it, plot, ir, fig, axes, im, grid, prob, physicalModel):
    isnap = 10  # snapshot frequency

    if it % isnap == 0 and plot:

        toPlot = p[-1].detach().cpu()
        toPlot = toPlot[0, :, :]
        im[0].set_data(toPlot)

        plt.gcf().canvas.draw()
        plt.ion()
        plt.show()
        plt.gcf().canvas.flush_events()


def InverseUSSolver(initialProperties, trans, GTvelocities, plotProcess, grid, prob, physicalModel):
    # torch.autograd.set_detect_anomaly(True)

    prob.pulse = getPulses(grid, prob, physicalModel)
    model, optimizer, scheduler = getModel(initialProperties, grid, prob, physicalModel, 1e0)

    # Defining the losses
    mseLoss = torch.nn.MSELoss()
    l1Loss = torch.nn.L1Loss()
    regularization = lambda x: (5 * model.sobel(x) + 5 * model.robert(x)).mean() # / 2

    if plotProcess:
        fig, axes, im = preparePlotOptimization(GTvelocities, grid, prob, physicalModel)

    losses = []
    epsVelocity = 0.06
    epsDensity = 0.1
    epsDamping = 0.01
    epsBeta = 0.01
    numItrations = 100
    block = 0
    zAx_n, x = np.ogrid[0: grid.nz, 0: grid.nx]
    zAx = torch.from_numpy(zAx_n).to(grid.dtype).to(grid.device)
    NLA = True
    zerosTensor = torch.zeros(model.velocity.shape).to(grid.dtype).to(grid.device)
    for t in range(numItrations):
        # Forward pass: Compute predicted signal on the transducers
        transPred = model(NLAmodel=NLA)

        loss = 1e5 * mseLoss(transPred, trans) \
                + 0.05 * 1e-2 * regularization(model.density) \
                + 0.1 * 1e-6 * regularization(model.velocity) \
                + 0.08 * l1Loss(model.velocity - physicalModel.c0, zerosTensor).pow(2)

        losses.append(loss.item())

        if plotProcess and (t > 0 and (t == 9 or t % 10 == 9)):
            plotOptimizationProcess(model, loss, losses, t + 1, fig, axes, im, grid, prob, physicalModel)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        border = 25
        limit = 50
        if grid.reconstructVelocity:
            epsG = epsVelocity * model.velocity.grad.abs().max()
            model.velocity.grad = model.velocity.grad * (model.velocity.grad.abs() > epsG) * (zAx >= (border + limit * block)) * (zAx <= (border + limit * (block + 1))) * grid.mask

        if grid.reconstructDensity:
            epsG = epsDensity * model.density.grad.abs().max()
            model.density.grad = model.density.grad * (model.density.grad.abs() > epsG) * (zAx >= (border + limit * block)) * (zAx <= (border + limit * (block + 1)))

        if grid.reconstructDamping:
            epsG = epsDamping * model.damping.grad.abs().max()
            model.damping.grad = model.damping.grad * (model.damping.grad < -epsG) * (zAx >= (border + limit * block)) * (zAx <= (border + limit * (block + 1))) * grid.mask

        if grid.reconstructBeta:
            epsG = epsBeta * model.beta.grad.abs().max()
            model.beta.grad = model.beta.grad * (model.beta.grad < -epsG) * (zAx >= (border + 120 * block)) * (zAx <= (border + 120 * (block + 1))) * grid.mask

        # Updating velocities
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    properties = {'velocity': model.velocity.detach().to('cuda:0'), 'density': model.density.detach().to('cuda:0'),
                  'damping': model.damping.detach().to('cuda:0'), 'beta': model.beta.detach().cpu().to('cuda:0')}
    del model

    return properties


def getPulses(grid, prob, physicalModel):
    probPulses = torch.zeros([grid.nt, grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    for tt in range(grid.nt):
        for s in range(prob.numChannels):
            probPulses[tt, prob.loc[s, 0], prob.loc[s, 1]] += prob.pulses[s, tt]
    return probPulses


def preparePlotOptimization(GTProperties, grid, prob, physicalModel):
    fig, axes, im = utils.plotProperties(GTProperties, GTProperties, True, grid, prob, physicalModel)
    axes[1, 4].set_title('Loss [dB]')
    return fig, axes, im


def plotOptimizationProcess(model, loss, losses, t, fig, axes, im,  grid, prob, physicalModel):
    velocityFiltered = model.velocity.detach().cpu().numpy()
    densityFiltered = model.density.detach().cpu().numpy()
    dampingFiltered = model.damping.detach().cpu().numpy()
    betaFiltered = model.beta.detach().cpu().numpy()

    fig.suptitle('Backpropogating to the velocities, iter = ' + str(t) + ', loss = ' + str(loss.item()))
    im[0].set_data(velocityFiltered)
    im[1].set_data(densityFiltered)
    im[2].set_data(dampingFiltered)
    im[3].set_data(betaFiltered)

    axes[1, 4].plot(np.log(losses), color='darkslategrey')
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()


def multipleTransmissions(initialProperties, GTProperties, computeDAS, grid, prob, physicalModel):
    gc.collect()
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_num_threads(1)
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    propertiesPred_array = []
    parallelLaterals = 8
    batchLaterals = int(prob.numLaterals / parallelLaterals) + 1
    for batch in range(batchLaterals):
        laterals = []
        for parLateral in range(parallelLaterals):
            lateral = batch * parallelLaterals + parLateral
            if lateral < prob.numLaterals:
                laterals.append(lateral)

        with mp.Pool(processes=parallelLaterals) as pool:
            asyncResult = pool.map_async(singleTransmission, zip(laterals, [initialProperties] * len(laterals),
                                                                 [grid] * len(laterals), [prob] * len(laterals),
                                                                 [physicalModel] * len(laterals)))
            asyncResult.get()
        propertiesPred_array.extend([utils.loadProperties('tmp' + str(lateral), dir='tmp') for lateral in laterals])

    velocitiesPred_array = [properties.velocity.to('cuda:0') for properties in propertiesPred_array]
    velocitiesPred = sum([velocity - physicalModel.c0 for velocity in velocitiesPred_array]) / (prob.numLaterals - 0) + physicalModel.c0
    densitiesPred_array = [properties.density.to('cuda:0') for properties in propertiesPred_array]
    densitiesPred = sum([density - 1 for density in densitiesPred_array]) / (prob.numLaterals - 0) + 1
    dampingPred_array = [properties.damping.to('cuda:0') for properties in propertiesPred_array]
    dampingPred = sum(dampingPred_array) / len(dampingPred_array)
    betaPred_array = [properties.beta.to('cuda:0') for properties in propertiesPred_array]
    betaPred = sum(betaPred_array) / len(betaPred_array)

    for c, r, D, b in zip(velocitiesPred_array, densitiesPred_array, dampingPred_array, betaPred_array):
        del c, r, D, b
    del propertiesPred_array

    propertiesPred = {'velocity': velocitiesPred, 'density': densitiesPred, 'damping': dampingPred, 'beta': betaPred}
    return propertiesPred


def singleTransmission(arguments):
    lateral, initialProperties, grid, prob, physicalModel = arguments
    numGPUs = 2
    print(str(lateral + 1) + '/' + str(prob.numLaterals), end=', ')
    cudaDevice = torch.device("cuda:" + str(int(lateral % numGPUs)))
    grid.device = cudaDevice
    grid.dt = grid.dt.to(device=cudaDevice)
    grid.mask = grid.mask.to(device=cudaDevice)
    grid.noise = [n.to(device=cudaDevice) for n in grid.noise]
    GTProperties = utils.getGTProperties(grid, prob, physicalModel)

    focusedBeam = True
    P = 5e-3
    prob.pulses = torch.zeros([prob.numChannels, grid.nt], dtype=grid.dtype, device=grid.device)
    for i in range(lateral * prob.lateralStride, lateral * prob.lateralStride + prob.sizeLateral):
        if focusedBeam:
            offsetChannel = int(i - lateral * prob.lateralStride - prob.sizeLateral / 2)
            offsetDist = offsetChannel * prob.stepSize * grid.dx
            delay = (np.sqrt(P ** 2 + offsetDist ** 2) - P) / physicalModel.c0
            prob.pulses[i] = utils.getBasePulse(prob.f0, delay, grid).clone().to(grid.dtype).to(grid.device)
        else:
            prob.pulses[i] = prob.base_pulse.clone().to(grid.dtype).to(grid.device)
    channelData = SoundWave2D(GTProperties, grid, prob, physicalModel, plot=False) + grid.noise[lateral]
    propertiesRef = utils.getRefProperties(grid, prob, physicalModel)
    channelDataRef = SoundWave2D(propertiesRef, grid, prob, physicalModel)
    if lateral == 0: print('SNR =', utils.SNR(channelData - channelDataRef, grid.stdNoise), '[dB]', end=' ')
    plotProcess = lateral == 4 or lateral == 12
    propertiesPred = InverseUSSolver(initialProperties, channelData, GTProperties, plotProcess, grid, prob, physicalModel)
    properties = utils.Properties()
    properties.velocity = propertiesPred['velocity']
    properties.density = propertiesPred['density']
    properties.damping = propertiesPred['damping']
    properties.beta = propertiesPred['beta']
    utils.saveProperties('tmp' + str(lateral), properties, dir='tmp')


def getModel(properties, grid, prob, physicalModel, lr):
    newProperties = utils.Properties()
    newProperties.velocity = properties.velocity
    newProperties.density = properties.density
    newProperties.damping = properties.damping
    model = SoundWave2DModel(properties, grid, prob, physicalModel)
    cudaDevice = grid.device
    grid.device = cudaDevice

    model = model.cuda(device=grid.device)
    optimizer = torch.optim.Adam([
        {'params': model.velocity, 'lr': grid.factor * 4e0},
        {'params': model.density, 'lr': grid.factor * 5e-3},
        {'params': model.damping, 'lr': grid.factor * 3e-9},
        {'params': model.beta, 'lr': grid.factor * 3e-2}], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.75)
    return model, optimizer, scheduler
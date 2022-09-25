import numpy as np
import torch
import timeit

import SoundWave2D
import utils


def main():
    # Initializing the simulator
    grid, prob, physicalModel = utils.initializeSimulator()
    grid.noise = utils.getNoise(grid, prob)

    # Defining the GT properties and initial properties
    GTProperties = utils.getGTProperties(grid, prob, physicalModel)
    initialProperties = utils.initializeProperties(0, 0, GTProperties, grid, prob, physicalModel, DASInitialization=False)

    # To improve the reconstruction, after estimating the density, we use it to define a mask.
    # The algorithm will update the gradients only in the regin of the mask.
    updateMask = False
    if updateMask:
        num = '1'
        predForMaks = torch.from_numpy(np.load('Results/densityPred' + num + '.npy', allow_pickle=True))
        grid.mask = utils.findMask(predForMaks, grid)

    # At each iteration, the algorithm will optimize over n_l lateral scan.
    # The variable repetition, define how many times this is done.
    repetitions = 10
    propertiesPred = [initialProperties]  # A list containing the predictions
    startTime = timeit.default_timer()
    for ind_r in range(repetitions):
        print('\nRepetition:', ind_r + 1, '/', repetitions, end=': ')

        propertiesPred.append(utils.dict2prop(SoundWave2D.multipleTransmissions(propertiesPred[-1], GTProperties, False, grid, prob, physicalModel)))
        grid.factor *= 0.95
        utils.plotProperties(propertiesPred[-1], GTProperties, False, grid, prob, physicalModel)
        utils.saveProperties('1', propertiesPred[-1])

    stopTime = timeit.default_timer()
    print('Inverse US Solver Time = ', stopTime - startTime)

    print('RMSE: c0,               rho0,              D,                 beta')
    print('     ', utils.RMSE(GTProperties.velocity.cpu(), propertiesPred[-1].velocity), utils.RMSE(GTProperties.density.cpu(), propertiesPred[-1].density),
          utils.RMSE(GTProperties.damping.cpu(), propertiesPred[-1].damping) * 1e5, utils.RMSE(GTProperties.beta.cpu(), propertiesPred[-1].beta))


if __name__ == "__main__":
    main()

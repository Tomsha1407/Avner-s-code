import numpy as np
import torch

import SoundWave2D
import utils


def DAS2(channel_data, grid, prob, physicalModel):
    offset = 60
    z_offset = 128
    #  Channel data is a channel x axial tensor
    base_z = prob.loc[0, 0] * grid.dz
    rf_data_temp = np.zeros([grid.nz, grid.nx, prob.numChannels])
    for ix in range((256 - offset), (256 + offset)):
        for iz in range(z_offset, (z_offset + 2 * offset)):
            for channel in range(prob.numChannels):
                x = ix * grid.dx
                z = iz * grid.dz
                element_x = prob.loc[channel, 1] * grid.dx
                element_z = prob.loc[channel, 0] * grid.dz
                t = (z - base_z) / physicalModel.c0
                pos_diff = np.sqrt((x - element_x) ** 2 + (z - element_z) ** 2) / physicalModel.c0
                tau = t + pos_diff
                if 0 <= int(tau / grid.dt) < grid.nt:
                    rf_data_temp[iz, ix, channel] += channel_data[int(tau / grid.dt), channel]

    rf_data = torch.from_numpy(np.sum(rf_data_temp, 2)).to(grid.dtype).to(grid.device)
    return rf_data


def DAS(channel_data, grid, prob, physicalModel):
    offset = int(grid.nx / 2)
    z_offset = prob.loc[0, 0]
    #  Channel data is a channel x axial tensor
    base_z = prob.loc[0, 0] * grid.dz
    rf_data_temp = np.zeros([grid.nz, grid.nx, prob.numChannels])
    element_x = prob.loc[:, 1] * grid.dx
    element_z = prob.loc[:, 0] * grid.dz
    dt = grid.dt.detach().cpu().numpy()
    for ix in range((int(grid.nx / 2) - offset), (int(grid.nx / 2) + offset)):
        for iz in range(z_offset, grid.nz):
            x = ix * grid.dx
            z = iz * grid.dz
            t = (z - base_z) / physicalModel.c0
            pos_diff = np.sqrt((x - element_x) ** 2 + (z - element_z) ** 2) / physicalModel.c0
            tau = t + pos_diff
            valid = ((tau / dt).astype(np.int32) >= 0) * ((tau / dt).astype(np.int32) < grid.nt)
            rf_data_temp[iz, ix] += np.diag(channel_data[(tau / dt * valid).astype(np.int32)].detach().cpu().numpy()) * valid

    apodization = getApodizations(prob)
    apodization = np.tile(apodization, (grid.nz, grid.nx, 1))
    rf_data = torch.from_numpy(np.sum(rf_data_temp * apodization, 2)).to(grid.dtype).to(grid.device)
    return rf_data


def getApodizations(prob):
    elements_vec = np.arange(prob.numChannels)
    # apodization = np.exp(((elements_vec - prob.central_element)/10) ** 2)
    # apodization = apodization / np.sum(apodization)
    apodization = np.abs(elements_vec - prob.central_element) < 15
    return apodization


def multipleTransmissionDAS(properties, grid, prob, physicalModel):
    das = torch.zeros([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device) + physicalModel.c0
    for lateral in range(prob.numLaterals):
        prob.pulses = torch.zeros([prob.numChannels, grid.nt], dtype=grid.dtype, device=grid.device)
        for i in range(lateral * prob.lateralStride, lateral * prob.lateralStride + prob.sizeLateral):
            prob.pulses[i] = prob.base_pulse.clone().to(grid.dtype).to(grid.device)
        trans = SoundWave2D.SoundWave2D(properties, grid, prob, physicalModel)
        propertiesRef = utils.getRefProperties(grid, prob, physicalModel)
        transRef = SoundWave2D.SoundWave2D(propertiesRef, grid, prob, physicalModel)

        prob.central_element = int(lateral * prob.lateralStride + prob.sizeLateral / 2)
        das_l = DAS(trans - transRef, grid, prob, physicalModel)
        das += das_l

    das = das.detach().cpu().numpy()
    # das = np.exp(das / np.max(das))
    # Log compression
    gamma = 0.1
    das = np.log(1 + das / np.max(np.abs(das)) * gamma)
    # das = das / np.sum(das)
    # das = das - np.mean(das)
    # das = das / np.max(np.abs(das)) * 80 + physicalModel.c0
    das = torch.from_numpy(das).to(grid.dtype).to(grid.device)
    return das

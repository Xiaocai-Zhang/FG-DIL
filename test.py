import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
import win32com.client
import config
from vissimModel import Vissim_Server
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tensorflow.keras import Model, Input
import random
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Conv1D, Activation, BatchNormalization,
    GlobalAveragePooling1D, Reshape, Multiply, GaussianDropout
)
import tensorflow as tf





start = datetime.now()
seed = 20894
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


# ====================
# Define Blocks
# ====================
def cwab_block(x, reduction=16):
    channels = x.shape[-1]
    squeeze = GlobalAveragePooling1D()(x)
    excitation = Dense(channels // reduction, activation='relu')(squeeze)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    excitation = Reshape((1, channels))(excitation)
    return Multiply()([x, excitation])


class FGNetworks:
    def dccb_block(self, x, dilation_rate, nb_filters, kernel_size, dropout, padding):
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                       kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(1e-4))
        batch1 = BatchNormalization(axis=-1)
        ac1 = Activation('relu')
        drop1 = GaussianDropout(dropout)

        conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                       kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(1e-4))
        batch2 = BatchNormalization(axis=-1)
        ac2 = Activation('relu')
        drop2 = GaussianDropout(dropout)

        downsample = Conv1D(filters=nb_filters, kernel_size=1, padding='same', kernel_initializer=init)
        ac3 = Activation('relu')

        pre_x = x

        x = conv1(x)
        x = batch1(x)
        x = ac1(x)
        x = drop1(x)

        x = conv2(x)
        x = batch2(x)
        x = ac2(x)
        x = drop2(x)

        x = cwab_block(x)

        if pre_x.shape[-1] != x.shape[-1]:  # to match the dimensions
            pre_x = downsample(pre_x)

        out = ac3(pre_x + x)
        return out

    def Network(self, x, num_channels, kernel_size, dropout):
        for i, filters in enumerate(num_channels):
            dilation_rate = 2 ** i
            x = self.dccb_block(x, dilation_rate, filters, kernel_size, dropout, padding='causal')
        return x


# ===============
# Model Building
# ===============
input = Input(shape=(config.temporal_dimension,config.feature_dimension))
x = Conv1D(config.h_dim, kernel_size=1, activation='relu')(input)
fg_net = FGNetworks()
x = fg_net.Network(x, config.num_channels, config.kernel_tcn, config.dropout_rate)
x = Flatten()(x)
output = Dense(config.n_action)(x)

model = Model(inputs=input, outputs=output)
model.load_weights(config.modelSavePath)


# ===============
# Simulation
# ===============
Vissim = win32com.client.gencache.EnsureDispatch("Vissim.Vissim.2022")
cur_dic = os.getcwd()
cur_dic = cur_dic.replace('\\', '/')
Filename_inpx = cur_dic + "/Vissim_model/Intersection_1.inpx"
Filename_layx = cur_dic + "/Vissim_model/Intersection_1.layx"
Vissim.LoadNet(Filename_inpx, False)
Vissim.LoadLayout(Filename_layx)
config.dfLinkInfo = pd.read_csv(config.pathlinkFile)

for simRun in Vissim.Net.SimulationRuns:
    Vissim.Net.SimulationRuns.RemoveSimulationRun(simRun)

VissimCOM = Vissim_Server(Vissim)
start_e, Sim_rdsd = VissimCOM.init_parameters()

VissimCOM.init_vehicle_composition()
VissimCOM.Run_Sim_warmup(config.simulation_warm_up)
VissimCOM.Set_First_phases()

VissimCOM = Vissim_Server(Vissim)
CurPhase = 1
state,queue,veh_li,avg_speed_li = VissimCOM.scDuration_execution_ini(CurPhase)
queue_li = [queue]
while True:
    action = model.predict(state)[0][0]
    action = np.clip(action, -1, 1)
    action = (action + 1) / 2 * (config.max_gt - config.min_gt) + config.min_gt
    action = int(round(action))
    # print(action)
    stateNxt,queue,vehs,avg_speeds = VissimCOM.scDuration_execution(action,CurPhase)
    queue_li.append(queue)
    veh_li = veh_li + vehs
    avg_speed_li = avg_speed_li + avg_speeds
    NxtPhase = VissimCOM.inferNextPhase(CurPhase)
    state = stateNxt
    CurPhase = NxtPhase
    done = True if VissimCOM.Step >= config.simualtion_execution else False
    if done: break

time = VissimCOM.Step
overall_queue = sum(queue_li)
veh_num = len(set(veh_li))

# ==========
# Evaluation
# ==========
print('cumulative queue length: ',overall_queue)
print('average queue length: ',overall_queue/(time/60))
print('average waiting speed: ',overall_queue/veh_num)
print('average vehicle speed: ',np.nanmean(avg_speed_li))
Vissim.Simulation.Stop()
VissimCOM.reset_Vissim()
safety_score = VissimCOM.get_reward_safety()
print('cumulative safety score: ',safety_score)
print('average safety score: ',safety_score/(time/60))
Vissim.Exit()

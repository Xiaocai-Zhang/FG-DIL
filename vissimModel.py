import time
import config
import pandas as pd
pd.set_option('chained_assignment',None)
import SSAM_auto as SSAM
import numpy as np





class Vissim_Server:
    def __init__(self, Vissim):
        self.Vissim = Vissim
        self.Step = 0
        self.Signal_t_1 = 0
        self.Signal_t_2 = 0

    def inferNextPhase(self, curentPhases):
        nxtPhase = config.nextPhasesRule[curentPhases]
        return nxtPhase

    def init_parameters(self):
        start_e = time.time()
        ''' Simulation Parameters '''
        self.Vissim.Simulation.SetAttValue('SimPeriod', config.simulation_period)
        Sim_rdsd = 100
        self.Vissim.Simulation.SetAttValue('RandSeed', Sim_rdsd)
        self.Vissim.Simulation.SetAttValue('SimRes', config.simulation_resolution)
        if config.graphics == 0:
            self.Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)
            self.Vissim.SuspendUpdateGUI()

        for sheetname in config.demand_sheets_veh:
            df = pd.read_excel(config.pathVehDemandFile, 'd' + str(sheetname), index_col=None)
            for col in df.columns.tolist():
                for index, row in df.iterrows():
                    demand_value = row[col]
                    self.Vissim.Net.Matrices.ItemByKey(sheetname).SetValue(index + 1, col, demand_value)

        return start_e, Sim_rdsd


    def init_vehicle_composition(self):
        # car, SUV, HGV, Bus, Van, Lite Truck
        Rel_flow = [0.45,0.35,0.05,0.05,0.05,0.05]
        Speed = 60
        len_Veh_compo = len(config.VehTypelist)
        Attributes = ("VehType", "DesSpeedDistr", "RelFlow")

        values = ()
        for i in range(len_Veh_compo):
            buff = (config.VehTypelist[i], Speed, Rel_flow[i])
            values = values + (buff,)
        self.Vissim.Net.VehicleCompositions.ItemByKey(1).VehCompRelFlows.SetMultipleAttributes(Attributes, values)
        return None


    def reset_Vissim(self):
        for simRun in self.Vissim.Net.SimulationRuns:
            self.Vissim.Net.SimulationRuns.RemoveSimulationRun(simRun)
        time.sleep(3)
        return None


    def Run_Sim_warmup(self, duration):
        for a in range(duration):
            self.Vissim.Simulation.RunSingleStep()
            self.Step += 1
        return None


    def Run_Sim(self, duration):
        for a in range(int(duration)):
            self.Vissim.Simulation.RunSingleStep()
            self.Step += 1
        return None


    def Set_First_phases(self):
        # First set all the phases to RED
        for SG in [1,2,3,4]:
            self.Vissim.Net.SignalControllers.ItemByKey(1).SGs.ItemByKey(SG).SetAttValue('SigState','RED')
        return None


    def scDuration_execution(self,action,CurPhase):
        queue_li = []
        state_all = []
        vehs_li = []
        avg_speed_li = []
        PhasesCode = self.encodePhase(CurPhase)
        for i in range(action):
            self.Vissim.Net.SignalControllers.ItemByKey(1).SGs.ItemByKey(CurPhase).SetAttValue('SigState','GREEN')
            self.Run_Sim(1)
            state_traffic = self.getTrafficState()
            state = np.concatenate((state_traffic, PhasesCode))
            state_all.append(state)
            queue,vehs,speed,df = self.get_reward_efficiency()
            queue_li.append(queue)
            avg_speed_li.append(speed)
            vehs_li = vehs_li+vehs
        for i in range(3):
            self.Vissim.Net.SignalControllers.ItemByKey(1).SGs.ItemByKey(CurPhase).SetAttValue('SigState', 'AMBER')
            self.Run_Sim(1)
            queue,vehs,speed,df = self.get_reward_efficiency()
            queue_li.append(queue)
            avg_speed_li.append(speed)
            vehs_li = vehs_li+vehs
        for i in range(2):
            self.Vissim.Net.SignalControllers.ItemByKey(1).SGs.ItemByKey(CurPhase).SetAttValue('SigState', 'RED')
            self.Run_Sim(1)
            queue,vehs,speed,df = self.get_reward_efficiency()
            queue_li.append(queue)
            avg_speed_li.append(speed)
            vehs_li = vehs_li+vehs
        state_all = np.vstack(state_all)
        state_all = state_all.astype(np.float32)
        state_all = state_all[-7:]
        state_all = state_all.reshape(1, *state_all.shape)
        return state_all, sum(queue_li), vehs_li, avg_speed_li


    def scDuration_execution_ini(self,CurPhase):
        queue_li = []
        state_all = []
        vehs_li = []
        avg_speed_li = []
        PhasesCode = self.encodePhase(CurPhase)
        for i in range(7):
            self.Vissim.Net.SignalControllers.ItemByKey(1).SGs.ItemByKey(CurPhase).SetAttValue('SigState','GREEN')
            self.Run_Sim(1)
            state_traffic = self.getTrafficState()
            state = np.concatenate((state_traffic, PhasesCode))
            state_all.append(state)
            queue,vehs,speed,df = self.get_reward_efficiency()
            queue_li.append(queue)
            avg_speed_li.append(speed)
            vehs_li = vehs_li+vehs
        for i in range(3):
            self.Vissim.Net.SignalControllers.ItemByKey(1).SGs.ItemByKey(CurPhase).SetAttValue('SigState', 'AMBER')
            self.Run_Sim(1)
            queue, vehs, speed, df = self.get_reward_efficiency()
            queue_li.append(queue)
            avg_speed_li.append(speed)
            vehs_li = vehs_li+vehs
        for i in range(2):
            self.Vissim.Net.SignalControllers.ItemByKey(1).SGs.ItemByKey(CurPhase).SetAttValue('SigState', 'RED')
            self.Run_Sim(1)
            queue, vehs, speed, df = self.get_reward_efficiency()
            queue_li.append(queue)
            avg_speed_li.append(speed)
            vehs_li = vehs_li+vehs
        state_all = np.vstack(state_all)
        state_all = state_all.astype(np.float32)
        state_all = state_all[-7:]
        state_all = state_all.reshape(1, *state_all.shape)
        return state_all, sum(queue_li), vehs_li, avg_speed_li


    def encodePhase(self, phase):
        code = [1, 2, 3, 4]
        res = [0, 0, 0, 0]
        indx = code.index(phase)
        res[indx] = 1
        return np.array(res)

    def CalReward(self,ttc):
        w = self.sigmoid(ttc)
        r = w * (5-ttc)
        return r

    def sigmoid(self,x):
        k = 1
        x0 = 2.5
        return 1 / (1 + np.exp(k * (x - x0)))


    def get_reward_safety(self):
        vissimFolder = './Vissim_model/Vissim_output'
        SSAM.Start(vissimFolder)
        results = pd.read_csv(vissimFolder+'/Intersection_1_001.csv')
        if results.empty:
            return 0
        else:
            results['reward'] = results.apply(lambda row: self.CalReward(row['TTC']),axis=1)
            overall_reward = results['reward'].sum()
            return overall_reward


    def get_reward_efficiency(self):
        all_veh_attributes = self.Vissim.Net.Vehicles.GetMultipleAttributes(
            ('Lane\Link', 'No', 'Speed', 'Acceleration', 'CoordFrontX', 'CoordFrontY', 'Pos', 'VehType'))
        df_attributes = pd.DataFrame(all_veh_attributes, columns=['Lane\Link', 'No', 'Speed', 'Acceleration', 'CoordFrontX', 'CoordFrontY', 'Pos', 'VehType'])
        df_attributes = df_attributes[df_attributes['Lane\Link'].notna()]
        queue_veh = df_attributes[df_attributes['Speed'] < 5]
        reward = len(queue_veh)
        vehs = queue_veh['No'].tolist()
        return reward,vehs,df_attributes['Speed'].mean(),queue_veh


    def divide_length_into_segments(self, total_length, segment_length):
        full_segments = total_length // segment_length
        remaining_segment = total_length % segment_length
        number_of_segments = int(full_segments + (1 if remaining_segment > 0 else 0))
        rangList = []
        for i in range(number_of_segments+1):
            rang = [i * segment_length, (i + 1) * segment_length]
            rangList.append(rang)
        return rangList

    def findSegIndex(self,row,rangList):
        pos = float(row['Pos'])
        for i in range(len(rangList)):
            rang = rangList[i]
            if pos>=rang[0] and pos<rang[1]:
                return i

    def getFeatures(self, row, df_veh):
        link = row['link']
        lane = row['lane']
        length = float(row['length'])
        df_veh_lane = df_veh[(df_veh['Lane\Link'] == str(int(link))) & (df_veh['Lane\Index'] == float(lane))]
        rangList = self.divide_length_into_segments(length, config.linkSegment)
        res = np.zeros((len(rangList), 3))
        if df_veh_lane.empty == False:
            df_veh_lane['seg_index'] = df_veh_lane.apply(lambda row: self.findSegIndex(row, rangList), axis=1, result_type='expand')
            for index, row in df_veh_lane.iterrows():
                try:
                    segIndex = int(row['seg_index'])
                    res[segIndex, 0] = 1
                    res[segIndex, 1] = row['Speed']
                    res[segIndex, 2] = row['Acceleration']
                except:
                    pass
        res = self.stateNormalization(res)
        res = res.reshape(-1)
        return res

    def extractFeatures(self,df_attributes,dfLinkInfo):
        df_veh = df_attributes[df_attributes['VehType'].isin(['100', '200','300','630','640','650'])]
        featureLi = []
        for index, row in dfLinkInfo.iterrows():
            feature = self.getFeatures(row, df_veh)
            featureLi.append(feature)
        TsFeature = np.concatenate(featureLi, axis=0)
        return TsFeature

    def getTrafficState(self):
        all_veh_attributes = self.Vissim.Net.Vehicles.GetMultipleAttributes(('Lane\Link','Lane\Index','No','Speed','Acceleration ','Pos','VehType'))
        df_attributes = pd.DataFrame(all_veh_attributes,columns=['Lane\Link','Lane\Index','No','Speed','Acceleration','Pos','VehType'])
        state_traffic = self.extractFeatures(df_attributes,config.dfLinkInfo)
        return state_traffic

    def stateNormalization(self,state_traffic):
        state_traffic[:,0] = 2*state_traffic[:,0]-1
        state_traffic[:,1] = 2*(state_traffic[:,1]/config.maxSpeed)-1
        state_traffic[:,2] = 2*((state_traffic[:,2]-config.minAcceleration)/(config.maxAcceleration-config.minAcceleration))-1
        return state_traffic

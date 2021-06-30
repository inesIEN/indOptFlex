from collections import defaultdict
import pandas as pd


class PHS():
    def __init__(self, id, desc, sim_time_step,N_sim):
        self.id = id
        self.desc = desc
        self.heat_pumps = defaultdict(lambda: defaultdict)
        self.storages = defaultdict(lambda: defaultdict)
        self.sim_time_step = sim_time_step
        self.ref_load_profile_df = None
        self.heating_profile = None
        self.N_sim = N_sim
        #self.res = defaultdict(lambda: defaultdict)
        #Create the res dataframe
        self.res_df = pd.DataFrame(index=N_sim,columns=[])


    def create_heat_pump(self,id,p_kw,cop,control=None,t_on=None,t_off=None):
        self.heat_pumps[id] = (self.HEAT_PUMP(id=id,p_kw=p_kw,cop=cop,control=control,t_on=t_on,t_off=t_off)).__dict__

    def create_phs_storage_tank(self,id, volume, k_v, temp_set, temp_min, temp_max):
        self.storages[id] = (self.PHS_STORAGE(id=id, volume=volume,k_v= k_v, temp_set=temp_set, temp_min=temp_min,
                                              temp_max=temp_max)).__dict__


    class HEAT_PUMP:
        def __init__(self, id, p_kw, cop,control=None,t_on=None,t_off=None):
            self.id = id
            self.p_kw = p_kw
            self.cop = cop
            self.control = control
            self.t_on = t_on
            self.t_off = t_off


    class PHS_STORAGE:
        def __init__(self, id, volume, k_v, temp_set, temp_min, temp_max):
            self.id = id
            self.volume = volume
            self.temp_set = temp_set
            self.temp_min = temp_min
            self.temp_max = temp_max
            self.k_v = k_v
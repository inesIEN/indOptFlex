from collections import defaultdict
import  pandas as pd


from tbs.CAS.cas import CAS
from tbs.PCS.pcs import PCS
from tbs.PHS.phs import PHS
from tbs.hvac.hvac import HVAC


class TBS:
    def __init__(self, id, desc,sim_time_step,N_sim):
        self.id = id
        self.desc = desc
        self.sim_time_step = sim_time_step
        self.N_sim = N_sim
        self.ref_load_profile_df = None
        self.cas = defaultdict(lambda: defaultdict)
        self.pcs = defaultdict(lambda: defaultdict)
        self.phs = defaultdict(lambda: defaultdict)
        self.hvac = defaultdict(lambda : defaultdict)


    #Functions that uses the class of the MFS objects and save the recorded data intro "Default Dicts"

    def create_cas(self, id, desc):
        self.cas = CAS(id=id,desc=desc,sim_time_step=self.sim_time_step,N_sim=self.N_sim)

    def creat_pcs(self,id,desc):
        self.pcs = PCS(id=id,desc=desc,sim_time_step=self.sim_time_step,N_sim=self.N_sim)

    def create_phs(self,id,desc):
        self.phs = PHS(id=id,desc=desc,sim_time_step=self.sim_time_step,N_sim=self.N_sim)

    def create_hvac(self,id,desc,space_area,heat_loss_coeff,thermal_capacity,t_set_min,t_set,t_set_max):

        self.hvac = HVAC(id=id,desc=desc,sim_time_step=self.sim_time_step,space_area=space_area,
                         heat_loss_coeff=heat_loss_coeff,thermal_capacity=thermal_capacity,
                         t_set=t_set,t_set_min=t_set_min,t_set_max=t_set_max,N_sim=self.N_sim)

    def get_load_profiles_from_enit_agent(self, URL, FROM, TO, ID_List, username, password,desc):
        self.ref_load_profile_df = get_load_profiles_from_enit_agent(URL=URL, FROM=FROM, TO=TO, ID_list=ID_List,desc=desc,
                                                                     time_step=self.sim_time_step,username=username, password=password)

    def get_load_profile_from_forecast_data_pickle(self, file_path, measurement_names):
        forecast = pd.read_pickle(file_path)
        self.ref_load_profile_df = forecast[measurement_names]/1000




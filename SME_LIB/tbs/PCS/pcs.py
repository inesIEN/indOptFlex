from collections import defaultdict

import pandas as pd



class PCS():
    def __init__(self, id, desc, sim_time_step,N_sim):
        self.id = id
        self.desc = desc
        self.chillers = defaultdict(lambda: defaultdict)
        self.storages = defaultdict(lambda: defaultdict)
        self.sim_time_step = sim_time_step
        self.ref_load_profile_df = None
        self.N_sim = N_sim
        self.cooling_profile = None

        # To store the results
        self.res_df = pd.DataFrame(index=N_sim, columns=[])

    def get_load_profiles_from_enit_agent(self,URL,FROM,TO,ID_List,username,password,desc):
        self.ref_load_profile_df = get_load_profiles_from_enit_agent(URL=URL,FROM=FROM,TO=TO,ID_list=ID_List,desc=desc,
                                                                    time_step=self.sim_time_step,
                                                                   username=username,password=password)

    def get_load_profile_from_forecast_data_pickle(self, file_path, measurement_names):
        forecast = pd.read_pickle(file_path)
        self.ref_load_profile_df = forecast[measurement_names]/1000



    def get_cooling_demand_load_profiles(self,cop_chiller):
        df = self.ref_load_profile_df
        #This a normalization function
        self.cooling_profile = df * cop_chiller
        column_name = (list(self.cooling_profile.columns))[0]
        self.cooling_profile.rename(columns={column_name: 'cooling_demand'}, inplace=True)

    def create_chiller(self, id, p_kw, cooling_capacity,cop, mass_flow_rate, temp_supply,temp_return,t_on=None,t_off=None,control=None):
        self.chillers[id] = (self.Chiller(id=id,p_kw=p_kw,cooling_capacity = cooling_capacity, cop=cop,
                                          mass_flow_rate=mass_flow_rate,temp_supply=temp_supply,
                                          temp_return=temp_return,t_on=t_on,t_off=t_off,control=control)).__dict__

    def create_storage_tank(self, id, volume,temp_set, temp_max,temp_min):
        self.storages[id] = (self.pcs_storage_tank(id=id, volume=volume, temp_set = temp_set, temp_min=temp_min,
                                                   temp_max=temp_max)).__dict__


    def laod_pcs_reference_load_profiles(self,df):
        #Load the compressor machines
        chillers = list(self.chillers.keys())

        #Create the empty pandas frame with column names as compressors
        df_pcs = pd.DataFrame(columns=chillers)
        for chiller in chillers:
            df_pcs[chiller] = df[chillers]

        self.ref_load_profile_df = df_pcs
        #Hardcode:
        self.ref_load_profile_df["Total_Ref"] =  self.ref_load_profile_df.sum(axis=1)



        #Create tthe laod demand profiles based on compressor running times
        df_pcs_cooling_demand = pd.DataFrame(columns=chillers)
        for chiller in chillers:
            df_pcs_cooling_demand[chiller] = df_pcs[chiller].apply(lambda x: self.chillers["K1"]["air_prod_rate"]/2
                                            if x>1 else 0)

        self.cooling_profile = df_pcs_cooling_demand
        self.cooling_profile['Total'] = self.cooling_profile.sum(axis=1)

        #Get the sum of all air demands



    class Chiller:
        def __init__(self, id, p_kw, cooling_capacity,cop, mass_flow_rate, temp_supply,temp_return,t_on=None,t_off=None,
                     control=None):
            self.id = id
            self.p_kw = p_kw
            self.temp_supply = temp_supply
            self.temp_return = temp_return
            self.cop = cop
            self.mass_flow_rate = mass_flow_rate
            self.cooling_capacity = cooling_capacity
            self.t_on = t_on
            self.t_off = t_off
            self.control = control



    class pcs_storage_tank:
        def __init__(self, id, volume,temp_set,temp_min,temp_max):
            self.id = id
            self.volume = volume
            self.temp_set = temp_set
            self.temp_min = temp_min
            self.temp_max = temp_max

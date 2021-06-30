from collections import defaultdict

import pandas as pd

class CAS():
    def __init__(self, id, desc,sim_time_step,N_sim):
        self.id = id
        self.desc = desc
        self.sim_time_step = sim_time_step
        self.compressors = defaultdict(lambda: defaultdict)
        self.storages = defaultdict(lambda: defaultdict)
        self.ref_load_profile_df = None
        self.N_sim = N_sim
        self.air_profile = None

        #To store the results
        self.res_df = pd.DataFrame(index=N_sim,columns=[])

    def get_load_profiles_from_enit_agent(self,URL,FROM,TO,ID_List,username,password,desc):
        self.ref_load_profile_df = get_load_profiles_from_enit_agent(URL=URL,FROM=FROM,TO=TO,ID_list=ID_List,desc=desc,
                                                                   time_step=self.sim_time_step,
                                                                   username=username,password=password)


    def get_load_profile_from_forecast_data_pickle(self, file_path, measurement_names):
        forecast = pd.read_pickle(file_path)
        self.ref_load_profile_df = forecast[measurement_names]/1000


    def get_air_demand_load_profiles(self,m3_per_min_kw,time_step):
        max_air_production = m3_per_min_kw*time_step #Convert cfm to m3/m*time step
        df = self.ref_load_profile_df
        self.air_profile = df*(max_air_production)
        column_name = (list(self.air_profile.columns))[0]
        self.air_profile.rename(columns={column_name:'air_demand'},inplace=True)



    def create_compressor(self, id, p_kw, pressure_max, m3_per_min,usage_number,control=None,t_on=None,t_off=None):
        self.compressors[id] = (
            self.Compressor(id=id,p_kw=p_kw, pressure_max=pressure_max, m3_per_min=m3_per_min,
                            usage_number=usage_number,control=control,t_on=t_on,t_off=t_off)).__dict__

    def create_storage_tank(self, id, capacity, soc_initial, soc_max, soc_min,pressure_min,pressure_max, p_set):
        self.storages[id] = (
            self.Air_storage_tank(id=id, capacity=capacity, soc_initial=soc_initial, soc_max=soc_max,
                                  soc_min=soc_min,pressure_min=pressure_min, pressure_max=pressure_max,
                                  p_set = p_set)).__dict__


    def laod_cas_reference_load_profiles(self,df):
        #Load the compressor machines
        compressors = list(self.compressors.keys())

        #Create the empty pandas frame with column names as compressors
        df_cas = pd.DataFrame(columns=compressors)
        for compressor in compressors:
            df_cas[compressor] = df[compressor]

        self.ref_load_profile_df = df_cas
        #Hardcode:
        self.ref_load_profile_df["Total_Ref"] =  self.ref_load_profile_df.sum(axis=1)


        #Create the reference air demand for the compressors using the MinMax Fitting method
        df_cas_air_demand = pd.DataFrame(columns=compressors)
        for compressor in compressors:
            df_cas_air_demand[compressor] = df_cas[compressor].apply(lambda x: self.compressors["K1"]["m3_per_min"]/2
                                            if x>1 else 0)

        self.air_profile = df_cas_air_demand
        self.air_profile['Total'] = self.air_profile.sum(axis=1)

        #Get the sum of all air demands



    class Compressor:
        def __init__(self, id, p_kw, pressure_max, m3_per_min,usage_number,control=None,t_on=None,t_off=None):
            self.id = id
            self.p_kw = p_kw
            self.pressure_max = pressure_max
            self.m3_per_min = m3_per_min
            self.usage_number = usage_number
            self.control = control
            self.t_on = t_on
            self.t_off = t_off



    class Air_storage_tank:
        def __init__(self, id, capacity, soc_initial, soc_max, soc_min,pressure_min,pressure_max,p_set,**kwargs):
            self.id = id
            self.capacity = capacity
            self.soc_initial = soc_initial
            self.soc_max = soc_max
            self.soc_min = soc_min
            self.pressure_min = pressure_min
            self.pressure_max = pressure_max
            self.p_set = p_set

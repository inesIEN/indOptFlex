from collections import defaultdict

import pandas as pd
import numpy as np


class ES:
    def __init__(self, id, desc,sim_time_step,N_sim):
        self.id = id
        self.desc = desc
        self.pv_plants = defaultdict(lambda: defaultdict)
        self.ess = defaultdict(lambda: defaultdict)
        self.pv_radiation_data = None
        self.pv_production = None
        self.sim_time_step = sim_time_step
        self.chp = defaultdict(lambda: defaultdict)
        self.dg = defaultdict(lambda : defaultdict)
        self.N_sim = N_sim
        #Optimization results
        self.res = defaultdict(lambda: defaultdict)

    #%%Functions to create the Components of the Energy Systems Block
    def create_pv(self,id,p_rated):
        self.pv_plants[id] = (self.PV_GEN(id=id, p_rated=p_rated,pv_radiation_data=self.pv_radiation_data,N_sim=self.N_sim)).__dict__

    def create_ess(self,id, kwh_max, initial_soc, charging_rate_max, charge_efficiency, discharge_rate_max,
                   soc_min,soc_max,discharge_efficiency):
        self.ess[id] = (
            self.ESS(id=id, kwh_max=kwh_max, initial_soc = initial_soc, charging_rate_max=charging_rate_max,
                     charge_efficiency=charge_efficiency,
                     discharge_rate_max=discharge_rate_max,
                     soc_min = soc_min, soc_max =soc_max,
                     discharge_efficiency=discharge_efficiency,N_sim=self.N_sim)).__dict__

    def get_pv_production_data(self,URL,FROM,TO,ID_List,username,password):
        self.pv_production = get_load_profiles_from_enit_agent(URL=URL,FROM=FROM,TO=TO,ID_list=ID_List,desc="PV",
                                                               time_step=self.sim_time_step, username=username,
                                                                                            password=password)

        self.pv_production = self.pv_production * - 1.7320

    def create_chp(self,id,f_nom,p_nom,q_nom,n_heat,n_ele,fuel_cost,storage_capacity,storage_capacity_min,soc_ini,
                     storage_capacity_max,storage_loss_factor,t_on,t_off,chp_on_past,n_switches):
        self.chp[id] = (
            self.CHP(id=id,f_nom=f_nom,p_nom=p_nom,q_nom=q_nom,n_heat=n_heat,n_ele=n_ele,fuel_cost=fuel_cost,
                     storage_capacity=storage_capacity,storage_capacity_min=storage_capacity_min, soc_ini = soc_ini,
                     storage_capacity_max=storage_capacity_max,storage_loss_factor=storage_loss_factor,
                     t_on=t_on,t_off=t_off,chp_on_past=chp_on_past,n_switches=n_switches,N_sim=self.N_sim)).__dict__


    def get_pv_radiation_data(self,file_path):

        radiation = pd.read_excel(file_path)
        radiation = (radiation["radiation"]).values.tolist()

        #Repeat the values of the list for the MPC
        radiation = list(np.repeat(radiation,int(15/self.sim_time_step)))

        self.pv_radiation_data = radiation

    def get_pv_production_from_clustered_profiles(self,file_path,cluster_id):
        # get the path for the file
        pkl_file = pd.read_pickle(file_path)
        pv_profile = pkl_file["PV_Anlage_Special"][cluster_id]["Average"]
        self.pv_production = pv_profile

    def get_pv_radiation_data_from_enit_profiles(self,URL,FROM,TO,ID_List,username,password,desc):
        self.pv_production = get_load_profiles_from_enit_agent(URL=URL,FROM=FROM,TO=TO,ID_list=ID_List,desc=desc,
                                                                   time_step=self.sim_time_step,
                                                                   username=username,password=password)
        column_name = self.pv_production.columns[0]
        pv_production = self.pv_production[column_name].tolist()
        #Muliply with negative sign
        pv_production = [-x for x in pv_production  ]
        self.pv_radiation_data =  [x / max(pv_production) for x in pv_production]

    def get_pv_production_from_forecast(self,file_path,measurement_names):
        forecast = pd.read_pickle(file_path)
        self.pv_production = forecast[measurement_names] / 1000
        column_name = self.pv_production.columns[0]
        pv_production = self.pv_production[column_name].tolist()
        # Muliply with negative sign
        pv_production = [abs(x) for x in pv_production]
        self.pv_radiation_data = [x / max(pv_production) for x in pv_production]

    def create_dg(self, id, p_rated, cons_a, cons_b, price_fuel, n_min, n_max, e_coef ):
        self.dg[id]=(self.DG(id=id,p_rated=p_rated, cons_a=cons_a, cons_b=cons_b, price_fuel=price_fuel, n_min=n_min,
                             n_max=n_max, e_coef=e_coef,N_sim=self.N_sim)).__dict__


    #%%Classes of the components of PV System
    class PV_GEN:
        def __init__(self, id, p_rated,pv_radiation_data,N_sim):
            self.id = id
            self.p_rated = p_rated
            self.pv_radiation_data = pv_radiation_data
            self.N_sim = N_sim

            #Calculate the production output of PV plant
            self.p_kw = [x*self.p_rated for x in self.pv_radiation_data]

            #Results Dataframe

            self.res_df = pd.DataFrame()


    class ESS:
        def __init__(self, id, kwh_max, initial_soc, charging_rate_max, charge_efficiency, discharge_rate_max,
                     soc_min,soc_max, discharge_efficiency,N_sim):
            self.id = id
            self.kwh_max = kwh_max
            self.initial_soc = initial_soc
            self.charging_rate_max = charging_rate_max
            self.charge_efficiency = charge_efficiency
            self.discharge_rate_max = discharge_rate_max
            self.discharge_efficiency = discharge_efficiency
            self.soc_min = soc_min
            self.soc_max = soc_max
            #%%Append the results
            self.res_df = pd.DataFrame(index=N_sim,columns=[])

    class CHP:
        def __init__(self,id,f_nom,p_nom,q_nom,n_heat,n_ele,fuel_cost,storage_capacity,storage_capacity_min,
                     soc_ini,storage_capacity_max,storage_loss_factor,t_on,t_off,chp_on_past,n_switches,N_sim):

            self.id = id
            self.f_nom = f_nom
            self.p_nom = p_nom
            self.q_nom = q_nom
            self.n_heat = n_heat
            self.n_elec = n_ele
            self.fuel_cost = fuel_cost
            self.storage_capacity = storage_capacity
            self.storage_capacity_min = storage_capacity_min
            self.storage_capacity_max = storage_capacity_max
            self.soc_ini = soc_ini
            self.stoage_loss_factor = storage_loss_factor
            self.t_on = t_on
            self.t_off = t_off
            self.chp_on_past = chp_on_past
            self.n_switches = n_switches

            #%%Append the results in dataframe
            self.res_df = pd.DataFrame(index=N_sim, columns=[])

        # Class for Diesel/Gasoline generator
    class DG:
        def __init__(self, id, p_rated, cons_a, cons_b, price_fuel, n_min, n_max, e_coef,N_sim):
            self.id = id
            self.p_rated = p_rated  # rated power kW
            self.cons_a = cons_a  # consumption curve coefficient l/kW.h
            self.cons_b = cons_b  # consumption curve coefficient l/kW.h
            self.price_fuel = price_fuel  # $/l
            self.n_min = n_min  # min power factor
            self.n_max = n_max  # max power factor
            # self.e_max = e_max # limit of emission rate kgCO2/h
            self.e_coef = e_coef  # emission rate coeficient kgCO2/kWh

            #%% Append the results in Dataframe
            self.res = pd.DataFrame(index=N_sim, columns=[])



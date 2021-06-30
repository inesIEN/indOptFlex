
import os
from pathlib import Path

#%%Make the  absolute path to SME_LIB Folder
path = str(Path(__file__).parents[2])
os.chdir(path)

#%% Import the SME Class

from sme import SME


#%%Create the SME object
steel_plant = SME(id="kw_lahr",name="Steel_plant_sme",ort="Lahr")


#%%Create the Manufacturing system
steel_plant.create_mfs(id="MFS",desc="This is the manufacturing systems of Steel_Plant")
#Load MFS from excel file:
steel_plant.mfs.create_mfs_from_file(file_path=path+'\modelled_smes\steel_plant\model_inputs\mfs.xlsx')


#%%Create the TBS System
steel_plant.create_tbs(id="KW_LAHR_TBS",desc="This is the TBS object of the SME")

#Create the commpressed air system (CAS) inside the TBS
steel_plant.tbs.create_cas(id="CAS",desc="This is the CAS System of KW Lahr")
#Create the compressors
#steel_plant.tbs.cas.create_compressor(id="K1", p_kw=3, pressure_max=8, m3_per_min=2.57,usage_number=2,control="Binary",t_on=3,t_off=2)
steel_plant.tbs.cas.create_compressor(id="K1", p_kw=11, pressure_max=8, m3_per_min=2.57,usage_number=2,control="Binary",t_on=2,t_off=2)
steel_plant.tbs.cas.create_compressor(id="K2", p_kw=30, pressure_max=8, m3_per_min=5.31,usage_number=2,control="Binary",t_on=2,t_off=2)
#steel_plant.tbs.cas.create_compressor(id="K1", p_kw=30, pressure_max=8, m3_per_min=10,usage_number=2,control="Binary")
steel_plant.tbs.cas.create_storage_tank(id="S_air",capacity=500,soc_initial=0.25,soc_max=0.99,soc_min=0.01,
                                    pressure_max=10,p_set=8,pressure_min=7)

#Create the PCS System
steel_plant.tbs.creat_pcs(id="PCS",desc="This is the PCS System")
#Create the chillers
steel_plant.tbs.pcs.create_chiller(id="Chiller-01",p_kw=18,cooling_capacity=50,cop=2.67,mass_flow_rate=0.00155,
                            temp_supply=21,temp_return=26, t_on=2,t_off=2,control="Binary")

#Create the water storage tank
steel_plant.tbs.pcs.create_storage_tank(id='s_cws',volume=5,temp_set=20,temp_max=24,temp_min=16)

#Create the PHS Systems
steel_plant.tbs.create_phs(id="PHS",desc="This is PHS System")
steel_plant.tbs.phs.create_heat_pump(id="HP-1",p_kw=20,cop=1,control="Continuous")
steel_plant.tbs.phs.create_heat_pump(id="HP-2",p_kw=10,cop=1,control="Binary",t_on=4,t_off=2)
steel_plant.tbs.phs.create_phs_storage_tank(id="S_PHS",volume=5,k_v=0.00534,temp_set=37,temp_max=80,temp_min=35)

#Create the HVAC System
steel_plant.tbs.create_hvac(id="HVAC",desc="This is a HVAC System of the KMU",
                            space_area=500,heat_loss_coeff=0.3,thermal_capacity=16500,t_set_min=19.5,t_set=20,t_set_max=20.5)


#%% Create the Energy System of the SME
steel_plant.create_es(id="KW_LAHR_ES",desc="This is the ES Object of the SME")


#Change the path to get the weather data:
path = str(Path(__file__).parents[2])

#get the pv radiation data
steel_plant.es.get_pv_radiation_data(file_path=path+"/external_model_inputs/weather_data.xlsx")

#get the ambient temperature data
steel_plant.tbs.hvac.load_ambient_temperatures(file_path=path+"/external_model_inputs/weather_data.xlsx")

#Create the PV plants
steel_plant.es.create_pv(id="PV_1",p_rated=50)
steel_plant.es.create_pv(id="PV_2", p_rated=50)

#Create the ESS
steel_plant.es.create_ess(id="Batt_1", kwh_max=300, initial_soc=0.2, charging_rate_max=75, charge_efficiency=1,
                     soc_min=0.1,soc_max=0.8,discharge_rate_max=75, discharge_efficiency=1)

#Create the CHP Plant
steel_plant.es.create_chp(id='CHP_1',f_nom=20.5,p_nom=5.5,q_nom=12.5,n_heat=0.9,n_ele=0.9,fuel_cost=0.05,storage_capacity=75,
                      storage_capacity_max=0.95,storage_capacity_min=0.35,soc_ini =0.01, t_on=45,t_off=15,chp_on_past=[0,0,1],
                      n_switches=50,storage_loss_factor=0.0534)




#Create the market of the SME
steel_plant.create_market(id="M-01",type="Fixed",desc="Fixed Price",price_buy=3,price_sell=2,
                         electiricity_tax=2.05,demand_rate=141.62,energy_rate=0.45,eeg_surcharge=6.5,kwk_surcharge=5,
                         offshore_surcharge=0.049,abschalt_surcharge=0.011,tax=19,pltw=[24,36],retailer_cut_da=10)

#%%Create the control system of SME
# kw_lahr.create_cs(id="KW_LAHR_CS",desc="This is the control system module of the SME")
# #get the day ahead prices in cs
steel_plant.get_day_ahead_prices(file_path=path+"/external_model_inputs/market_price.xlsx",index_with_agent=False)

#%% Run the MPC Optimization with Fixed Price
# results_state_varaibes,results_input_variables,results_power,results_battery =\
#                     steel_plant.sme_opt_mpc(K=30,N_sim=96,L1=1000,L2=0,c_buy=0.30,c_sell=0.3,market_id="M-01",
#                    participation_mechanism="Fixed",cas_opt=True,pcs_opt=True,phs_opt=True,hvac_opt=True,chp_opt=True)

#%%Run the MPC Optimization with Variable price

results_state_var,results_input_variables_var,results_power_var,results_battery_var =\
                    steel_plant.sme_opt_mpc(K=30,N_sim=96,L1=1000,L2=0,c_buy=0.30,c_sell=0.3,market_id="M-01",
                   participation_mechanism="Day_ahead",cas_opt=True,pcs_opt=True,phs_opt=True,hvac_opt=True,chp_opt=True)





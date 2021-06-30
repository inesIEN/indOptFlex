import os
from pathlib import Path
#%%Make the  absolute path to SME_LIB Folder
path = str(Path(__file__).parents[2])
os.chdir(path)

#%%Import the SME Class
from sme import SME

"""
THis file is for Experimental Twin SME Model
"""

#%% Time Parameters"
time_step = 15
N_sim = 96
#%%Create the SME Object

#Create the SME object
reduced_sme = SME(id="exp_twin_sme",name="Experimental Twin",ort="HS-Offenburg",sim_time_step=time_step)

#%%Create the Manufacturing system
reduced_sme.create_mfs(id="MFS",desc="This is the manufacturing systems of Experimental Twin SME",sim_time_step=15)

#Load MFS from excel file:
reduced_sme.mfs.create_mfs_from_file(file_path=path+'\modelled_smes\exp_twin_sme\model_inputs\mfs.xlsx')

#%%Create the TBS System
reduced_sme.create_tbs(id="KW_LAHR_TBS",desc="This is the TBS object of the SME")


#%%reate the commpressed air system (CAS) inside the TBS
reduced_sme.tbs.create_cas(id="CAS",desc="This is the CAS System of KW Lahr")

#Create the compressors
reduced_sme.tbs.cas.create_compressor(id="K1", p_kw=11, pressure_max=8, m3_per_min=2.57,usage_number=2,control="Continuous",t_on=2,t_off=2)
reduced_sme.tbs.cas.create_compressor(id="K2", p_kw=30, pressure_max=8, m3_per_min=5.31,usage_number=2,control="Continuous",t_on=2,t_off=2)

#Create the storage tank
reduced_sme.tbs.cas.create_storage_tank(id="S_air",capacity=120,soc_initial=0.25,soc_max=0.99,soc_min=0.01,
                                    pressure_max=9,p_set=8,pressure_min=7)

#%%Create Cooling System
#Create the PCS System
reduced_sme.tbs.creat_pcs(id="PCS",desc="This is the PCS System")
#Create the chillers
reduced_sme.tbs.pcs.create_chiller(id="Chiller-01",p_kw=78,cooling_capacity=192,cop=2.46,mass_flow_rate=0.00155,
                            temp_supply=21,temp_return=26, t_on=1,t_off=1,control="Continuous") #Riedel PC1801

reduced_sme.tbs.pcs.create_chiller(id="Chiller-02",p_kw=63,cooling_capacity=170,cop=2.70,mass_flow_rate=0.00155,
                            temp_supply=21,temp_return=26, t_on=2,t_off=1,control="Continuous") #Ridel

#Create the water storage tank
reduced_sme.tbs.pcs.create_storage_tank(id='s_cws',volume=7,temp_set=15,temp_max=20,temp_min=7)


#%% Create the Energy System of the SME
reduced_sme.create_es(id="ES",desc="This is the ES Object of the SME")

#get the pv radiation data
reduced_sme.es.get_pv_radiation_data(file_path="external_model_inputs/weather_data.xlsx")

#Create the PV plants
reduced_sme.es.create_pv(id="PV_1",p_rated=100)

#Create the ESS
reduced_sme.es.create_ess(id="Batt_1", kwh_max=300, initial_soc=0.1, charging_rate_max=75, charge_efficiency=1,
                     soc_min=0.01,soc_max=0.95,discharge_rate_max=75, discharge_efficiency=1)

#Create the market of the SME
reduced_sme.create_market(id="M-01",type="Fixed",desc="Fixed Price",price_buy=3,price_sell=2,
                         electiricity_tax=2.05,demand_rate=141.62,energy_rate=0.45,eeg_surcharge=6.5,kwk_surcharge=5,
                         offshore_surcharge=0.049,abschalt_surcharge=0.011,tax=19,pltw=[24,36],retailer_cut_da=10)

reduced_sme.get_day_ahead_prices(file_path="external_model_inputs/market_price.xlsx",index_with_agent=False,n_days=1
                                 )

# %%Run the MILP Optimization with Variable Prices
machines_run_var,results_power_var,results_input_variables_var,results_state_varaibes_var,results_es_var, results_costs_var = \
reduced_sme.opt_milp(show_figure=False,L1=1000,L2=0,c_buy=0.30,c_sell=0.3,flex_interval=[0,20],market_id="M-01",
                      participation_mechansim="Day_ahead",p_target=200,cas_opt=True,pcs_opt=True,phs_opt=False,
                      hvac_opt=False, chp_opt=False)
# #
# #%%Run the MILP Optimization with Fixed Prices
# machines_run_var,results_power_fix,results_input_variables_fix,results_state_varaibes_fix,results_es_fix,results_costs_fix = \
# reduced_sme.opt_milp(show_figure=False,L1=1000,L2=0,c_buy=0.30,c_sell=0.3,flex_interval=[0,20],market_id="M-01",
#                       participation_mechansim="Day_ahead",p_target=200,cas_opt=True,pcs_opt=True,phs_opt=False,
#                       hvac_opt=False, chp_opt=False)











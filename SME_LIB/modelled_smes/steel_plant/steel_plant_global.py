
import os
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
#%%Make the  absolute path to SME_LIB Folder
path = str(Path(__file__).parents[2])
os.chdir(path)

#%%Import the SME Class
from sme import SME

#%%

"""
This file is supposed to be for the generation of results and analysis on the Steel_Plant_Case study for the research
paper
"""
#%% Time Parameters"
time_step = 15
N_sim = 96

#Create the SME object
steel_plant = SME(id="steel_plant",name="Kunstoff Werk Lahr",ort="Lahr",sim_time_step=15)


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
steel_plant.tbs.cas.create_compressor(id="K1", p_kw=11, pressure_max=8, m3_per_min=2.57,usage_number=2,control="Continuous",t_on=2,t_off=2)
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
steel_plant.tbs.phs.create_heat_pump(id="HP-1",p_kw=20,cop=1,control="Continous",t_on=2,t_off=2)
steel_plant.tbs.phs.create_heat_pump(id="HP-2",p_kw=10,cop=1,control="Binary",t_on=2,t_off=2)
steel_plant.tbs.phs.create_phs_storage_tank(id="S_PHS",volume=5,k_v=0.00534,temp_set=37,temp_max=80,temp_min=35)

#Create the HVAC System
steel_plant.tbs.create_hvac(id="HVAC",desc="This is a HVAC System of the KMU",
                    space_area=500,heat_loss_coeff=0.3,thermal_capacity=16500,t_set_min=19.5,t_set=20,t_set_max=20.5)


#%% Create the Energy System of the SME
steel_plant.create_es(id="KW_LAHR_ES",desc="This is the ES Object of the SME")

#get the pv radiation data
steel_plant.es.get_pv_radiation_data(file_path="external_model_inputs/weather_data.xlsx")
steel_plant.tbs.hvac.load_ambient_temperatures(file_path="external_model_inputs/weather_data.xlsx")

#Create the PV plants
steel_plant.es.create_pv(id="PV_1",p_rated=100)


#Create the ESS
steel_plant.es.create_ess(id="Batt_1", kwh_max=300, initial_soc=0.1, charging_rate_max=75, charge_efficiency=1,
                     soc_min=0.01,soc_max=0.95,discharge_rate_max=75, discharge_efficiency=1)

#Create the CHP Plant
steel_plant.es.create_chp(id='CHP_1',f_nom=20.5,p_nom=5.5,q_nom=12.5,n_heat=0.9,n_ele=0.9,fuel_cost=0.05,storage_capacity=75,
                      storage_capacity_max=0.95,storage_capacity_min=0.35,soc_ini =0.01, t_on=45,t_off=30,chp_on_past=[0,0,1],
                      n_switches=50,storage_loss_factor=0.0534)

# #Create the Diesel Generator Class
steel_plant.es.create_dg(id="DG-1", p_rated=50,cons_a=0.246,cons_b=0.08145, price_fuel=0.1,
                         n_min = 0.3, n_max=1, e_coef=3.5)

#Create the market of the SME
steel_plant.create_market(id="M-01",type="Fixed",desc="Fixed Price",price_buy=3,price_sell=2,
                         electiricity_tax=2.05,demand_rate=141.62,energy_rate=0.45,eeg_surcharge=6.5,kwk_surcharge=5,
                         offshore_surcharge=0.049,abschalt_surcharge=0.011,tax=19,pltw=[24,36],retailer_cut_da=10)

#%%Create the control system of SME
# kw_lahr.create_cs(id="KW_LAHR_CS",desc="This is the control system module of the SME")
# #get the day ahead prices in cs
steel_plant.get_day_ahead_prices(file_path="external_model_inputs/market_price.xlsx",index_with_agent=False,n_days=1)



#%% Run the MPC Optimization
# results_state_varaibes,results_input_variables,results_power,results_battery =\
#                     steel_plant.steel_sme_opt_mpc(K=30,N_sim=96,L1=1000,L2=0,c_buy=0.30,c_sell=0.3,market_id="M-01",
#                    participation_mechanism="Day_ahead",cas_opt=True,pcs_opt=True,phs_opt=True,hvac_opt=True,chp_opt=True)


#%%Perform the MILP Optimization with MFS
machines_run_fix,results_power_fix,results_input_variables_fix,results_state_varaibes_fix,results_es_fix,results_costs_fixed  = \
steel_plant.opt_milp(show_figure=False,L1=1000,L2=0,c_buy=0.30,c_sell=0.3,flex_interval=[0,20],market_id="M-01",
                      participation_mechansim="Fixed",p_target=200,cas_opt=True,pcs_opt=True,phs_opt=True)

# print("\n \n Fixed Price OPT DONE")

machines_run_var,results_power_var,results_input_variables_var,results_state_varaibes_var,results_es_var,results_costs_var = \
steel_plant.opt_milp(show_figure=False,L1=1000,L2=0,c_buy=0.30,c_sell=0.3,flex_interval=[0,20],market_id="M-01",
                      participation_mechansim="Day_ahead",p_target=200,cas_opt=True,pcs_opt=True,phs_opt=True,
                      dg_opt=False)

# print("\n \n Variable Price OPT DONE")

# machines_run_cap,results_power_cap,results_input_variables_cap,results_state_varaibes_cap,results_es_cap  = \
#     steel_plant.steel_sme_opt_milp(show_figure=False,L1=1000,L2=0,c_buy=0.30,c_sell=0.3,flex_interval=[20,40],market_id="M-01",
#                       participation_mechansim="Capacity",p_target=150,cas_opt=True,pcs_opt=True,phs_opt=True)
#
# print("\n \n Capacity OPT DONE")

# pd.plotting.register_matplotlib_converters()
# import matplotlib.pyplot as plt
# import matplotlib
# import matplotlib.ticker as ticker
# import matplotlib.dates as mdates
# import matplotlib.gridspec as gridspec
# x = steel_plant.sme_load_profiles.index.strftime("%H:%M")
# #dt = pd.date_range("00:00","23:45",freq="15min")
#
# #%%PLot the MFS with price on secondary axis
# machines_run_fix.index = pd.date_range("00:00","23:45",freq="15min")
# fig_mfs_fix = plt.figure() # Create matplotlib figure
# ax = machines_run_fix.plot(kind="bar",stacked=True,fontsize=18) # Create matplotlib axes
# ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
# width = 0.4
# results_power_fix["Electricity_price"].plot(ax=ax2,color="black",style="--",linewidth=1.5,fontsize=25)
# ax.set_ylabel('Power - kW',fontsize=30)
# # Make most of the ticklabels empty so the labels don't get too crowded
# ticklabels = ['']*len(machines_run_fix.index)
# # Every 4th ticklable shows the month and day
# ticklabels[::4] = [item.strftime('%H:%M') for item in machines_run_fix.index[::4]]
# # Every 12th ticklabel includes the year
# ticklabels[::12] = [item.strftime('%H:%M') for item in machines_run_fix.index[::12]]
# ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
# plt.gcf().autofmt_xdate()
# ax.legend(fontsize=30)
# ax.set_xlabel("Time in Hours",fontsize=35)
# ax2.set_ylabel("Electricity cost in €/kWH",fontsize=32)
# plt.show()
#
# machines_run_var.index = pd.date_range("00:00","23:45",freq="15min")
# fig_mfs_var = plt.figure() # Create matplotlib figure
# ax = machines_run_var.plot(kind="bar",stacked=True,fontsize=20) # Create matplotlib axes
# ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
# width = 0.4
# results_power_var["Electricity_price"].plot(ax=ax2,color="black",style="--",linewidth=1.5,fontsize=25)
# ax.set_ylabel('Power - kW',fontsize=30)
# # Make most of the ticklabels empty so the labels don't get too crowded
# ticklabels = ['']*len(machines_run_var.index)
# # Every 4th ticklable shows the month and day
# ticklabels[::4] = [item.strftime('%H:%M') for item in machines_run_var.index[::4]]
# # Every 12th ticklabel includes the year
# ticklabels[::12] = [item.strftime('%H:%M') for item in machines_run_var.index[::12]]
# ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
# plt.gcf().autofmt_xdate()
# ax.legend(fontsize=30)
# ax.set_xlabel("Time in Hours",fontsize=30)
# ax2.set_ylabel("Electricity cost in €/kWH",fontsize=32)
# plt.show()
#
# #%%Plot the Input variables
# matplotlib.rc('xtick', labelsize=20)
# matplotlib.rc('ytick', labelsize=20)
# plt.rcParams['ytick.major.pad'] = 2
# fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,figsize=(10,200),sharex=True)
# x = pd.date_range("00:00","23:45",freq="15min")
# font_size = 28
#
# ax1.set_ylabel('K1\nKW',fontsize=font_size)
# ax1.plot(x, results_input_variables_fix["K1"],"b-",label="Fixed price",linewidth=1.5)
# ax1.plot(x, results_input_variables_var["K1"],"k--",label="Variable price",linewidth=1.5)
# ax.xaxis.labelpad = 5
#
# ax2.set_ylabel('K2\nKW',fontsize=font_size)
# ax2.plot(x, results_input_variables_fix["K2"],"b-",label="Fixed price",linewidth=1.5)
# ax2.plot(x, results_input_variables_var["K2"],"k--",label="Variable price",linewidth=1.5)
#
#
# ax3.set_ylabel('C1\nKW',fontsize=font_size)
# ax3.plot(x, results_input_variables_fix["C1"],"b-",label="Vair,in-Fixed price")
# ax3.plot(x, results_input_variables_var["C1"],"k--",label="Variable price",linewidth=1.5)
#
#
#
# ax4.set_ylabel('HP-1\nKW',fontsize=font_size)
# ax4.plot(x, results_input_variables_fix["HP1"],"b-",label="Vair,in-Fixed price")
# ax4.plot(x, results_input_variables_var["HP1"],"k--",label="Variable price",linewidth=1.5)
#
#
#
# ax5.set_ylabel('HP-2\nKW',fontsize=font_size)
# ax5.plot(x, results_input_variables_fix["HP2"],"b-",label="Vair,in-Fixed price")
# ax5.plot(x, results_input_variables_var["HP2"],"k--",label="Variable price",linewidth=1.5)
#
#
# # Format the x axis
# ax6.xaxis.set_major_locator(mdates.HourLocator(interval=1))
# ax6.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# ##
# ax6.set_ylabel('Price \n €/KWH',fontsize=font_size)
# ax6.set_xlabel("Time in Hours",fontsize=font_size)
# ax6.plot(x, results_input_variables_fix["Electricity_price"],"b-",label="Fixed Price",linewidth=1.5)
# ax6.plot(x, results_input_variables_var["Electricity_price"],"k--",label="Variable Price",linewidth=1.5)
# handles, labels = ax6.get_legend_handles_labels()
# fig.legend(handles=handles, labels=labels, loc='upper center',fontsize=30,ncol=2)
# #fig.tight_layout()# otherwise the right y-label is slightly clipped
# plt.gcf().autofmt_xdate()
# plt.show()
#
# #%% Plot the State Variables
# params = {'mathtext.default': 'regular' }
# plt.rcParams.update(params)
# fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,figsize=(10,200),sharex=True)
# x =  pd.date_range("00:00","23:45",freq="15min")
# font_size = 28
#
# ax1.set_ylabel('$p_{CAS}$\n(bar)',fontsize=font_size)
# ax1.plot(x, results_state_varaibes_fix["p_t"],"b-",label="Fixed price",linewidth=1.5)
# ax1.plot(x, results_state_varaibes_var["p_t"],"k--",label="Variable price",linewidth=1.5)
# ax1.plot(x,[8.9]*len(x),"r--",linewidth=1)
# ax1.plot(x,[7.5]*len(x),"r--",linewidth=1)
# # ax1.plot(x,[steel_plant.tbs.cas.storages["S_air"]["pressure_min"]]*len(x),"r--",linewidth=1)
# # ax1.plot(x,[steel_plant.tbs.cas.storages["S_air"]["pressure_max"]]*len(x),"r--",linewidth=1)
#
#
# ax2.set_ylabel('$T_{PCS}$\n(°C)',fontsize=font_size)
# ax2.plot(x, results_state_varaibes_fix["Ts_pcs"],"b-",label="Fixed price",linewidth=1.5)
# ax2.plot(x, results_state_varaibes_var["Ts_pcs"],"k--",label="Variable price",linewidth=1.5)
# # ax2.plot(x,[20]*len(x),"r--",linewidth=1)
# # ax2.plot(x,[40]*len(x),"r--",linewidth=1)
# ax2.plot(x,[steel_plant.tbs.pcs.storages["s_cws"]["temp_min"]]*len(x),"r--",linewidth=1)
# ax2.plot(x,[steel_plant.tbs.pcs.storages["s_cws"]["temp_max"]]*len(x),"r--",linewidth=1)
#
# ax3.set_ylabel('$T_{PHS}$\n(°C)',fontsize=font_size)
# ax3.plot(x, results_state_varaibes_fix["Ts_phs"],"b-",label="Fixed price",linewidth=1.5)
# ax3.plot(x, results_state_varaibes_var["Ts_phs"],"k--",label="Variable price",linewidth=1.5)
# ax3.plot(x,[30]*len(x),"r--",linewidth=1)
# ax3.plot(x,[50]*len(x),"r--",linewidth=1)
# # ax3.plot(x,[steel_plant.tbs.phs.storages["S_PHS"]["temp_min"]]*len(x),"r--",linewidth=1)
# # ax3.plot(x,[steel_plant.tbs.phs.storages["S_PHS"]["temp_max"]]*len(x),"r--",linewidth=1)
#
# ax4.set_ylabel('$T_{indoor}$\n(°C)',fontsize=font_size)
# ax4.plot(x, results_state_varaibes_fix["Tr"],"b-",label="Fixed price",linewidth=1.5)
# ax4.plot(x, results_state_varaibes_var["Tr"],"k--",label="Variable price",linewidth=1.5)
# ax4.plot(x,[steel_plant.tbs.hvac.t_set_min]*len(x),"r--",linewidth=1)
# ax4.plot(x,[steel_plant.tbs.hvac.t_set_max]*len(x),"r--",linewidth=1)
#
# # Format the x axis
# ax5.xaxis.set_major_locator(mdates.HourLocator(interval=1))
# ax5.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# ##
# ax5.set_ylabel('Price \n €/KWH',fontsize=font_size)
# ax5.set_xlabel("Time in Hours",fontsize=font_size)
# ax5.plot(x, results_input_variables_fix["Electricity_price"],"b-",label="Fixed Price",linewidth=1.5)
# ax5.plot(x, results_input_variables_var["Electricity_price"],"k--",label="Variable Price",linewidth=1.5)
# handles, labels = ax5.get_legend_handles_labels()
# fig.legend(handles=handles, labels=labels, loc='upper center',fontsize=30,ncol=2)
# #fig.tight_layout()# otherwise the right y-label is slightly clipped
# plt.gcf().autofmt_xdate()
# plt.show()
#
# #%% Plot the ES Systems
# fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,figsize=(10,300),sharex=True)
# x =  pd.date_range("00:00","23:45",freq="15min")
# font_size = 28
#
# ax1.set_ylabel('$E_{BES}$\nKWH',fontsize=font_size)
# ax1.plot(x, results_es_fix["E_batt"],"b-",label="Fixed price",linewidth=1.5)
# ax1.plot(x, results_es_var["E_batt"],"k--",label="Variable price",linewidth=1.5)
# ax1.plot(x,[steel_plant.es.ess["Batt_1"]["soc_min"]*steel_plant.es.ess["Batt_1"]["kwh_max"]]*len(x),"r--",linewidth=1)
# ax1.plot(x,[steel_plant.es.ess["Batt_1"]["soc_max"]*steel_plant.es.ess["Batt_1"]["kwh_max"]]*len(x),"r--",linewidth=1)
#
#
# ax2.set_ylabel('$P_{ch,BES}$\nkW',fontsize=font_size)
# ax2.plot(x, results_es_fix["P_batt_ch"],"b-",label="Fixed price",linewidth=1.5)
# ax2.plot(x, results_es_var["P_batt_ch"],"k--",label="Variable price",linewidth=1.5)
#
# ax3.set_ylabel('$P_{disch,BES}$\nkW',fontsize=font_size)
# ax3.plot(x, results_es_fix["P_batt_disch"],"b-",label="Fixed price",linewidth=1.5)
# ax3.plot(x, results_es_var["P_batt_disch"],"k--",label="Variable price",linewidth=1.5)
#
# ax4.set_ylabel('$P_{CHP}$\nkW',fontsize=font_size)
# ax4.plot(x, results_es_fix["CHP_elec"],"b-",label="Fixed price",linewidth=1.5)
# ax4.plot(x, results_es_var["CHP_elec"],"k--",label="Variable price",linewidth=1.5)
#
# ax5.set_ylabel('$Q_{CHP}$\nkW',fontsize=font_size)
# ax5.plot(x, results_es_fix["CHP_q"],"b-",label="Fixed price",linewidth=1.5)
# ax5.plot(x, results_es_var["CHP_q"],"k--",label="Variable price",linewidth=1.5)
#
# ax6.set_ylabel('$P_{PV}$\nkW',fontsize=font_size)
# ax6.plot(x, results_es_fix["P_PV"],"r-")
#
# # Format the x axis
# ax7.xaxis.set_major_locator(mdates.HourLocator(interval=1))
# ax7.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# ax7.set_ylabel('Price \n €/KWH',fontsize=font_size)
# ax7.set_xlabel("Time in Hours",fontsize=font_size)
# ax7.plot(x, results_es_fix["Electricity_price"],"b-",label="Fixed Price",linewidth=1.5)
# ax7.plot(x, results_es_var["Electricity_price"],"k--",label="Variable Price",linewidth=1.5)
# handles, labels = ax7.get_legend_handles_labels()
# fig.legend(handles=handles, labels=labels, loc='upper center',fontsize=font_size,ncol=2)
# #fig.tight_layout()# otherwise the right y-label is slightly clipped
# plt.show()
# plt.gcf().autofmt_xdate()
# #%% Plot the Flexibility Profiles
# import numpy as np
# fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,figsize=(10,300),sharex=True)
# #x = steel_plant.sme_load_profiles.index.strftime("%H:%M")
# x = pd.date_range("00:00","23:45",freq="15min")
# font_size = 28
# axes_limit = [-240,240]
#
#
# ax1.set_ylabel('$P_{Grid}$\nKW',fontsize=font_size)
# y = np.array(results_power_fix["P_Grid"]-results_power_var["P_Grid"])
# mask1 = y <0
# mask2 = y >= 0
# ax1.bar(x[mask1], y[mask1], color="red",label="Negative Flexibility")
# ax1.bar(x[mask2], y[mask2], color="blue",label="Positive Flexibility")
# ax1.set_ylim(axes_limit)
#
#
# ax2.set_ylabel('$P_{CAS}$\nKW',fontsize=font_size)
# y = np.array(results_power_fix["P_CAS"]-results_power_var["P_CAS"])
# mask1 = y <0
# mask2 = y >= 0
# ax2.bar(x[mask1], y[mask1], color="red",label="Negative Flexibility")
# ax2.bar(x[mask2], y[mask2], color="blue",label="Positive Flexibility")
# ax2.set_ylim(axes_limit)
#
# ax3.set_ylabel('$P_{PCS}$\nKW',fontsize=font_size)
# y = np.array(results_power_fix["P_PCS"]-results_power_var["P_PCS"])
# mask1 = y <0
# mask2 = y >= 0
# ax3.bar(x[mask1], y[mask1], color="red",label="Negative Flexibility")
# ax3.bar(x[mask2], y[mask2], color="blue",label="Positive Flexibility")
# ax3.set_ylim(axes_limit)
#
# ax4.set_ylabel('$P_{PHS}$\nKW',fontsize=font_size)
# y = np.array(results_power_fix["P_PHS"]-results_power_var["P_PHS"])
# mask1 = y <0
# mask2 = y >= 0
# ax4.bar(x[mask1], y[mask1], color="red",label="Negative Flexibility")
# ax4.bar(x[mask2], y[mask2], color="blue",label="Positive Flexibility")
# ax4.set_ylim(axes_limit)
#
# ax5.set_ylabel('P_Batt\nKW',fontsize=font_size)
# #Calculation P_battery
# P_batt_fixed = results_es_fix["P_batt_ch"]-results_es_fix["P_batt_disch"]
# P_batt_var = results_es_var["P_batt_ch"]-results_es_var["P_batt_disch"]
# y = np.array(P_batt_fixed-P_batt_var)
# mask1 = y <0
# mask2 = y >= 0
# ax5.bar(x[mask1], y[mask1], color="red",label="Negative Flexibility")
# ax5.bar(x[mask2], y[mask2], color="blue",label="Positive Flexibility")
# ax5.set_ylim(axes_limit)
#
# # ax5.xaxis.set_major_locator(mdates.HourLocator(interval=1))
# # ax5.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# ax6.set_ylabel('Price \n €/KWH',fontsize=font_size)
# ax6.set_xlabel("Time steps",fontsize=font_size)
# ax6.plot(x, results_es_fix["Electricity_price"],"b-",label="Fixed Price",linewidth=1.5)
# ax6.plot(x, results_es_var["Electricity_price"],"k--",label="Variable Price",linewidth=1.5)
# handles, labels = ax6.get_legend_handles_labels()
# fig.legend(handles=handles, labels=labels, loc='upper center',fontsize=font_size,ncol=2)
# plt.gcf().autofmt_xdate()
# #fig.tight_layout()# otherwise the right y-label is slightly clipped
# plt.show()
#
# #%%Plot Flexibility Stacked Bar Plot
# results_flex = pd.DataFrame(columns=["MFS","CAS","PCS","PHS","CHP","BES"])
# results_flex["MFS"] = results_power_fix["P_MFS"]-results_power_var["P_MFS"]
# results_flex["CAS"] = results_power_fix["P_CAS"]-results_power_var["P_CAS"]
# results_flex["PCS"] = results_power_fix["P_PCS"]-results_power_var["P_PCS"]
# results_flex["PHS"] = results_power_fix["P_PHS"]-results_power_var["P_PHS"]
# results_flex["CHP"] = results_es_fix["CHP_elec"]-results_es_var["CHP_elec"]
# results_flex["BES"] = (results_es_fix["P_batt_ch"]-results_es_fix["P_batt_disch"]) - \
#                        (results_es_var["P_batt_ch"]-results_es_var["P_batt_disch"])
# results_flex.index = x
# fig_flex_mix = plt.figure() # Create matplotlib figure
# ax = fig_flex_mix.add_subplot(111) # Create matplotlib axes
# ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
# width = 0.4
#
# results_flex.plot.bar(ax=ax,stacked=True)
#
# results_power_fix["Electricity_price"].plot(ax=ax2,color="black",style="--",linewidth=1.5,label="Fix Price",fontsize=25)
# results_power_var["Electricity_price"].plot(ax=ax2,color="blue",style="--",linewidth=1.5,label="Variable Price",fontsize=25)
#
#
#
# ticklabels = ['']*len(machines_run_var.index)
# # Every 4th ticklable shows the month and day
# ticklabels[::4] = [item.strftime('%H:%M') for item in machines_run_var.index[::4]]
# # Every 12th ticklabel includes the year
# ticklabels[::12] = [item.strftime('%H:%M') for item in machines_run_var.index[::12]]
# ax2.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
# plt.gcf().autofmt_xdate()
# ax.set_ylabel('Flexibility - kW',fontsize=28)
# ax2.set_ylabel("Electricity cost in €/KWH",fontsize=28)
# ax.set_xlabel("Time in Hours",fontsize=28)
# #Legends
# h1, l1 = ax.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()
# ax.legend(h1+h2, l1+l2, loc=2, fontsize=24)
#
# plt.show()
#
# #%% PLot the Flexibility Profiles
# import numpy as np
# fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,figsize=(10,300),sharex=True)
# x = steel_plant.sme_load_profiles.index.strftime("%H:%M")
# font_size = 28
# ax1.set_ylabel("$P_{Grid}$\nkW",fontsize=font_size)
# ax1.plot(x, results_power_fix["P_Grid"],"b-",label="Fixed price",linewidth=1.5)
# ax1.plot(x, results_power_var["P_Grid"],"k--",label="Variable price",linewidth=1.5)
#
# ax2.set_ylabel('$P_{SME}$\nkW',fontsize=font_size)
# ax2.plot(x, results_power_fix["P_demand"],"b-",label="Fixed price",linewidth=1.5)
# ax2.plot(x, results_power_var["P_demand"],"k--",label="Variable price",linewidth=1.5)
#
# ax3.set_ylabel('Flexibility\nkW',fontsize=font_size)
# y = np.array(results_power_fix["P_Grid"]-results_power_var["P_Grid"])
# cc = ["colors"]*len(y)
# for n,val in enumerate(y):
#     if val<0:
#         cc[n] = "red"
#         label = "Negative Flexibility"
#     elif val>=0:
#         cc[n] = "blue"
#         label = "Positive Flexibility"
# ax3.bar(x, y, color=cc,label=label)
# ax4.legend()
#
# # ax3.bar(x,y,label="Flexibility",linewidth=1.5)
# # Make most of the ticklabels empty so the labels don't get too crowded
# ticklabels = ['']*len(machines_run_var.index)
# # Every 4th ticklable shows the month and day
# ticklabels[::4] = [item.strftime('%H:%M') for item in machines_run_var.index[::4]]
# # Every 12th ticklabel includes the year
# ticklabels[::12] = [item.strftime('%H:%M') for item in machines_run_var.index[::12]]
# ax4.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
# plt.gcf().autofmt_xdate()
# ax4.set_ylabel('Price \n €/KWH',fontsize=font_size)
# ax4.set_xlabel("Time in Hours",fontsize=font_size)
# ax4.plot(x, results_es_fix["Electricity_price"],"b-",label="Fixed Price",linewidth=1.5)
# ax4.plot(x, results_es_var["Electricity_price"],"k--",label="Variable Price",linewidth=1.5)
# handles, labels = ax4.get_legend_handles_labels()
# fig.legend(handles=handles, labels=labels, loc='upper center',fontsize=font_size,ncol=2)
# plt.gcf().autofmt_xdate()
# #fig.tight_layout()# otherwise the right y-label is slightly clipped
# plt.show()
#
#
# """
# ax2.set_xlabel('Time (min)',fontsize=6)
# ax2.set_ylabel('KW',fontsize=6)
# ax2.plot(results_input_variables.index, results_input_variables["p_batt_ch"],linewidth=2,label="P_batt_charge")
# ax2.plot(results_input_variables.index, results_input_variables["p_batt_disch"],linewidth=2,label="P_batt_discharge")
# ax2.legend()
# ax2.set_title("Charging and discharging of the Battery")
#
# ax3.set_xlabel('Time (min)',fontsize=8)
# ax3.set_ylabel('€/KWH',fontsize=8)
# ax3.plot(results_input_variables.index, results_input_variables["Electricity_price"],linewidth=2,label="Price",color="orange")
# ax3.set_title("Electricity cost in €/KWH ")
#
# fig.tight_layout()# otherwise the right y-label is slightly clipped
# plt.show()
#
# # #%%Plot the State Variables
# # fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,figsize=(10,15))
# #
# # ax1.set_xlabel('Time',fontsize=8)
# # ax1.set_ylabel('Bar',fontsize=8)
# # ax1.plot(results_state_varaibes.index, results_state_varaibes["Pressure"],linewidth=2,label="Pressure - Vcas")
# # ax1.legend()
# # ax1.set_title("Pressure in Bar")
# #
# # ax2.set_xlabel('Time',fontsize=8)
# # ax2.set_ylabel('m3',fontsize=8)
# # ax2.plot(results_state_varaibes.index, results_state_varaibes["V_air"],linewidth=2,label="Volume of CA - m3")
# # ax2.legend()
# # ax2.set_title("Volume of CAS Tank in m3")
# #
# # ax3.set_xlabel('Time',fontsize=6)
# # ax3.set_ylabel('°C',fontsize=6)
# # ax3.plot(results_state_varaibes.index, results_state_varaibes["Temperature"],linewidth=2,label="Temp -°C")
# # ax3.legend()
# # ax3.set_title("Temperature inside the Water Storage Tank")
# #
# # ax4.set_xlabel('Time',fontsize=6)
# # ax4.set_ylabel('KWH',fontsize=6)
# # ax4.plot(results_state_varaibes.index, results_state_varaibes["E_battery"],linewidth=2,label="KWH")
# # ax4.legend()
# # ax4.set_title("Energy content of the Battery in KWH")
# #
# # ax5.set_xlabel('Time (min)',fontsize=8)
# # ax5.set_ylabel('€/KWH',fontsize=8)
# # ax5.plot(results_state_varaibes.index, results_state_varaibes["Electricity_price"],linewidth=2,label="Price",color="orange")
# # ax5.set_title("Electricity cost in €/KWH ")
# #
# # fig.tight_layout()# otherwise the right y-label is slightly clipped
# # plt.show()
#
# """










import random
from collections import defaultdict

import numpy as np
import pandas as pd
import pylab as pl
from gurobipy import Model, GRB, quicksum, and_, max_ , min_, abs_

#MFS Formulation
from cs.opt_formulations_mpc.mfs_opt_formulation import mfs_opt_formulation
from cs.opt_formulations_mpc.mfs_opt_formulation import mfs_opt_formulation_scheduling
#from cs.opt_formulations_mpc.mfs_opt_formulation_relaxed import mfs_opt_formulation

#Import MPC Formulation(CAS;PCS:PHS)
from cs.opt_formulations_mpc.cas_formulation import cas_opt_formulation_mpc
from cs.opt_formulations_mpc.phs_formulation import phs_opt_formulation_energy_based_mpc
from cs.opt_formulations_mpc.pcs_formulation import pcs_opt_formulation_mpc
from cs.opt_formulations_mpc.hvac_formulation import hvac_opt_formulation_mpc
from cs.opt_formulations_mpc.es_formulation import es_pv_opt_formulation_mpc
#Battery Formulation
from cs.opt_formulations_mpc.es_formulation import es_bess_opt_formulation_mpc
from cs.opt_formulations_mpc.es_formulation import chp_formulation_mpc

#Import MILP Formulation(CAS;PCS:PHS)
from cs.opt_formulations_milp.cas_formulation import cas_opt_formulation
from cs.opt_formulations_milp.phs_formulation import phs_opt_formulation_energy_based

from cs.opt_formulations_milp.pcs_formulation import pcs_opt_formulation
from cs.opt_formulations_milp.hvac_formulation import hvac_opt_formulation
from cs.opt_formulations_milp.es_formulation import es_pv_opt_formulation
#CHP Formulation
from cs.opt_formulations_milp.es_formulation import chp_formulation
#DG Formulation
from cs.opt_formulations_milp.es_formulation import es_dg_opt_formulation
#Battery Formulation
from cs.opt_formulations_milp.es_formulation import es_bess_opt_formulation


#Import the functions for appending the results
from cs.append_results.append_results import mfs_opt_results
from cs.append_results.append_results import cas_opt_results
from cs.append_results.append_results import pcs_opt_results
from cs.append_results.append_results import phs_opt_results
from cs.append_results.append_results import es_bess_opt_results
from cs.append_results.append_results import es_chp_opt_results
from cs.append_results.append_results import hvac_opt_results
from cs.append_results.append_results import es_pv_results


class CS():

    def __init__(self,id,desc):
        self.id = id
        self.desc = desc
        self.day_ahead_prices = None

    def get_day_ahead_prices(self, file_path="external_model_inputs/market_price.xlsx", n_days=2, index_with_agent=True,
                             N_sim=96):

        prices_original = pd.read_excel(file_path)
        prices = pd.DataFrame(columns=["price"])

        # Create a dataframe based on simulation time_step, For e.g 15min--->sim_time_step
        if index_with_agent == True:
            prices = pd.DataFrame(columns=["price"],
                                  index=self.sme_load_profiles.index[0:N_sim])  # As we are loading one day --- 96 steps
            if self.sim_time_step == 15:
                range_x = int(len(prices) / len(prices_original))
            else:
                range_x = int((len(prices)) / len(prices_original))

            for p in range(len(prices_original)):
                p1 = p * range_x
                p2 = p1 + range_x

                prices["price"][p1:p2] = prices_original["price"][p] / 1000
        else:
            if len(prices) == int(self.sim_time_step):
                prices = prices_original["price"] / 1000
            else:
                prices = list(np.repeat(list(prices_original["price"]), 15 / int(self.sim_time_step)))
                prices = [x / 1000 for x in prices]

        if index_with_agent == True:
            prices = (prices["price"]).values.tolist()
        else:
            prices = prices

        # prices = (prices["price"]).values.tolist()
        prices = prices * n_days
        self.day_ahead_prices = prices


    def opt_milp(self,show_figure=True,L1=500,L2=100,c_buy=0.25,c_sell=0.3,flex_interval=[16,28],market_id="M-01",
                           participation_mechansim="Fixed",p_target=100,cas_opt=True,pcs_opt=True,phs_opt=True,hvac_opt=True,
                           chp_opt=True,dg_opt=False):

        # %%Create the optimization model framework
        mdl = Model('flex_start')
        mdl.__len__ = 1

        # Number of time steps
        t_hr = [t for t in self.hr]  # time step in hr
        T = self.N_sim  # time step in 0,1,2,3,..
        K = T

        # %%Get the market information
        # Flex Interval
        flex_interval = self.markets["M-01"].pltw
        pltw = list(range(flex_interval[0], flex_interval[1])) + list(
            range(flex_interval[0] + 96, flex_interval[1] + 96))

        p_fix = self.markets[market_id].electiricity_tax + self.markets[market_id].eeg_surcharge + \
                self.markets[market_id].abschalt_surcharge + self.markets[market_id].kwk_surcharge + \
                self.markets[market_id].kwk_surcharge

        p_fix = p_fix / 100  # convert cents to euros

        tax = self.markets[market_id].tax / 100

        DR = self.markets[market_id].demand_rate
        ER = self.markets[market_id].energy_rate / 100

        if participation_mechansim == "Day_ahead":
            prices = [(x + p_fix) * (1 + tax) for x in self.day_ahead_prices]
            self.markets[market_id].prices_variable["Price"] = prices
            # prices = list(self.markets[market_id].price_variable)*n_days
        else:
            prices = (self.markets[market_id].price_buy / 100 + p_fix) * (1 + tax)
            prices = [prices] * len(K)
            self.markets[market_id].prices_fixed["Price"] = prices

        # %%Call the MFS Optimization System

        # If MFS Optimization is to be included, the de-comment the following two lines and comments the rest section
        # which reads the data from the excel file.

        p_mfs_t, air_demand_t, q_cool_demand_t, q_heat_demand_t, tasks, machines, products, storages, p_r_t, N_i_t, Y_i_t, R_i_t, S_i_t, R_r_t, R_s_t = \
            mfs_opt_formulation(self=self, mdl=mdl, T=T)

        # If MFS Optimization needs not be included, and the saved data from optimization results can also be
        # added as parameters.

        # mfs_data = pd.read_excel("mfs_data.xlsx")
        # p_mfs_t = mfs_data["p_mfs"]
        # air_demand_t = mfs_data["air_demand"]
        # q_cool_demand_t = mfs_data["q_cool_demand"]
        # q_heat_demand_t = mfs_data["q_heat_demand"]
        print("\n\n\n MFS OPTIMIZATION DONE \n\n\n")

        # %%Call the CAS System Optimization
        if cas_opt == True:
            p_cas_t, Pel_k, v_out, v_in, v_stor_t, p_t, N_k_t, compressors = cas_opt_formulation(self=self, mdl=mdl,
                                                                                                 K=K,
                                                                                                 mode="MILP", x_o={},
                                                                                                 from_agent=False,
                                                                                                 air_demand_opt=air_demand_t)
        elif cas_opt == False:
            # Total electrical power of the CAS Systems
            p_cas_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_t")
            mdl.addVars(p_cas_t[t] == 0 for t in K)

        # %%Call the HVAC Optimization
        if hvac_opt == True:
            T_r, T_amb, Qc_hvac, Qh_hvac = hvac_opt_formulation(self=self, mode="MILP", mdl=mdl, x_o={}, K=K)
        elif hvac_opt == False:
            Qc_hvac = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="Qc_hvac")
            Qh_hvac = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="Qh_hvac")
            mdl.addVars(Qc_hvac[t] == 0 for t in K)
            mdl.addVars(Qh_hvac[t] == 0 for t in K)

        # %%Call the PCS System Optimization
        if pcs_opt == True:
            p_pcs_t, Pel_c, Ts_pcs, q_tes_t, qc_in_t, qc_out_t, chillers, N_c_t, delta_t = \
                pcs_opt_formulation(self=self, mdl=mdl, K=K, mode="MILP", x_o={}, from_agent=False,
                                    cooling_demand_t=q_cool_demand_t, Qc_hvac=Qc_hvac)
        elif pcs_opt == False:
            p_pcs_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_pcs_t")
            mdl.addVars(p_pcs_t[t] == 0 for t in K)

        # %%CHP Formulation as it is needed by the Heating system
        if chp_opt == True:
            q_chp_in, p_chp_use, p_chp_sold, chp_cost, chp_on, chp_off = chp_formulation(self=self, mdl=mdl, K=K,
                                                                                         x_o={},
                                                                                         mode="MILP",
                                                                                         chp_on_past=[0, 0, 1])
        elif chp_opt == False:
            chp_cost = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_pcs_t")
            p_chp_use = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_pcs_t")
            p_chp_sold = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_pcs_t")
            q_chp_in = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_pcs_t")

            mdl.addConstrs(chp_cost[t] == 0 for t in K)
            mdl.addConstrs(p_chp_use[t] == 0 for t in K)
            mdl.addConstrs(p_chp_sold[t] == 0 for t in K)
            mdl.addConstrs(q_chp_in[t] == 0 for t in K)

        # %%Call the PHS System Optimization
        if phs_opt == True:
            N_h_t,p_phs_t, Pel_h, H_test_t, T_phs_t, Qh_in_t, Qmfs_out_t, Qchp_in_t = \
                phs_opt_formulation_energy_based(self=self, mdl=mdl, K=K, x_o={}, from_agent=False,
                                                 heating_demand_t=q_heat_demand_t,
                                                 q_chp_in=q_chp_in, Qh_hvac=Qh_hvac, mode="MILP", hvac_opt=hvac_opt)
        elif phs_opt == False:
            p_phs_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_phs_t")
            mdl.addVars(p_phs_t[t] == 0 for t in K)

        # %% Call the ES System Optimization

        # %%Battery Energy Systems
        # todo : Implment when a BES doesn't exists
        p_batt_ch, p_batt_disch, p_batt_use, p_batt_sold, E_batt, y_batt = es_bess_opt_formulation(self=self, mdl=mdl,
                                                                                                   K=K,
                                                                                                   mode="MILP", x_o={})

        # %%PV Formulation
        pv_prod, p_pv_use, p_pv_sold = es_pv_opt_formulation(self=self, mdl=mdl, K=K, mode="MILP")

        # %%DG OPT Formulation
        if dg_opt == True:
            p_dg_out, fuel_cons_dg, e_dg_rate, n_dg_gen, p_dg_sold, p_dg_use, p_dg_cost, N_dg = es_dg_opt_formulation(
                self=self, mdl=mdl, K=K)
        elif dg_opt == False:
            p_dg_out = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_dg_out")
            p_dg_use = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_dg_out")
            p_dg_sold = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_dg_out")
            p_dg_cost = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_dg_out")
            mdl.addConstrs((p_dg_out[t] == 0 for t in K))
            mdl.addConstrs((p_dg_use[t] == 0 for t in K))
            mdl.addConstrs((p_dg_sold[t] == 0 for t in K))
            mdl.addConstrs((p_dg_cost[t] == 0 for t in K))

        # %% Define the Grid Variables
        p_grid = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=0, ub=L1, name="p_grid")
        p_grid_sold = mdl.addVars(K, vtype=GRB.CONTINUOUS, lb=0, ub=L2, name="p_sold")
        y_grid = mdl.addVars(len(K), vtype=GRB.BINARY, name="y_grid")

        # %% SME Load Demand
        demand_kw = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="demand_kw")

        # Energy/Power Management
        mdl.addConstrs(demand_kw[t] == p_cas_t[t] + p_mfs_t[t] + p_phs_t[t] + p_pcs_t[t] for t in K)

        # Power Balance (Pgrid, Ppv, Pchp, Pdg, Pbatt, demand)
        mdl.addConstrs(p_grid[t] + p_pv_use[t] + p_chp_use[t] + p_dg_use[t] + p_batt_disch[t] ==
                       demand_kw[t] + p_batt_ch[t] for t in K)

        # Total power injected into the Grid
        mdl.addConstrs(p_grid_sold[t] == p_batt_sold[t] + p_pv_sold[t] + p_chp_sold[t] + p_dg_sold[t] for t in K)
        # Power transaction constraints
        mdl.addConstrs(p_grid[t] <= L1 * (y_grid[t]) for t in K)
        mdl.addConstrs(p_grid_sold[t] <= L2 * (1 - y_grid[t]) for t in K)

        # %%# %% Objective function

        mdl.setObjective(quicksum((p_grid[t] * (prices[t]) * (self.sim_time_step / 60)) -
                                  (p_grid_sold[t] * c_sell * (self.sim_time_step / 60)) for t in K) +
                         quicksum(chp_cost[t] for t in K) + quicksum(p_dg_cost[t] for t in K))

        # mdl.setObjective(quicksum(Qh_hvac[t]*(self.sim_time_step)/60 for t in K))

        # elif participation_mechansim == "Capacity":
        #     t0 = flex_interval[0]
        #     t1 = flex_interval[1] + 1
        #     interval = list(range(t0, t1))
        #
        #     y1 = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="Y1")
        #     y2 = mdl.addVars(len(T), vtype=GRB.CONTINUOUS,name="Y2")
        #     y3 = mdl.addVars(len(T), vtype=GRB.CONTINUOUS,name="Y3")
        #
        #
        #     #calculation of the difference between Pgrid and Ptarget
        #     mdl.addConstrs(y1[t] == p_grid[t] - p_target for t in interval)
        #
        #     #Taking the absoulte of the (Pgrid-Ptarget)
        #     mdl.addConstrs(y2[t] ==  abs_(y1[t]) for t in interval)
        #
        #     #Finding the factors --> (1-abs(Pgrid-Ptarget)/Ptarget)
        #     mdl.addConstrs(y3[t] == 1-(y2[t]/p_target) for t in interval)
        #     # # mdl.addConstrs(flex[t] == 0 for t in T if t not in interval)
        #     #
        #     # mdl.setObjective(quicksum((p_grid[t] * (c_buy) * 15 / 60) for t in T if t not in interval) -
        #     #                  quicksum((1 - y2[t]/p_target) * 10 for t in interval if t in T))

        # mdl.setObjective(quicksum(p_grid[t] * (c_buy) * (self.sim_time_step / 60)  for t in K)-
        #                  quicksum(y3[t]*0.8*15/60 for t in interval))

        # mdl.setObjective(quicksum((p_grid[t] * (c_buy) * self.sim_time_step / 60) for t in K) +
        #                  1000 * quicksum(
        #     (p_grid[t] - p_target) * (p_grid[t] - p_target) for t in interval if t in K))
        # %% Solve the model
        # Senseofthemode
        mdl.modelSense = GRB.MINIMIZE

        mdl.setParam("MIPGap", 1)
        # mdl.setParam("TimeLimit",60)

        mdl.optimize()

        # Battery Capacity Model formulation
        # p_f_ch, p_f_disch, E_f = es_bess_opt_formulation_capp(self, K, E_batt, p_batt_ch, p_batt_disch,y_batt)

        # %% Extract the results and append them in the relevent sections
        # #MFS
        mfs_opt_results(self, K, p_mfs_t, p_r_t, R_r_t, N_i_t, R_s_t, mode="MILP")

        # CAS
        if cas_opt == True:
            cas_opt_results(self=self,K=K, p_cas_t=p_cas_t, N_k_t=N_k_t, Pel_k=Pel_k, v_stor_t=v_stor_t, v_in=v_in,
                            v_out=v_out, p_t=p_t)
        # PCS
        if pcs_opt == True:
            pcs_opt_results(self, K, p_pcs_t, N_c_t=N_c_t, Pel_c=Pel_c, t_pcs_t=Ts_pcs, qc_out_t=qc_out_t,
                                                                        qc_in_t=qc_in_t, Qc_hvac=Qc_hvac)

        # PHS
        if phs_opt == True:
            phs_opt_results(self,K,p_phs_t, Pel_h=Pel_h,N_h_t=N_h_t, H_test_t=H_test_t, T_phs_t=T_phs_t,
                            Qh_in_t=Qh_in_t,Qchp_in_t=Qchp_in_t,Qmfs_out_t=Qmfs_out_t,Qh_hvac=Qh_hvac)

        #BESS
        es_bess_opt_results(self=self, K=T, E_batt=E_batt, p_batt_ch=p_batt_ch, p_batt_disch=p_batt_disch,
                                                            p_batt_sold=p_batt_sold, p_batt_use=p_batt_use)

        #PV
        es_pv_results(self=self,K=T,pv_prod=pv_prod,p_pv_use=p_pv_use,p_pv_sold=p_pv_sold)

        #CHP
        if chp_opt == True:
            es_chp_opt_results(self=self,K=T,p_chp_use=p_chp_use,p_chp_sold=p_chp_sold,q_chp_in=q_chp_in,chp_cost=chp_cost)

        #HVAC
        if hvac_opt == True :
            hvac_opt_results(self=self,K=T,Qh_hvac=Qh_hvac,Qc_hvac=Qc_hvac,T_amb=T_amb,T_r=T_r)

        #
        # # HVAC Results
        # if hvac_opt == True:
        #     hvac_opt_results(self, K, Qh_hvac, Qc_hvac, T_amb, T_r)
        #
        # # ES- BESS
        # es_bess_opt_results(self, K, E_batt, p_batt_ch, p_batt_disch, p_batt_sold, p_batt_use)
        #
        # # ES - CHP
        # es_chp_opt_results(self, K, p_chp_use, p_chp_sold, q_chp_in, chp_cost)
        #
        # # ES - PV results
        # es_pv_results(self, K, pv_prod, p_pv_use, p_pv_sold)

        # %%MFS RESULTS
        # task_schedule = {}
        # for i in tasks:
        #     task_schedule[i] = [N_i_t[i, t].x for t in T]
        #
        # self.mfs.opt_task_schedule = task_schedule
        #
        # #Get the dataframe for each task
        #
        # task_sch_dict = {}
        #
        # for i in tasks:
        #     task_sch_dict[i] = pd.DataFrame(columns=["HR","N_i_t","Y_i_t","R_i_t","S_i_t"])
        #
        # for i in tasks:
        #     task_sch_dict[i]["HR"] = self.hr
        #     task_sch_dict[i]["N_i_t"] = [N_i_t[i, t].x for t in T]
        #     task_sch_dict[i]["S_i_t"] = [S_i_t[i, t].x for t in T]
        #     task_sch_dict[i]["Y_i_t"] = [Y_i_t[i, t].x for t in T]
        #     task_sch_dict[i]["R_i_t"] = [R_i_t[i, t].x for t in T]

        # df_task = task_sch_dict["SPR"]

        # Plotting the Machine Profile from MFS Optimization
        machines_run = pd.DataFrame(columns=[])
        for m in machines:
            machines_run[m] = [p_r_t[m, t].x for t in T]
        machines_run.plot.bar(stacked=True)

        A = "STOP"

        # def add_to_anylsis_df():
        #     self.mfs.load_profile_df = pd.DataFrame()
        #     self.mfs.load_profile_df["Hour"] = self.hr
        #     # self.mfs.price_df["Hour"] = self.hr
        #     # self.mfs.price_df["Price"] = self.day_ahead_prices
        #     self.mfs.products_state_df = pd.DataFrame()
        #     self.mfs.storages_state_df = pd.DataFrame()
        #     self.mfs.products_state_df["Hour"] = self.hr
        #     self.mfs.storages_state_df["Hour"] = self.hr
        #
        #     # List of Machines in MFS System
        #     for m in machines:
        #          self.mfs.load_profile_df[m] = [p_r_t[m, t].x for t in T]
        #
        #     #Append total Load of MFS system
        #     self.mfs.load_profile_df.set_index("Hour")
        #     self.mfs.load_profile_df["total"] = \
        #         self.mfs.load_profile_df[list(self.mfs.load_profile_df.columns)].sum(axis=1)
        #
        #     #Append the total electricity costs
        #     self.mfs.load_profile_df["cost"] = [(a * self.mfs.sim_time_step / 60) * b / 1000 for a, b in
        #                                             zip(list(self.mfs.load_profile_df["total"]),
        #                                                 self.day_ahead_prices)]
        #
        #     # Add the states of resource to state_df
        #     for r in products:
        #         self.mfs.products_state_df[r] = [R_r_t[r, t].x for t in T]
        #     for s in storages:
        #         self.mfs.storages_state_df[r] = [R_s_t[s, t].x for t in T]
        # add_to_anylsis_df()
        #
        # # self.mfs.products_state_df.plot(subplots=True)
        # # self.mfs.storages_state_df.plot(subplots=True)
        #
        #
        # def print_opt_results():
        #     import pandas as pd
        #     import plotly.graph_objects as go
        #     from plotly.subplots import make_subplots
        #     cols_tasks = tasks
        #     res_df = pd.DataFrame(columns=cols_tasks)
        #     res_df["HR"] = t_hr
        #     for task in cols_tasks:
        #         res_df[task] = [N_i_t[task, t].x for t in T]
        #     res_df["price"] = self.day_ahead_prices
        #
        #     cols = res_df.columns.to_list()
        #     cols.remove("HR")
        #
        #     cols_rev = cols[::-1]
        #
        #     fig = make_subplots(rows=len(cols), shared_xaxes=True, vertical_spacing=0.02, row_titles=cols_rev,
        #                         y_title="Values")
        #
        #     for i in range(len(cols)):
        #         fig.add_trace(go.Scatter(x=res_df.HR,
        #                                  y=res_df[cols[i]]),
        #                       row=len(cols) - i, col=1,
        #                       )
        #     fig['layout'].update(height=1000)
        #     fig.show()
        # if show_figure == True :
        #     print_opt_results()

        # %% Overall Results
        # %%Extract the results
        results_power = pd.DataFrame(index=range(len(K)),
                                     columns=["P_demand", "P_MFS", "P_CAS", "P_PCS", "P_PHS", "P_Grid", "P_Grid_sell",
                                              "P_PV", "Pdg_out", "Electricity_price"])

        results_state_variables = pd.DataFrame(index=range(len(K)),
                                               columns=["p_t", "Ts_pcs", "Ts_phs", "Tr",
                                                        "Electricity_price"])

        results_input_variables = pd.DataFrame(index=range(len(K)),
                                               columns=["Electricity_price"])

        results_es = pd.DataFrame(index=range(len(K)),
                                  columns=["E_batt", "P_batt_ch", "P_batt_disch", "CHP_elec", "CHP_q", "P_PV",
                                           "Electricity_price"])
        results_es_cap = pd.DataFrame(index=range(len(K)),
                                      columns=["E_b", "E_f", "Pb_ch", "Pb_disch", "Pf_ch", "Pf_disch"])

        results_costs = pd.DataFrame(index=range(len(K)),columns=["elec_cost_buy","elec_cost_sold","CHP_cost","prices"])
        #Append the results
        results_costs["elec_cost_buy"] = [x*y*(self.sim_time_step/60) for x,y in zip([p_grid[t].x for t in K], prices)]

        results_costs["elec_cost_sold"] = [x * c_buy * (self.sim_time_step / 60) for x in [p_grid_sold[t].x for t in K]
                                        ]
        results_costs["CHP_cost"] = [chp_cost[t].x for t in K]

        results_costs["total"] = results_costs["elec_cost_buy"] - results_costs["elec_cost_sold"] + \
                                 results_costs["CHP_cost"]

        results_costs["prices"] = prices



        # results_es_cap = pd.DataFrame(index=range(len(K)),
        # columns=["E_batt","E_b", "E_f", "Pb", "Pf"])

        # %% Append the results
        # Power
        results_power.P_demand = [demand_kw[t].x for t in K]
        # results_power.P_MFS = [p_mfs_t[t].x for t in K]
        results_power.P_MFS = [p_mfs_t[t].x for t in K]
        results_power.P_CAS = [p_cas_t[t].x for t in K]
        results_power.P_PCS = [p_pcs_t[t].x for t in K]
        results_power.P_Grid = [p_grid[t].x for t in K]
        results_power.P_PHS = [p_phs_t[t].x for t in K]
        results_power.Pdg_out = [p_dg_out[t].x for t in K]
        results_power.P_Grid_sell = [p_grid_sold[t].x for t in K]
        results_power.P_PV = pv_prod
        results_power.Electricity_price = prices

        # #Battery Capacity Price Results
        # results_es_cap.E_b = [E_batt[t].x for t in K]
        # results_es_cap.E_f = E_f
        # results_es_cap.Pb_ch = [p_batt_ch[t].x for t in K]
        # results_es_cap.Pb_disch = [p_batt_disch[t].x for t in K]
        # results_es_cap.Pf_ch = p_f_ch
        # results_es_cap.Pf_disch = p_f_disch

        # State Variables
        results_state_variables.p_t = [p_t[t].x for t in K]
        results_state_variables.Ts_pcs = [Ts_pcs[t].x for t in K]
        if phs_opt == True:
            results_state_variables.Ts_phs = [T_phs_t[t].x for t in K]
        if hvac_opt == True:
            results_state_variables.Tr = [T_r[t].x for t in K]
        results_state_variables.Electricity_price = prices

        # Input Variables
        if cas_opt == True:
            for k in compressors:
                results_input_variables.loc[results_input_variables.index[0:len(T)], k] = [Pel_k[k, t].x for t in T]
        if pcs_opt == True:
            for c in chillers:
               results_input_variables.loc[results_input_variables.index[0:len(T)], c] = [Pel_c[c, t].x for t in T]
        if phs_opt == True:
            for h in list(self.tbs.phs.heat_pumps.keys()):
                results_input_variables.loc[results_input_variables.index[0:len(T)], h] = [Pel_h[h, t].x for t in T]

        results_input_variables.Electricity_price = prices

        # Results ES
        results_es.E_batt = [E_batt[t].x for t in K]
        results_es.P_batt_ch = [p_batt_ch[t].x for t in K]
        results_es.P_batt_disch = [p_batt_disch[t].x for t in K]

        results_es.CHP_elec = [p_chp_use[t].x for t in K]
        results_es.CHP_q = [q_chp_in[t].x for t in K]
        results_es.P_PV = pv_prod
        results_es.Electricity_price = prices


        return machines_run, results_power, results_input_variables, results_state_variables, results_es,results_costs



    def  sme_opt_mpc(self,K,N_sim,L1,L2,c_buy,c_sell,market_id,participation_mechanism="Day_ahead",
                          cas_opt=True,pcs_opt=True,phs_opt=True,hvac_opt=True,chp_opt=True):


        #%% Get the market information

        flex_interval = self.markets["M-01"].pltw
        pltw = list(range(flex_interval[0], flex_interval[1])) + list(
            range(flex_interval[0] + 96, flex_interval[1] + 96))

        p_fix = self.markets[market_id].electiricity_tax + self.markets[market_id].eeg_surcharge + \
                self.markets[market_id].abschalt_surcharge + self.markets[market_id].kwk_surcharge + \
                self.markets[market_id].kwk_surcharge

        p_fix = p_fix / 100  # convert cents to euros

        tax = self.markets[market_id].tax / 100

        DR = self.markets[market_id].demand_rate
        ER = self.markets[market_id].energy_rate / 100

        if participation_mechanism == "Day_ahead":
            prices = [(x + p_fix) * (1 + tax) for x in self.day_ahead_prices]
            self.markets[market_id].prices_variable["Price"] = prices
            # prices = list(self.markets[market_id].price_variable)*n_days
        else:
            prices = (self.markets[market_id].price_buy / 100 + p_fix) * (1 + tax)
            prices = [prices] * N_sim
            self.markets[market_id].prices_fixed["Price"] = prices

        #%% Solve MFS Scheduling Problem
        p_mfs_t,air_demand_t,q_cool_demand_t,q_heat_demand_t = mfs_opt_formulation_scheduling(self,prices=prices,T=96)



        # mfs_data = pd.read_excel("mfs_data.xlsx")
        # p_mfs_t = mfs_data["p_mfs"]
        # air_demand_t = mfs_data["air_demand"]
        # q_cool_demand_t = mfs_data["q_cool_demand"]
        # q_heat_demand_t = mfs_data["q_heat_demand"]
        # print("\n\n\n MFS OPTIMIZATION DONE \n\n\n")

        #%% Initialize the state variables

        x_o = {}

        if cas_opt == True:
            x_o["v_stor_t"] = float(self.tbs.cas.storages["S_air"]["capacity"])*float(self.tbs.cas.storages["S_air"]["soc_initial"])
            x_o["p_t"] = float(self.tbs.cas.storages["S_air"]["p_set"])
        else:
            x_o["v_stor_t"] = 0
            x_o["p_t"] = 0

        if pcs_opt == True:
            x_o["Ts_pcs"] = float(self.tbs.pcs.storages["s_cws"]["temp_set"])
        else:
            x_o["Ts_pcs"] = 0

        if phs_opt == True:
            x_o["Ts_phs"] = float(self.tbs.phs.storages["S_PHS"]["temp_set"])
        else:
            x_o["Ts_phs"] = 0

        if hvac_opt == True:
            x_o["T_r"] = float(self.tbs.hvac.t_set)

        #todo : Look for delta_t and q_tes_uses
        x_o["delta_t"] = 0
        x_o["q_tes_t"] = 0
        x_o["E_batt_t"] = (self.es.ess["Batt_1"]["kwh_max"] * self.es.ess["Batt_1"]["initial_soc"])


        # x_o = {"v_stor_t": float(self.tbs.cas.storages["S_air"]["capacity"])*float(self.tbs.cas.storages["S_air"]["soc_initial"]),
        #        "p_t": float(self.tbs.cas.storages["S_air"]["p_set"]),
        #        "delta_t":0,
        #        "q_tes_t":0,
        #        "T_r": self.tbs.hvac.t_set,
        #        "Ts_pcs": self.tbs.pcs.storages["s_cws"]["temp_set"],
        #        "Ts_phs": self.tbs.phs.storages["S_PHS"]["temp_set"],
        #        "E_batt_t": (self.es.ess["Batt_1"]["kwh_max"] * self.es.ess["Batt_1"]["initial_soc"])
        #        }

        #%%Past Variables for the MPC
        u_past = {}

        if cas_opt == True:
            #CAS systems
            for k in list(self.tbs.cas.compressors.keys()):
                if self.tbs.cas.compressors[k]["control"] == "Binary":
                    u_past[k] = [0] * self.tbs.cas.compressors[k]["t_on"]


        #PCS System
        if pcs_opt == True:
            for c in list(self.tbs.pcs.chillers.keys()):
                if self.tbs.cas.compressors[k]["control"] == "Binary":
                    u_past[c] = [0]* self.tbs.pcs.chillers[c]["t_on"]

        #PHS System
        if phs_opt == True:
            for h in list(self.tbs.phs.heat_pumps.keys()):
                if self.tbs.phs.heat_pumps[h]["control"] == "Binary":
                    u_past[h] = [0]* self.tbs.phs.heat_pumps[h]["t_on"]



        def model(K,x_o,n,N_sim,results_state_varaibes,results_input_variables,results_power,results_battery,
                  p_mfs_t,air_demand_t,q_cool_demand_t,q_heat_demand_t,u_past=u_past):
            mdl = Model("MFS;TBS;PCS")
            mdl.__len__ = 1


            T  = list(range(K))


            mode = "MPC"

            #Applying the MPC loop
            n = n
            while n<=(N_sim-K):

                if n==27:
                    A = "STOP"

                #%%Call the CAS System Optimization
                if cas_opt == True:
                    p_cas_t, Pel_k, v_out, v_in, v_stor_t, p_t, N_k_t,compressors,z_on,z_off = cas_opt_formulation_mpc(self=self, mdl=mdl,
                                                K=T,mode=mode, x_o=x_o,from_agent=False,air_demand_opt=air_demand_t,n=n,
                                                u_past=u_past)

                #%%Call the HVAC Optimization
                if hvac_opt == True:
                    T_r, T_amb, Qc_hvac, Qh_hvac = hvac_opt_formulation_mpc(self=self, mode=mode, mdl=mdl, x_o=x_o, K=T,n=n)
                elif hvac_opt == False:
                    Qc_hvac = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="Qc_hvac")
                    T_r = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="T_r")
                    mdl.addConstrs(Qc_hvac[t] == 0 for t in T)
                    mdl.addConstrs(T_r[t] == 0 for t in T)

                #%%Call the PCS System Optimization
                if pcs_opt == True:
                    p_pcs_t, Pel_c, T_pcs_t, q_tes_t, qc_in_t, qc_out_t, chillers, N_c_t, delta_t = \
                        pcs_opt_formulation_mpc(self=self, mdl=mdl, K=T, mode=mode, x_o=x_o,
                                            cooling_demand_t=q_cool_demand_t, Qc_hvac=Qc_hvac,n=n,u_past=u_past)
                elif pcs_opt == False:
                    p_pcs_t = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="p_pcs_t")
                    mdl.addConstrs(p_pcs_t[t] == 0 for t in T)
                #
                # %%CHP Formulation as it is needed by the Heating system
                if chp_opt == True:
                    N_chp,q_chp_in, p_chp_use, p_chp_sold, chp_cost, chp_on, chp_off = chp_formulation_mpc(self=self, mdl=mdl, K=T,
                                                                                                 x_o=x_o,
                                                                                                 mode=mode,
                                                                                                 chp_on_past=[0, 0, 0])
                elif chp_opt == False:
                    q_chp_in = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="q_chp_in")
                    p_chp_use =  mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="p_chp_use")
                    p_chp_sold = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="p_chp_sold")
                    chp_cost = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="chp_cost")
                    mdl.addConstrs(q_chp_in[t]==0 for t in T)
                    mdl.addConstrs(p_chp_use[t] == 0 for t in T)
                    mdl.addConstrs(p_chp_sold[t] == 0 for t in T)
                    mdl.addConstrs(chp_cost[t] == 0 for t in T)

                # %%Call the PHS System Optimization
                if phs_opt == True:
                    N_h_t,p_phs_t, Pel_h, H_test_t, T_phs_t, Qh_in_t, Qmfs_out_t, Qchp_in_t,heaters = \
                        phs_opt_formulation_energy_based_mpc(self=self, mdl=mdl, K=T, x_o=x_o, from_agent=False,
                                                         heating_demand_t=q_heat_demand_t,
                                                         q_chp_in=q_chp_in, Qh_hvac=Qh_hvac, mode=mode,n=n,u_past=u_past)
                elif phs_opt == False:
                    p_phs_t = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="p_phs_t")
                    T_phs_t = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="T_phs_t")
                    H_test_t = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="H_test_t")
                    Pel_h = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="Pel_h")
                    mdl.addConstrs(p_phs_t[t] == 0 for t in T)
                    mdl.addConstrs(T_phs_t[t] == 0 for t in T)
                    mdl.addConstrs(H_test_t[t] == 0 for t in T)
                    mdl.addConstrs(Pel_h[t] == 0 for t in T)

                # %% Call the ES System Optimization
                # Battery Energy Systems
                p_batt_ch, p_batt_disch, p_batt_use, p_batt_sold, E_batt, y_batt = es_bess_opt_formulation(self=self, mdl=mdl,
                                                                                                   K=T,mode=mode, x_o=x_o)

                # PV Formulation
                pv_prod, p_pv_use, p_pv_sold = es_pv_opt_formulation_mpc(self=self, mdl=mdl, K=T, mode=mode, n=1)

                # %% Grid Variables
                p_grid = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, lb=0, ub=L1, name="p_grid")
                p_grid_sold = mdl.addVars(T, vtype=GRB.CONTINUOUS, lb=0, ub=L2, name="p_sold")
                y_grid = mdl.addVars(len(T), vtype=GRB.BINARY, name="y_grid")

                # %% SME Load Demand
                demand_kw = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="demand_kw")

                # Energy/Power Management
                #mdl.addConstrs(demand_kw[t] == p_cas_t[t] + p_mfs_t[t] + p_phs_t[t] + p_pcs_t[t] for t in K)

                # Energy/Power Management
                mdl.addConstrs(demand_kw[t] == p_cas_t[t] + p_pcs_t[t] + p_phs_t[t] + p_mfs_t[t+n] for t in T)

                # Power Balance
                # mdl.addConstrs(
                #     p_grid[t] + p_batt_use[t] + p_pv_use[t] + p_chp_use[t] == demand_kw[t] + p_batt_ch[t] for t in K)

                mdl.addConstrs(
                    p_grid[t] + p_batt_use[t] + p_pv_use[t] + p_chp_use[t]  == demand_kw[t] + p_batt_ch[t] for t in T)

                # Total power injected into the Grid
                #mdl.addConstrs(p_grid_sold[t] == p_batt_sold[t] + p_pv_sold[t] + p_chp_sold[t] for t in K)
                mdl.addConstrs(p_grid_sold[t] == p_batt_sold[t] + p_pv_sold[t] + p_chp_sold[t]  for t in T)
                # Power transaction constraints
                mdl.addConstrs(p_grid[t] <= L1 * (y_grid[t]) for t in T)
                mdl.addConstrs(p_grid_sold[t] <= L2 * (1 - y_grid[t]) for t in T)

                # %%# %% Objective function
                if participation_mechanism == "Fixed":
                    mdl.setObjective(quicksum(p_grid[t] * prices[t+n] * (self.sim_time_step / 60) for t in T))

                elif participation_mechanism == "Day_ahead":
                    # mdl.setObjective(quicksum((p_grid[t] * (prices[t+n]) * (self.sim_time_step / 60)) -
                    #                           (p_grid_sold[t+n] * c_sell * (self.sim_time_step / 60)) for t in K) +
                    #                           quicksum(chp_cost[t] for t in K))

                    mdl.setObjective(quicksum((p_grid[t] * (prices[t + n]) * (self.sim_time_step / 60)) -
                                              (p_grid_sold[t] * c_sell * (self.sim_time_step / 60)) for t in T)+
                                               quicksum(chp_cost[t] for t in T))

                    # mdl.setObjective(quicksum(Qh_hvac[t]*(self.sim_time_step)/60 for t in K))


                # %% Solve the model
                # Senseofthemode
                mdl.modelSense = GRB.MINIMIZE
                mdl.setParam("MIPGap",1e-4)
                # mdl.setParam("MIPGapAbs",5)
                # mdl.setParam("Method",1)
                # mdl.tune()
                # mdl.setParam("TimeLimit",20)
                mdl.optimize()

                #%% Initialization parameters for the next MPC Loop

                # #MFS Initialization for the next MPC Loop
                # for r in (list(self.mfs.raw_materials.keys())+list(self.mfs.products.key())):
                #     self.mfs.Ro[r] = [R_r_t[r,t].x for t in K][1]
                #
                # for s in (list(self.mfs.storage_facilities.keys())):
                #     self.mfs.Ro[s] = [R_s_t[s,t].x for t in K][1]

                #TBS and ES Initialization variables
                # x_o = {"v_stor_t" : [v_stor_t[t].x for t in K][1],
                #        "p_t" : [p_t[t].x for t in K ][1],
                #        "T_r" : [T_r[t].x for t in K][1],
                #        "Ts_pcs":[T_pcs_t[t].x for t in K][1],
                #        "Ts_phs":[T_phs_t[t].x for t in K][1],
                #         "H_tes": [H_test_t[t].x for t in K][1],
                #        "E_batt_t":[E_batt[t].x for t in K][1]
                #        }

                x_o = {"v_stor_t": [v_stor_t[t].x for t in T][1],
                       "p_t": [p_t[t].x for t in T][1],
                       "E_batt_t": [E_batt[t].x for t in T][1],
                       "Ts_pcs": [T_pcs_t[t].x for t in T][1],
                       "delta_t": [delta_t[t].x for t in T][1],
                       "q_tes_t": [q_tes_t[t].x for t in T][1],
                       "Ts_phs": [T_phs_t[t].x for t in T][1],
                       "T_r" : [T_r[t].x for t in T][1]
                       }

                #Input values
                u_o = {}

                if n==0 :
                    A = 12

                #Append the results
                results_state_varaibes.loc[results_state_varaibes.index[n:n+K],"v_stor_t"] = [v_stor_t[t].x for t in T]
                results_state_varaibes.loc[results_state_varaibes.index[n:n+K],"p_t"] = [p_t[t].x for t in T]
                results_state_varaibes.loc[results_state_varaibes.index[n:n+K],"T_r"] = [T_r[t].x for t in T]
                results_state_varaibes.loc[results_state_varaibes.index[n:n+K], "Ts_pcs"] = [T_pcs_t[t].x for t in T]
                results_state_varaibes.loc[results_state_varaibes.index[n:n+K], "Ts_phs"] = [T_phs_t[t].x for t in T]
                results_state_varaibes.loc[results_state_varaibes.index[n:n+K], "H_tes"] = [H_test_t[t].x for t in T]
                results_state_varaibes.loc[results_state_varaibes.index[n:n+K], "E_batt_t"] = [E_batt[t].x for t in T]
                results_state_varaibes.loc[results_state_varaibes.index[n:n+K], "Prices"]   = prices[n:n+K]

                #Input Variables
                if cas_opt == True:
                    for k in compressors:
                        results_input_variables.loc[results_input_variables.index[n:n + K], k] = [Pel_k[k,t].x for t in T]

                if pcs_opt == True:
                    for c in chillers:
                        results_input_variables.loc[results_input_variables.index[n:n + K], c] = [Pel_c[c, t].x for t in T]

                if phs_opt == True:
                    for h in heaters:
                        results_input_variables.loc[results_input_variables.index[n:n + K], h] = [Pel_h[h, t].x for t in T]



                results_input_variables.loc[results_input_variables.index[n:n + K], "Prices"] = prices[n:n + K]


                #Power Variables
                results_power.loc[results_power.index[n:n + K], "P_demand"] = [demand_kw[t].x for t in T]
                results_power.loc[results_power.index[n:n + K], "P_MFS"] = p_mfs_t[n:n+K]
                results_power.loc[results_power.index[n:n + K], "P_CAS"] = [p_cas_t[t].x for t in T]
                results_power.loc[results_power.index[n:n + K], "P_PCS"] = [p_pcs_t[t].x for t in T]
                results_power.loc[results_power.index[n:n + K], "P_PHS"] = [p_phs_t[t].x for t in T]
                results_power.loc[results_power.index[n:n + K], "P_Grid"] = [p_grid[t].x for t in T]
                results_power.loc[results_power.index[n:n + K], "P_Grid_sell"] = [p_grid_sold[t].x for t in T]
                results_power.loc[results_power.index[n:n + K], "P_PV"] = pv_prod[n:n+K]
                results_power.loc[results_power.index[n:n + K], "P_CHP"] = [p_chp_use[t].x for t in T]
                results_power.loc[results_power.index[n:n + K], "Prices"] = prices[n:n + K]



                # results_power.loc[n].P_demand = [demand_kw[t].x for t in T][0]
                # results_power.loc[n].P_MFS = p_mfs_t[n]
                # results_power.loc[n].P_CAS = [p_cas_t[t].x for t in T][0]
                # results_power.loc[n].P_PCS = [p_pcs_t[t].x for t in T][0]
                # results_power.loc[n].P_PHS = [p_phs_t[t].x for t in T][0]
                # results_power.loc[n].P_Grid = [p_grid[t].x for t in T][0]
                # results_power.loc[n].P_Grid_sell = [p_grid_sold[t].x for t in T][0]
                # results_power.loc[n].P_PV = pv_prod[n],
                # results_power.loc[n].P_CHP = [p_chp_use[t].x for t in T]
                # results_power.loc[n].Prices = prices[n]

                #Battery
                results_battery.loc[results_battery.index[n:n + K], "E_batt_kwh"] = [E_batt[t].x for t in T]
                results_battery.loc[results_battery.index[n:n + K], "p_batt_ch_kw"] = [p_batt_ch[t].x for t in T]
                results_battery.loc[results_battery.index[n:n + K], "P_batt_disch_kw"] = [p_batt_disch[t].x for t in T]
                results_battery.loc[results_battery.index[n:n + K], "p_pv_kw"] = pv_prod[n:n+K]
                results_battery.loc[results_battery.index[n:n + K], "Electricity_price"] = prices[n:n + K]


                # results_battery.loc[n].E_batt_kwh = [E_batt[t].x for t in T][0]
                # results_battery.loc[n].p_batt_ch_kw = [p_batt_ch[t].x for t in T][0]
                # results_battery.loc[n].P_batt_disch_kw = [p_batt_disch[t].x for t in T][0]
                # results_battery.loc[n].p_pv_kw = pv_prod[n]
                # results_battery.loc[n].Electricity_price = prices[n]


                print("\n\n **** LOOP " + str(abs(n)) + "  Completed ****\n")

                A = 1


                #%%Append the results directly into dataframe attributes of each components

                #CAS
                if cas_opt == True:
                    cas_opt_results(self=self,K=T,p_cas_t=p_cas_t,N_k_t=N_k_t,Pel_k=Pel_k,v_stor_t=v_stor_t,v_in=v_in,
                                    v_out=v_out,p_t=p_t,n=n)

                #PCS
                if pcs_opt == True:
                    pcs_opt_results(self=self,K=T,p_pcs_t=p_pcs_t,N_c_t=N_c_t,Pel_c=Pel_c,t_pcs_t=T_pcs_t,
                                    qc_out_t=qc_out_t,qc_in_t=qc_in_t,Qc_hvac=Qc_hvac,n=n)

                # PHS
                if phs_opt == True:
                    phs_opt_results(self, K=T, p_phs_t=p_phs_t, Pel_h=Pel_h, N_h_t=N_h_t, H_test_t=H_test_t,
                                    T_phs_t=T_phs_t, Qh_in_t=Qh_in_t, Qchp_in_t=Qchp_in_t, Qmfs_out_t=Qmfs_out_t,
                                    Qh_hvac=Qh_hvac,n=n)
                #BESS
                es_bess_opt_results(self=self,K=T,E_batt=E_batt,p_batt_ch=p_batt_ch,p_batt_disch=p_batt_disch,
                                    p_batt_sold=p_batt_sold,p_batt_use=p_batt_use,n=n)

                # PV
                es_pv_results(self=self, K=T, p_pv_use=p_pv_use, p_pv_sold=p_pv_sold,pv_prod=pv_prod,n=n)

                # CHP
                if chp_opt == True:
                    es_chp_opt_results(self=self, K=T, p_chp_use=p_chp_use, p_chp_sold=p_chp_sold,q_chp_in=q_chp_in,
                                       chp_cost=chp_cost,n=n)

                # HVAC
                if hvac_opt == True:
                    hvac_opt_results(self=self, K=T, Qh_hvac=Qh_hvac, Qc_hvac=Qc_hvac, T_amb=T_amb, T_r=T_r,n=n)



                #Get the past history values

                #CAS System
                if cas_opt == True:
                    for k in compressors:
                        if self.tbs.cas.compressors[k]["control"] == "Binary":
                            mr = self.tbs.cas.compressors[k]["t_on"]
                            u_past[k] = [N_k_t[k,t].x for t in list(range(-mr+1,1))]# 1:MR+1

                #PCS System
                if pcs_opt == True:
                    for c in chillers:
                        if self.tbs.pcs.chillers[c]["control"] == "Binary":
                            mr = self.tbs.pcs.chillers[c]["t_on"]
                            u_past[c] = [N_c_t[c,t].x for t in list(range(-mr+1,1))]
                 #PHS System
                if phs_opt == True:
                    for h in heaters:
                        if self.tbs.phs.heat_pumps[h]["control"] == "Binary":
                            mr = self.tbs.phs.heat_pumps[h]["t_on"]
                            u_past[h] = [N_h_t[h,t].x for t in list(range(-mr+1,1))]

                return model(K=K,x_o=x_o,n=n+1,N_sim=N_sim,results_state_varaibes=results_state_varaibes,results_power=results_power,results_input_variables=results_input_variables,
                             results_battery=results_battery,p_mfs_t=p_mfs_t,air_demand_t=air_demand_t,
                             q_heat_demand_t=q_heat_demand_t,q_cool_demand_t=q_cool_demand_t,u_past=u_past)


        #%%Optimization Results Variables


        results_state_varaibes = pd.DataFrame(index=range(N_sim),
                                              columns=["v_stor_t", "p_t", "T_r","Ts_pcs",
                                                       "Ts_phs", "H_tes","E_batt_t","Prices"])

        results_input_variables = pd.DataFrame(index=range(N_sim),columns=["Prices"])

        results_power = pd.DataFrame(index=range(N_sim),
                                    columns=["P_demand", "P_MFS", "P_CAS", "P_PCS", "P_PHS", "P_Grid", "P_Grid_sell",
                                             "P_PV", "Prices"])

        results_battery = pd.DataFrame(index=range(N_sim),
                                       columns=["E_batt_kwh", "p_batt_ch_kw", "P_batt_disch_kw", "p_pv_kw",
                                                "Electricity_price"])

        #Start the MPC Loops here
        model(K=K,x_o=x_o,n=0,N_sim=N_sim,results_state_varaibes=results_state_varaibes,results_input_variables=results_input_variables,results_power=results_power,results_battery=results_battery,
              p_mfs_t=p_mfs_t,air_demand_t=air_demand_t,
              q_heat_demand_t=q_heat_demand_t,q_cool_demand_t=q_cool_demand_t)



        return  results_state_varaibes,results_input_variables,results_power,results_battery



























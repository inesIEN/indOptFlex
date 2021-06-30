from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd


def es_bess_opt_formulation(self, mdl, K,x_o, mode):
    # Get the battery parameters

    # Battery Parameters
    p_ch_max = self.es.ess["Batt_1"]["discharge_rate_max"]
    e_batt_min = self.es.ess["Batt_1"]["kwh_max"] * self.es.ess["Batt_1"]["soc_min"]
    e_batt_max = self.es.ess["Batt_1"]["kwh_max"] * self.es.ess["Batt_1"]["soc_max"]

    # Add the battery varaibles
    E_batt = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="E_batt")
    p_batt_ch = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=0, ub=p_ch_max, name="p_batt_ch")
    p_batt_disch = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=0, ub=p_ch_max, name="p_batt_disch")
    p_batt_use = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_batt_use")
    p_batt_sold = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_batt_sold")
    # Binary variable for controlling switching between charging and discharging
    y_batt = mdl.addVars(len(K), vtype=GRB.BINARY, name="Y_ess_t")

    # Add the Battery Constraints
    # SOC of the battery
    if mode == "MILP":
        mdl.addConstrs(E_batt[t] == (self.es.ess["Batt_1"]["kwh_max"] * self.es.ess["Batt_1"]["initial_soc"])
                       for t in K if t == 0)
    elif mode == "MPC":
        mdl.addConstrs(E_batt[t] == x_o['E_batt_t'] for t in K if t == 0)

    mdl.addConstrs(
        E_batt[t + 1] == E_batt[t] + (p_batt_ch[t] * self.es.ess["Batt_1"]["charge_efficiency"] -
                                      p_batt_disch[t] / self.es.ess["Batt_1"]["discharge_efficiency"])
        * (self.sim_time_step / 60) for t in K if t < len(K) - 1)

    mdl.addConstrs(E_batt[t] >= e_batt_min for t in K)
    mdl.addConstrs(E_batt[t] <= e_batt_max for t in K)

    mdl.addConstrs(p_batt_ch[t] <= (self.es.ess["Batt_1"]["charging_rate_max"] * y_batt[t]) for t in K)

    mdl.addConstrs(p_batt_disch[t] <= (self.es.ess["Batt_1"]["discharge_rate_max"] * (1 - y_batt[t])) for t in K)

    # # # Battery Ramping Constraints
    mdl.addConstrs(p_batt_ch[t - 1] - p_batt_ch[t] <= 10 for t in K if t > 0)
    mdl.addConstrs(p_batt_ch[t - 1] - p_batt_ch[t] >= -10 for t in K if t > 0)

    mdl.addConstrs(p_batt_disch[t - 1] - p_batt_disch[t] <= 10 for t in K if t > 0)
    mdl.addConstrs(p_batt_disch[t - 1] - p_batt_disch[t] >= -10 for t in K if t > 0)

    mdl.addConstrs(p_batt_use[t] + p_batt_sold[t] == p_batt_disch[t] for t in K)

    #mdl.addConstrs(p_batt_ch[t] == 10 for t in list(range(0,4)))
    # mdl.addConstrs(p_batt_disch[t] == 10 for t in list(range(40, 44)))

    return p_batt_ch, p_batt_disch, p_batt_use, p_batt_sold, E_batt, y_batt

def es_bess_opt_formulation_cap(self, mdl, K,x_o, mode):

    #Get the battery parameters

    p_ch_max = self.es.ess["Batt_1"]["charging_rate_max"]
    p_disch_max = self.es.ess["Batt_1"]["discharge_rate_max"]
    e_batt_min = self.es.ess["Batt_1"]["kwh_max"] * self.es.ess["Batt_1"]["soc_min"]
    e_batt_max = self.es.ess["Batt_1"]["kwh_max"] * self.es.ess["Batt_1"]["soc_max"]

    #Add the Variables
    E_batt = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="E_batt",lb=e_batt_min,ub=e_batt_max)
    p_batt_ch = mdl.addVars(len(K),vtype=GRB.CONTINUOUS, name="P_ch",lb=0,ub=p_ch_max)
    p_batt_disch = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="P_disch", lb=0, ub=p_disch_max)


    E_b = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="E_b",lb=e_batt_min,ub=e_batt_max)
    p_b_ch = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=0, ub=p_ch_max, name="p_b_ch")
    p_b_disch = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=0, ub=p_disch_max, name="p_b_disch")

    E_f = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=e_batt_min, ub=e_batt_max, name="E_f")
    p_f_ch = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=0, ub=p_ch_max, name="p_b_ch")
    p_f_disch = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=0, ub=p_disch_max, name="p_b_disch")

    #Binary variable
    y_batt =  mdl.addVars(len(K), vtype=GRB.BINARY, name="y_b")
    y_b = mdl.addVars(len(K), vtype=GRB.BINARY, name="y_b")
    y_f = mdl.addVars(len(K), vtype=GRB.BINARY, name="y_f")

    #Battery Initial Condition Equaton
    mdl.addConstrs(E_batt[t] == (self.es.ess["Batt_1"]["kwh_max"] * self.es.ess["Batt_1"]["initial_soc"])
                                                                                     for t in K if t == 0)
    mdl.addConstrs(E_b[t] == 0.6 * E_batt[t] for t in K if t==0)


    #Battery Energy Modelling Equation
    mdl.addConstrs(E_f[t] == E_batt[t] - E_b[t] for t in K)
    mdl.addConstrs(p_batt_ch[t] == p_b_ch[t] + p_f_ch[t] for t in K)
    mdl.addConstrs(p_batt_disch[t] == p_b_disch[t] + p_f_disch[t] for t in K)

    mdl.addConstrs(E_b[t+1] == E_b[t] + (p_b_ch[t] - p_b_disch[t])*15/60 for t in K if t < len(K) - 1)
    mdl.addConstrs(p_b_ch[t] <= p_ch_max*y_b[t] for t in K)
    mdl.addConstrs(p_b_disch[t] <= p_ch_max * (1-y_b[t]) for t in K)


    mdl.addConstrs(E_f[t + 1] == E_f[t] + (p_f_ch[t] - p_f_disch[t]) * 15 / 60 for t in K if t < len(K) - 1)
    mdl.addConstrs(p_f_ch[t] <= (p_ch_max ) * y_f[t] for t in K)
    mdl.addConstrs(p_f_disch[t] <= (p_disch_max) * (1-y_f[t]) for t in K)

    #Total Battery SOC Contraints
    mdl.addConstrs(E_batt[t] >= e_batt_min for t in K)
    mdl.addConstrs(E_batt[t] <= e_batt_max for t in K)

    # Battery Ramping Constraints
    mdl.addConstrs(p_batt_ch[t - 1] - p_batt_ch[t] <= 10 for t in K if t > 0)
    mdl.addConstrs(p_batt_ch[t - 1] - p_batt_ch[t] >= -10 for t in K if t > 0)

    mdl.addConstrs(p_batt_disch[t - 1] - p_batt_disch[t] <= 10 for t in K if t > 0)
    mdl.addConstrs(p_batt_disch[t - 1] - p_batt_disch[t] >= -10 for t in K if t > 0)

    #mdl.addConstrs(y_f[t] == y_b[t]  for t in K)

    mdl.addConstrs(p_f_ch[t+1] == p_f_ch[t] for t in K if t<len(K)-1)

    # # Get the battery parameters
    #
    # # Battery Parameters
    # p_b_max = self.es.ess["Batt_1"]["discharge_rate_max"]
    # p_b_min = -self.es.ess["Batt_1"]["discharge_rate_max"]
    # e_batt_min = self.es.ess["Batt_1"]["kwh_max"] * self.es.ess["Batt_1"]["soc_min"]
    # e_batt_max = self.es.ess["Batt_1"]["kwh_max"] * self.es.ess["Batt_1"]["soc_max"]
    # charging_efficiency = self.es.ess["Batt_1"]["charge_efficiency"]
    #
    # # Add the battery varaibles
    # E_batt = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="E_batt",lb=e_batt_min,ub=e_batt_max)
    # E_b = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="E_b",lb=e_batt_min,ub=e_batt_max)
    # E_f = mdl.addVars(len(K), vtype=GRB.CONTINUOUS,lb=e_batt_min,ub=e_batt_max, name="E_f")
    # p_b = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=p_b_min, ub=p_b_max, name="p_b")
    # p_f = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=p_b_min, ub=p_b_max, name="p_f")
    #
    # #Binary Indicator Variable
    # Y = mdl.addVars(len(K), vtype=GRB.BINARY,name="Y")
    #
    # # Add the Battery Constraints
    # # SOC of the battery
    # if mode == "MILP":
    #     mdl.addConstrs(E_batt[t] == (self.es.ess["Batt_1"]["kwh_max"] * self.es.ess["Batt_1"]["initial_soc"])
    #                    for t in K if t == 0)
    #     mdl.addConstrs(E_b[t] == 0.5 * E_batt[t] for t in K if t==0)
    #
    #
    #
    #
    # elif mode == "MPC":
    #     mdl.addConstrs(E_batt[t] == x_o['E_batt_t'] for t in K if t == 0)
    #     mdl.addConstrs(E_f[t] == 0 for t in K if t == 0)
    #
    # #Battery Balance
    #
    # mdl.addConstrs(E_f[t] == E_batt[t] - E_b[t] for t in K)
    #
    # mdl.addConstrs(E_b[t + 1] == E_batt[t] + (p_b[t] * charging_efficiency * (self.sim_time_step / 60))
    #                for t in K if t < len(K) - 1)
    #
    # mdl.addConstrs(E_f[t + 1] == E_f[t] + (p_f[t] * charging_efficiency * (self.sim_time_step / 60))
    #                for t in K if t < len(K) - 1)
    #
    # # Total Battery SOC Contraints
    # mdl.addConstrs(E_batt[t] >= e_batt_min for t in K)
    # mdl.addConstrs(E_batt[t] <= e_batt_max for t in K)
    #
    #
    # #Indicator variable Y based on the sign of Pb
    # M = 1000
    # mdl.addConstrs(M*Y[t] >= p_b[t] - 1 for t in K)
    # mdl.addConstrs(M * (1-Y[t]) >= 1 - p_b[t]  for t in K)
    #
    #
    # #Constraints of Pf
    # mdl.addConstrs(p_f[t] >= p_b[t] - p_b_max for t in K)
    # mdl.addConstrs(p_f[t] <= p_b[t] - p_b_min for t in K)
    #
    #
    #
    #
    #
    #
    #
    #
    # # Battery Ramping Constraints
    #
    #
    # #%%Calculate the remaining charge and discharge for the capacity market

    return p_b_ch,p_b_disch,p_f_ch,p_f_disch,E_b,E_f,E_batt,p_batt_ch,p_batt_disch

def es_bess_opt_formulation_capp(self, K, E_b, p_batt_ch, p_batt_disch,y_batt):

    mdl = Model('flex_start')
    mdl.__len__ = 1

    #Get the battery parameters
    e_batt_max = self.es.ess["Batt_1"]["kwh_max"]
    e_batt_min = 0

    #Scheduleed Battery Variables
    E_b = [E_b[t].x for t in K]

    p_batt_ch = [p_batt_ch[t].x for t in K]
    p_batt_disch = [p_batt_disch[t].x for t in K]
    p_b = [sum(x) for x in zip(p_batt_ch,p_batt_disch)]

    #Get the binary variablr
    y_batt = [y_batt[t].x for t in K]

    E_f_max = [e_batt_max - x for x in E_b]

    #Add the Variables
    E_f = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="E_batt",lb=e_batt_min,ub=e_batt_max)
    p_f_ch = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="P_f",lb=0)
    p_f_disch = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="P_f",lb=0)
    y_f = mdl.addVars(len(K), vtype=GRB.BINARY, name="P_f")


    #Equations :
    mdl.addConstrs(E_f[t] == e_batt_max - E_b[t] for t in K if t==0)
    mdl.addConstrs(E_f[t+1] == E_f[t] + (p_f_ch[t] - p_f_disch[t])*15/60  for t in K if t < len(K) - 1)

    #Battery Constraints for Ef
    mdl.addConstrs(E_f[t] <= E_f_max[t] for t in K)
    mdl.addConstrs(E_f[t] >= 0 for t in K)

    #Constraints Limits on P_f_ch and P_f_disc
    mdl.addConstrs(p_f_ch[t] <=  (75 - p_batt_ch[t])*y_f[t] for t in K)
    mdl.addConstrs(p_f_disch[t] <= (75 - p_batt_disch[t])*(1-y_f[t]) for t in K)

    #mdl.addConstrs(y_f[t] == y_batt[t] for t in K if t>0 )

    mdl.setObjective(quicksum(p_f_ch[t] for t in K))
    mdl.modelSense = GRB.MAXIMIZE
    print("\n\n Solving Capacity Model\n\n")
    mdl.optimize()

    A =2

    p_f_ch = [p_f_ch[t].x for t in K]
    p_f_disch = [p_f_disch[t].x for t in K]
    E_f = [E_f[t].x for t in K]




    return p_f_ch, p_f_disch, E_f

def es_dg_opt_formulation(self, mdl, K):

    # Collecting parameters
    n_min = self.es.dg["DG-1"]["n_min"] #Min PF
    n_max = self.es.dg["DG-1"]["n_max"] #Max PF
    # e_max = self.es.dg["DG-1"]["e_max"]
    p_rated = self.es.dg["DG-1"]["p_rated"] #KVA
    cons_a = self.es.dg["DG-1"]["cons_a"]   #
    cons_b = self.es.dg["DG-1"]["cons_b"]
    e_coef = self.es.dg["DG-1"]["e_coef"]
    price_fuel = self.es.dg["DG-1"]["price_fuel"]

    # Add Diesel Generator variables
    p_dg_out = mdl.addVars(len(K),vtype=GRB.CONTINUOUS,lb=0,ub=n_max*float(p_rated),name="p_dg_out")
    fuel_cons_dg = mdl.addVars(len(K),vtype=GRB.CONTINUOUS,name="fuel_cons_dg")
    e_dg_rate = mdl.addVars(len(K),vtype=GRB.CONTINUOUS,name="e_dg_rate")
    n_dg_gen = mdl.addVars(len(K),vtype=GRB.CONTINUOUS,lb=n_min,ub=n_max,name="n_dg_gen")
    N_dg = mdl.addVars(len(K),vtype=GRB.BINARY, name="N_dg")
    p_dg_use = mdl.addVars(len(K),vtype=GRB.CONTINUOUS,name="p_dg_use")
    p_dg_sold = mdl.addVars(len(K),vtype=GRB.CONTINUOUS, name="p_dg_sold")
    p_dg_cost = mdl.addVars(len(K),vtype=GRB.CONTINUOUS, name="p_dg_cost")

    # Add Constraints
    mdl.addConstrs(p_dg_out[t] == n_dg_gen[t]*p_rated*N_dg[t] for t in K)
    mdl.addConstrs(fuel_cons_dg[t] == (cons_a*p_dg_out[t]+cons_b*p_rated)*N_dg[t] for t in K)
    mdl.addConstrs(e_dg_rate[t] == e_coef*fuel_cons_dg[t] for t in K)
    mdl.addConstrs(p_dg_cost[t] == price_fuel*fuel_cons_dg[t] for t in K)
    mdl.addConstrs(p_dg_out[t] == p_dg_use[t]+p_dg_sold[t] for t in K)

    mdl.addConstr(quicksum(N_dg[t] for t in K) <= 16 )


    return p_dg_out, fuel_cons_dg, e_dg_rate,n_dg_gen, p_dg_sold, p_dg_use, p_dg_cost, N_dg

def es_pv_opt_formulation(self, mdl, K,mode):
    # Get the total PV production
    # total PV production
    pv_prod = np.zeros(len(self.hr))
    pv_plants = list(self.es.pv_plants.keys())
    for pv in pv_plants:
        pv_prod = pv_prod + np.array(self.es.pv_plants[pv]["p_kw"])
        # pv_prod = [x * 15 / 60 for x in pv_prod]

    # %% PV Plant Variables
    p_pv_use = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_pv_use")
    p_pv_sold = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_pv_sold")

    # %% DER constraints
    if mode == "MILP":
        mdl.addConstrs(p_pv_use[t] + p_pv_sold[t] == pv_prod[t] for t in K)
    elif mode == "MPC":
        mdl.addConstrs(p_pv_use[t] + p_pv_sold[t] == pv_prod[t] for t in K)

    return pv_prod, p_pv_use, p_pv_sold

def chp_formulation(self,mdl,K,x_o,mode,chp_on_past=[0,0,1]):
    # Get the storage tank of CHP parameters
    Estor_chp_max = float(self.es.chp["CHP_1"]["storage_capacity"])
    e_chp_min = Estor_chp_max * float(self.es.chp["CHP_1"]["storage_capacity_min"])
    e_chp_max = Estor_chp_max * float(self.es.chp["CHP_1"]["storage_capacity_max"])
    e_chp_ini = Estor_chp_max * float(self.es.chp["CHP_1"]["soc_ini"])

    t_on = int(self.es.chp["CHP_1"]["t_on"]/self.sim_time_step)
    t_off = int(self.es.chp["CHP_1"]["t_off"]/self.sim_time_step)

    # CHP Varibles
    chp_mr = self.es.chp["CHP_1"]["t_on"]
    p_chp = mdl.addVars(len(K),  vtype=GRB.CONTINUOUS, name='p_chp_', lb=0)
    p_chp_use = mdl.addVars(len(K),  vtype=GRB.CONTINUOUS, name='p_chp_use')
    p_chp_sold = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name='p_chp_sold')
    q_chp_in = mdl.addVars(len(K),  vtype=GRB.CONTINUOUS, name='q_chp', lb=0)
    N_chp = mdl.addVars(len(K),  vtype=GRB.BINARY, name="N_chp")
    chp_on = mdl.addVars(len(K), vtype=GRB.BINARY, name="chp_on")
    chp_off = mdl.addVars(len(K),  vtype=GRB.BINARY, name="chp_off")

    chp_flank = mdl.addVars(len(K),  vtype=GRB.BINARY, name="chp_flank")
    chp_cost = mdl.addVars(len(K),  vtype=GRB.CONTINUOUS, name="chp_cost")

    # Add the past flank variables
    #Past variables
   # mdl.addConstrs(chp_on[t] == chp_on_past[t] for t in list(range(len(chp_on_past))))

    mdl.addConstrs(q_chp_in[t] == self.es.chp["CHP_1"]["q_nom"] * (N_chp[t] * (self.es.chp["CHP_1"]["n_heat"]))
                   for t in K)
    mdl.addConstrs(p_chp[t] == self.es.chp["CHP_1"]["p_nom"] * (N_chp[t] * (self.es.chp["CHP_1"]["n_elec"]))
                   for t in K)

    mdl.addConstrs(p_chp_use[t] + p_chp_sold[t] == p_chp[t] for t in K)

    mdl.addConstrs(chp_cost[t] == self.es.chp["CHP_1"]["fuel_cost"] * self.es.chp["CHP_1"]["f_nom"] * N_chp[t]
                   * (self.sim_time_step / 60 )for t in K)

    mdl.addConstr(quicksum(N_chp[t] for t in K) <=24)
    #mdl.addConstrs(N_chp[t] ==0 for t in K if t==0)

    #MR, MO and No. of switching of the CHP

    mdl.addConstrs(N_chp[t]-N_chp[t-1] == chp_on[t] -chp_off[t] for t in K if t>0)
    mdl.addConstrs(chp_on[t]+chp_off[t] <=1 for t in K)

    mdl.addConstrs(quicksum(chp_on[t-j] for j in list(range(t_on)) if t-j>0)<=1-chp_off[t] for t in K)
    mdl.addConstrs(quicksum(chp_off[t - j] for j in list(range(t_off)) if t - j > 0) <= 1 - chp_on[t] for t in K)






    # #Switching
    # mdl.addConstr(quicksum(chp_on[t]+chp_off[t] for t in K) <= 10)
    #
    # #MR
    # for t in K:
    #     if t-3+1 >0:
    #         MR = list(range(t-3+1,t))
    #         mdl.addConstr(quicksum(chp_on[j] for j in MR ) <=N_chp[t])
    #     if t-6+1 >0:
    #         MO = list(range(t-6+1,t))
    #         mdl.addConstr(quicksum(chp_off[j] for j in MO ) <=N_chp[t])

    #mdl.addConstrs(N_chp[t-1] <= N_chp[t] for t in K if t>0)

    # MO = list(range(2))
    # for t in K:
    #     mdl.addConstr(quicksum(chp_off[t-j] for j in MO if t-j>0 ) <=1)

    # # ON/OFF of the CHP
    # for t in range(1, (N - 1)):
    #     mdl.addConstr(chp_on[t - 1] - chp_on[t] - chp_flank[t] <= 0)
    #     mdl.addConstr(-chp_on[t - 1] + chp_on[t] - chp_flank[t] <= 0)
    #     mdl.addConstr(quicksum(chp_flank[t - j] for j in list(range(self.es.chp["CHP_1"]["t_on"])) if (t - j) > 0) <= 1)
    #
    # mdl.addConstr(quicksum(chp_flank[t] for t in K if t > 0) <= self.es.chp["CHP_1"]["n_switches"])

    return q_chp_in,p_chp_use,p_chp_sold,chp_cost,chp_on,chp_off


def es_bess_opt_results(self,K,E_batt,p_batt_ch,p_batt_disch,p_batt_sold,p_batt_use):
    self.es.res["Battery"] = pd.DataFrame(columns=["E_batt","P_charge","P_discharge","P_batt_use","P_batt_sold"])
    self.es.res["Battery"]["E_batt"] = [E_batt[t].x for t in K]
    self.es.res["Battery"]["P_charge"] = [p_batt_ch[t].x for t in K]
    self.es.res["Battery"]["P_discharge"] = [p_batt_disch[t].x for t in K]
    self.es.res["Battery"]["P_batt_use"] = [p_batt_use[t].x for t in K]
    self.es.res["Battery"]["P_batt_sold"] = [p_batt_sold[t].x for t in K]

def es_chp_opt_results(self,K,p_chp_use,p_chp_sold,q_chp_in,chp_cost):
    self.es.res["CHP"] = pd.DataFrame(columns=["P_CHP_USE","P_CHP_SOLD","Q_CHP_IN","chp_cost"])
    self.es.res["CHP"]["P_CHP_USE"] = [p_chp_use[t].x for t in K]
    self.es.res["CHP"]["P_CHP_SOLD"] = [p_chp_sold[t].x for t in K]
    self.es.res["CHP"]["Q_CHP_IN"] = [q_chp_in[t].x for t in K]
    self.es.res["CHP"]["chp_cost"] = [chp_cost[t].x for t in K]

def es_pv_results(self,K,pv_prod,p_pv_use,p_pv_sold):
    self.es.res["PV"] = pd.DataFrame(columns=["PV_prod","PV_use","PV_sold"])
    self.es.res["PV"]["PV_prod"] = pv_prod
    self.es.res["PV"]["PV_use"] = [p_pv_use[t].x for t in K]
    self.es.res["PV"]["PV_sold"] = [p_pv_sold[t].x for t in K]

def es_dg_results(self,K,p_dg_out, fuel_cons_dg, e_dg_rate,n_dg_gen, p_dg_sold, p_dg_use, p_dg_cost,N_dg):
    self.es.res["DG"] = pd.DataFrame(columns=["P_DG_out", "Fuel_DG", "Emission_Rate", "P_Factor", "P_DG_sold", "P_DG_use", "P_DG_cost", "N_dg"])
    self.es.res["DG"]["P_DG_out"] = [p_dg_out[t].x for t in K]
    self.es.res["DG"]["Fuel_DG"] = [fuel_cons_dg[t].x for t in K]
    self.es.res["DG"]["Emission_Rate"] = [e_dg_rate[t].x for t in K]
    self.es.res["DG"]["P_Factor"] = [n_dg_gen[t].x for t in K]
    self.es.res["DG"]["P_DG_sold"] = [p_dg_sold[t].x for t in K]
    self.es.res["DG"]["P_DG_use"] = [p_dg_use[t].x for t in K]
    self.es.res["DG"]["P_DG_cost"] = [p_dg_cost[t].x for t in K]
    self.es.res["DG"]["N_dg"] = [N_dg[t].x for t in K]




from gurobipy import Model,GRB,quicksum
import pandas as pd

def pcs_opt_formulation_mpc(self,mdl,K,x_o,mode,cooling_demand_t,Qc_hvac,n,u_past):

    # Get the list of chillers
    chillers = list(self.tbs.pcs.chillers.keys())

    mr = 8
    t_on = list(range(-mr,0))

    cooling_demand_t = [(x * self.sim_time_step / 60) for x in cooling_demand_t]


    # # get the cooling demand
    # if from_agent == True:
    #     cooling_demand = list(self.tbs.pcs.cooling_profile["cooling_demand"])
    # else:
    #     cooling_demand = cooling_demand_t

    # %% Add the PCS Varaibales
    # Binary variable for chiler ON/OFF
    # N_c_t = mdl.addVars([(c, t) for c in chillers for t in K], vtype=GRB.CONTINUOUS, lb=0,ub=1,name="N_c_t")
    # steps = [0,0.5,1.0]
    # aux = mdl.addVars([(c,t,i) for c in chillers for t in K for i in steps],vtype=GRB.BINARY)


    # mdl.addConstrs(quicksum(aux[c,t,i] for c in chillers for t in K) ==1 for i in steps)
    #
    # # mdl.addConstrs(aux[(c, t, i) for c in chillers for t in K for i in steps].sum() ==1)
    # #
    # # mdl.addConstr(aux[(c,t) for c in chillers for t in K].sum() == 1)
    # mdl.addConstrs(N_c_t[c, t] == quicksum(aux[c,t,i] for i in steps) for c in chillers for t in K)
    #%%Add the model variables

    #Get the list of Chillers with binary control
    C_binary = [x for x in list(self.tbs.pcs.chillers.keys()) if self.tbs.pcs.chillers[x]["control"] == "Binary"]
    #Create the binary control variables
    N_c_b_t = mdl.addVars([(c, t) for c in C_binary for t in list(range(-self.tbs.pcs.chillers[c]["t_on"], 0)) + K],
                          vtype=GRB.BINARY, ub=1, lb=0, name="N_c_b_t")

    #Get the list of chiller with continuous control and make variables
    C_cont = [x for x in list(self.tbs.pcs.chillers.keys()) if self.tbs.pcs.chillers[x]["control"] == "Continuous"]
    N_c_c_t = mdl.addVars([(c, t) for c in C_cont for t in K], vtype=GRB.CONTINUOUS, ub=1, lb=0, name="N_c_c_t")

    #Add the overall control variable for chiller
    N_c_t = mdl.addVars([(c, t) for c in chillers for t in t_on+K], vtype=GRB.CONTINUOUS, ub=1, lb=0, name="N_c_t")

    # Equate this Chiller control to both binary and continuous
    mdl.addConstrs(N_c_t[c, t] == N_c_b_t[c, t] for c in chillers for t in K if c in C_binary)
    mdl.addConstrs(N_c_t[c, t] == N_c_c_t[c, t] for c in chillers for t in K if c in C_cont)

    #On and off Variables
    z_on = mdl.addVars(
        [(c, t) for c in (chillers) for t in list(range(-self.tbs.pcs.chillers[c]["t_on"], 0)) + K if
         c in C_binary], vtype=GRB.BINARY)
    z_off = mdl.addVars(
        [(c, t) for c in (chillers) for t in list(range(-self.tbs.pcs.chillers[c]["t_on"], 0)) + K if
         c in C_binary], vtype=GRB.BINARY)

    # Cooling power and demand by chillers
    qc_in_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS,lb=0,ub=100000, name="qc_in_t")
    qc_out_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="qc_out_t")

    # Electrical power of the chiller
    Pel_c = mdl.addVars([(c, t) for c in chillers for t in K], vtype=GRB.CONTINUOUS, name="Pel_chillers")

    # Total electrical power of the chiller
    p_pcs_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_pcs_t")

    # Temperature of the storage tank variable
    q_tes_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=-1000000, ub=1000000, name="e_tes_t")
    t_tes_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS,lb=-1000,ub=1000, name="t_tes_t")
    delta_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=-10000, ub=10000, name="delta_t")

    # PCS System parmeters and constant
    m_tes = (self.tbs.pcs.storages["s_cws"]["volume"] * 998.78)
    d_w = 998.78  # density of water
    c_p_w = 4.19  # Specific Heat of Water kJ/Kelvin.Kg

    #%% Add the constraints if PCS is part of optimization

    # %% PCS System

    ######################################Initialization parameters#################################################
    # Volume and state of charge of the tank
    if mode=="MILP":
        #mdl.addConstrs(delta_t[t] == 0 for t in K if t == 0)
        #mdl.addConstrs(q_tes_t[t] == 50*15*60  for t in K if t == 0)
        mdl.addConstrs((t_tes_t[t] == self.tbs.pcs.storages["s_cws"]["temp_set"] for t in K if t == 0))
    elif mode == "MPC":
        mdl.addConstrs((t_tes_t[t] == x_o['Ts_pcs'] for t in K if t == 0))
        mdl.addConstrs((delta_t[t] == x_o['delta_t'] for t in K if t == 0))
        mdl.addConstrs((q_tes_t[t] == x_o['q_tes_t'] for t in K if t == 0))


    ##################################Tank/Cooling systems#######################################################

    # Heat Flow in and heat flow out  of the tank
    if mode == "MPC":
        mdl.addConstrs(qc_out_t[t] == cooling_demand_t[t+n] for t in K)
    elif mode == "MILP":
        mdl.addConstrs(qc_out_t[t] == cooling_demand_t[t] for t in K)
    mdl.addConstrs(qc_in_t[t] == quicksum(self.tbs.pcs.chillers[c]["cooling_capacity"] * N_c_t[c, t]
                                          for c in chillers) for t in K)

    # Total Heat Energy inside the storage tank
    # mdl.addConstrs(q_tes_t[t + 1] == q_tes_t[t] + (qc_out_t[t] - qc_in_t[t]) * (self.sim_time_step * 60) for t in K if
    #                t < len(K) - 1)

    mdl.addConstrs(q_tes_t[t] ==  (qc_out_t[t] + Qc_hvac[t] - qc_in_t[t]) * (self.sim_time_step * 60) for t in K )


    # Temperature of water inside the tank
    mdl.addConstrs(delta_t[t] == (q_tes_t[t]) / (m_tes * c_p_w) for t in K )
    mdl.addConstrs(t_tes_t[t + 1] == t_tes_t[t] + delta_t[t] for t in K if t < len(K) - 1)

    ##Temperature Regulation
    mdl.addConstrs(t_tes_t[t] >= self.tbs.pcs.storages["s_cws"]["temp_min"] for t in K)
    mdl.addConstrs(t_tes_t[t] <= self.tbs.pcs.storages["s_cws"]["temp_max"] for t in K)

    #####################################Chiller Constraints and Power###########################################
    # 5 Electrical Power
    mdl.addConstrs(Pel_c[c, t] == self.tbs.pcs.chillers[c]["p_kw"] * N_c_t[c, t] for c in chillers for t in K)
    # Total PCS system power
    mdl.addConstrs(p_pcs_t[t] == quicksum(Pel_c[c, t] for c in chillers) for t in K)

    #%%Run time constraints for the chiller (in MPC) control

    #Add the past values for only binary controlled chillers
    for c in C_binary:
        mdl.addConstrs(N_c_t[c,t] == u_past[c][t+self.tbs.pcs.chillers[c]["t_on"]]
                                 for t in list(range(-self.tbs.pcs.chillers[c]["t_on"], 0)))

    #ON/OFF Equations
    for c in C_binary:
        mdl.addConstrs(N_c_t[c, t] - N_c_t[c, t - 1] == z_on[c, t] - z_off[c, t]
                       for t in list(range(-self.tbs.pcs.chillers[c]["t_on"], 0)) + K if
                       t > -self.tbs.pcs.chillers[c]["t_on"])

        mdl.addConstrs(z_on[c, t] + z_off[c, t] <= 1 for t in list(range(-self.tbs.pcs.chillers[c]["t_on"], 0)) + K)

    #Implement MO and MR times
    for c in chillers:
        if c in C_binary:
            MR = list(range(0,int(self.tbs.pcs.chillers[c]["t_on"])))
            MO = list(range(0,int(self.tbs.pcs.chillers[c]["t_off"])))
            if n==2:
                A = "STOP"
            #mdl.addConstrs(quicksum(z_on[k, t - j] for j in MR) <= 1  for t in K)
            mdl.addConstrs(quicksum(z_on[c,t-j] for j in MR ) <= 1-z_off[c,t] for t in K)
            mdl.addConstrs(quicksum(z_off[c, t - j] for j in MO ) <= 1 - z_on[c, t] for t in K)

    #Chiller
    # mdl.addConstrs(qc_in_t[t - 1] - qc_in_t[t] <= 10 for t in K if t > 0)
    # mdl.addConstrs(qc_in_t[t - 1] - qc_in_t[t] >= -10 for t in K if t > 0)


    return p_pcs_t,Pel_c,t_tes_t,q_tes_t,qc_in_t,qc_out_t,chillers,N_c_t,delta_t

def pcs_opt_results(self,K,p_pcs_t,N_c_t,Pel_c,t_tes_t,qc_out_t,qc_in_t,Qc_hvac):

    chillers = list(self.tbs.pcs.chillers.keys())

    #%%Append Total and individual load profiles of the compressors
    self.tbs.pcs.res["Load_profiles"] = pd.DataFrame(columns=["KW_Total"])
    self.tbs.pcs.res["Load_profiles"]["KW_Total"] = [p_pcs_t[t].x for t in K]

    for c in chillers:
        self.tbs.pcs.res["Load_profiles"][c] = [Pel_c[c,t].x for t in K]

    #%%Append the State Variables of the CAS
    self.tbs.pcs.res["States"] = pd.DataFrame(columns=["Q_in","Q_out","Qc_hvac","T_s_pcs","Price"])
    self.tbs.pcs.res["States"]["Q_in"] =  [qc_in_t[t].x for t in K]
    self.tbs.pcs.res["States"]["Q_out"] = [qc_out_t[t].x for t in K]
    self.tbs.pcs.res["States"]["Qc_hvac"] = [Qc_hvac[t].x for t in K]
    self.tbs.pcs.res["States"]["T_s_pcs"] = [t_tes_t[t].x for t in K]
    self.tbs.pcs.res["States"]["Price"] = self.day_ahead_prices

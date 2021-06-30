from gurobipy import Model,GRB,quicksum
import pandas as pd

def cas_opt_formulation_mpc(self,mdl,K,x_o,mode,from_agent,air_demand_opt,n,u_past):

    mr = 8
    mo = 2

    t_on = list(range(-mr,0))
    #t_off = list(range(-mo, 0))

    #%% Get the compressors keys
    compressors = list(self.tbs.cas.compressors.keys())

    #%% Get the air demand profile from optimized task schedule (based on N_i_t)
    if from_agent == True:
        air_demand = list(self.tbs.cas.air_profile["air_demand"])
    elif from_agent == False:
        air_demand = air_demand_opt

    #Ressample the air demand as per MPC Time Scale
    air_demand_opt = [(x * self.sim_time_step / 60) for x in air_demand_opt]

    # Get the storage tank parameters
    Vstor = float(self.tbs.cas.storages["S_air"]["capacity"])
    soc_ini = float(self.tbs.cas.storages["S_air"]["soc_initial"])
    soc_min = float(self.tbs.cas.storages["S_air"]["soc_min"])
    soc_max = float(self.tbs.cas.storages["S_air"]["soc_max"])

    p_min = float(self.tbs.cas.storages["S_air"]["pressure_min"])
    p_max = float(self.tbs.cas.storages["S_air"]["pressure_max"])
    p_set = float(self.tbs.cas.storages["S_air"]["p_set"])

    # Constants for CAS System
    density_air = 1.1839  # kg/m3
    R_air = 287  # J/Kg
    Temp = 298.15  # Kelvin

    K_const = Vstor / (R_air * Temp) * 100000



    #%% Add the model variables
    # Get the set of Compressors with Binary Control
    K_binary = [x for x in list(self.tbs.cas.compressors.keys()) if self.tbs.cas.compressors[x]["control"] == "Binary"]
    #Create the binary control variables
    #N_k_b_t = mdl.addVars([(k, t) for k in compressors for t in t_on+K], vtype=GRB.BINARY, ub=1, lb=0, name="N_k_b_t")
    N_k_b_t = mdl.addVars([(k, t) for k in K_binary for t in list(range(-self.tbs.cas.compressors[k]["t_on"], 0)) + K],
                          vtype=GRB.BINARY, ub=1, lb=0, name="N_k_b_t")

    # Get the set of Compressors with Continuous Control
    K_cont = [x for x in list(self.tbs.cas.compressors.keys()) if self.tbs.cas.compressors[x]["control"] == "Continuous"]
    N_k_c_t = mdl.addVars([(k, t) for k in K_cont for t in K], vtype=GRB.CONTINUOUS, ub=1, lb=0, name="N_k_c_t")

    # Add the overal control Variable for compressors
    N_k_t = mdl.addVars([(k, t) for k in compressors for t in t_on+K], vtype=GRB.CONTINUOUS, ub=1, lb=0, name="N_k_t")

    # Equate this Compressor control to both binary and continuous
    mdl.addConstrs(N_k_t[k, t] == N_k_b_t[k, t] for k in compressors for t in K if k in K_binary)
    mdl.addConstrs(N_k_t[k, t] == N_k_c_t[k, t] for k in compressors for t in K if k in K_cont)

    #Binary variable for compresssor ON/OFF
    # N_k_t = mdl.addVars([(k, t) for k in compressors for t in K], vtype=GRB.CONTINUOUS, ub=1, lb=0, name="N_k_t")

    #Electrical power of the compressor
    Pel_k = mdl.addVars([(k, t) for k in compressors for t in K], vtype=GRB.CONTINUOUS, name="Pel_k")

    # Total electrical power of the CAS Systems
    p_cas_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_t")

    # Varibale for no. of compressors  being executed in each time step
    count_k_t = mdl.addVars(len(K), vtype=GRB.BINARY)

    #Switching variable for the compressor
    S_k_t = mdl.addVars([(k, t) for k in compressors for t in K if k in K_binary], vtype=GRB.CONTINUOUS, lb=0)

    # variable for compressor being ON or OFF
    z_on = mdl.addVars([(k, t) for k in (compressors) for t in list(range(-self.tbs.cas.compressors[k]["t_on"], 0)) + K if k in K_binary], vtype=GRB.BINARY)
    z_off = mdl.addVars([(k, t) for k in (compressors) for t in list(range(-self.tbs.cas.compressors[k]["t_on"], 0)) + K if k in K_binary], vtype=GRB.BINARY)

    # Volume/state of charge of air in the tank
    v_stor_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="e_stor_t")

    # Mass Flow of air coming in from the compressor in the circuit
    v_in = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="v_in")

    # Mass Flow of Air going out from the compressor
    v_out = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="v_out")


    # Pressure inside the air storange tank
    p_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_t")
    delta_p = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="delta_p", lb=-1000, ub=1000)

    # %% CAS System Model Constraints
    ######################################Initialization parameters#################################################
    if mode == "MILP":
    # Volume and state of charge of the tank
        mdl.addConstrs(v_stor_t[t] == (Vstor * float(soc_ini)) for t in K if t == 0)
        mdl.addConstrs((p_t[t] == p_set for t in K if t == 0))
    elif mode == "MPC":
        mdl.addConstrs(v_stor_t[t] == x_o['v_stor_t']  for t in K if t == 0)
        mdl.addConstrs((p_t[t] == x_o['p_t'] for t in K if t == 0))
        mdl.addConstrs(v_out[t] == air_demand_opt[t+n] for t in K)

    ##################################Tank/Circuit Parameters#######################################################

    # # Flow in and Flow out of the tanks
    # if from_agent == True:
    #     mdl.addConstrs(v_out[t] == air_demand[t] for t in K)
    # else:
    #     mdl.addConstrs(v_out[t] == air_demand_opt[t] for t in K)

    mdl.addConstrs(v_in[t] == (quicksum(self.tbs.cas.compressors[k]["m3_per_min"] * self.sim_time_step * N_k_t[k, t]
                                       for k in compressors)) for t in K)

    # Volume/State of charge of the air in the tank
    mdl.addConstrs(v_stor_t[t + 1] == (v_stor_t[t] - v_out[t] + v_in[t]) for t in K if t < len(K) - 1)

    #Take the air from storage or directly from compressor


    # At all the times volume of the tank must be enough to take in
    # mdl.addConstrs(v_stor_t[t]>=v_in[t] for t in K if t>0)

    # Min and Max volume of the tank
    mdl.addConstrs(v_stor_t[t] >= Vstor * soc_min for t in K)
    mdl.addConstrs(v_stor_t[t] <= Vstor * soc_max for t in K)

    # #pressure in the storage tank
    mdl.addConstrs(delta_p[t] == (1 / K_const) * (v_in[t] - v_out[t]) * density_air for t in K)
    mdl.addConstrs(p_t[t + 1] == p_t[t] + delta_p[t] for t in K if t < len(K) - 1)
    #
    # #Pressure regulation
    mdl.addConstrs(p_t[t] >= p_min for t in K)
    mdl.addConstrs(p_t[t] <= p_max for t in K)

    #####################################Compressor Constraints and Power###########################################
    # 5 Electrical Power
    mdl.addConstrs(Pel_k[k, t] == self.tbs.cas.compressors[k]["p_kw"] * N_k_t[k, t] for k in compressors for t in K)
    # Total Compressor Power
    mdl.addConstrs(p_cas_t[t] == quicksum(Pel_k[k, t] for k in compressors) for t in K)

    #Minimum Run time, Minimum Off time of the compressors

    #Add the past values only for Binary Compressors
    for k in K_binary:
        mdl.addConstrs(N_k_t[k,t] == u_past[k][t+self.tbs.cas.compressors[k]["t_on"]]
                                 for t in list(range(-self.tbs.cas.compressors[k]["t_on"], 0)))


    for k in K_binary:
        mdl.addConstrs(N_k_t[k,t] - N_k_t[k,t-1] == z_on[k,t] - z_off[k,t]
                       for t in list(range(-self.tbs.cas.compressors[k]["t_on"], 0))+K if t>-self.tbs.cas.compressors[k]["t_on"])

        mdl.addConstrs(z_on[k,t] + z_off[k,t] <=1 for t in list(range(-self.tbs.cas.compressors[k]["t_on"], 0))+K)


    #Run time constraints only for Binary Controlled Compressors
    for k in compressors:
        if k in K_binary:
            MR = list(range(0,int(self.tbs.cas.compressors[k]["t_on"])))
            MO = list(range(0,int(self.tbs.cas.compressors[k]["t_off"])))
            if n==2:
                A = "STOP"
            #mdl.addConstrs(quicksum(z_on[k, t - j] for j in MR) <= 1  for t in K)
            mdl.addConstrs(quicksum(z_on[k,t-j] for j in MR ) <= 1-z_off[k,t] for t in K)
            mdl.addConstrs(quicksum(z_off[k, t - j] for j in MO ) <= 1 - z_on[k, t] for t in K)




    # 2.2 Calculate the switching varibale, whenever task starts and ends
    # mdl.addConstrs(N_k_t[k, t - 1] - N_k_t[k, t] - S_k_t[k, t] <= 0 for k in compressors for t in K if t > 0)
    # mdl.addConstrs(-N_k_t[k, t - 1] + N_k_t[k, t] - S_k_t[k, t] <= 0 for k in compressors for t in K if t > 0)

    # for k in compressors:
    #     MR = list(range(0,2))
    #     for t in K:
    #         mdl.addConstr(quicksum(S_k_t[k, t - j] for j in MR if t - j > 0) <= 1)

    return p_cas_t,Pel_k,v_out,v_in,v_stor_t,p_t,N_k_t,compressors,z_on,z_off


def cas_opt_results(self,K,p_cas_t,N_k_t,Pel_k,v_stor_t,v_in,v_out,p_t,mode):
    compressors = list(self.tbs.cas.compressors.keys())

    #%%Append Total and individual load profiles of the compressors
    self.tbs.cas.p_cas_res["Load_profiles"] = pd.DataFrame(columns=["KW_Total"])
    if mode == "MILP":
        self.tbs.cas.p_cas_res["Load_profiles"]["KW_Total"] = [p_cas_t[t].x for t in K]
    elif mode == "MPC":
        self.tbs.cas.p_cas_res["Load_profiles"]["KW_Total"] = p_cas_t

    for k in compressors:
        self.tbs.cas.p_cas_res["Load_profiles"][k] = [Pel_k[k,t].x for t in K]

    #%%Append the State Variables of the CAS
    if mode == "MILP":
        self.tbs.cas.p_cas_res["States"] = pd.DataFrame(columns=["Vair_out","Vair_in","pressure_t"])
        self.tbs.cas.p_cas_res["States"]["Vair_out"] =  [v_out[t].x for t in K]
        self.tbs.cas.p_cas_res["States"]["Vair_in"] = [v_in[t].x for t in K]
        self.tbs.cas.p_cas_res["States"]["pressure_t"] = [p_t[t].x for t in K]
    elif mode == "MPC":
        self.tbs.cas.p_cas_res["States"] = pd.DataFrame(columns=["Vair_out", "Vair_in", "pressure_t"])
        self.tbs.cas.p_cas_res["States"]["Vair_out"] = v_out
        self.tbs.cas.p_cas_res["States"]["Vair_in"] = v_in
        self.tbs.cas.p_cas_res["States"]["pressure_t"] = p_t








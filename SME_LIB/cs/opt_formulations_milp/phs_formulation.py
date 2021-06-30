from gurobipy import Model,GRB,quicksum
import pandas as pd

def phs_opt_formulation(self,mdl,K,x_o,mode,from_agent,heating_demand_t,q_chp_in):
    # Get the list of heaters
    heaters = list(self.tbs.phs.heat_pumps.keys())
    heating_demand = heating_demand_t

    #Add the PHS Variables
    N_h_t = mdl.addVars([(h, t) for h in heaters for t in K], vtype=GRB.BINARY,ub=1,lb=0, name="N_h_t")

    #Add the Sotrage Internal Energy Variable
    U_test_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=0, ub=100000, name="U_tes_t")


    #Heating power in by heaters
    qh_in_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=-10000, ub=100000, name="qh_in_t")
    qh_out_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="qh_out_t")

    #Electrical Power of the heater
    Pel_h = mdl.addVars([(h, t) for h in heaters for t in K], vtype=GRB.CONTINUOUS, name="Pel_heaters")

    # Total electrical power of the PHS
    p_phs_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_phs_t")

    #Temperature of the storage tank variables
    # Temperature of the storage tank variable
    q_phs_tes_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=-10000, ub=100000, name="q_phs_tes_t")
    t_phs_tes_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=-1000, ub=1000, name="t_phs_tes_t")
    delta_phs_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=-10000, ub=10000, name="delta_t")

    #PCS System parmeters and constant
    m_phs_tes = (self.tbs.phs.storages["S_PHS"]["capacity"] * 998.78)
    d_w = 998.78  # density of water
    c_p_w = 4.19  # Specific Heat of Water kJ/Kelvin.Kg

    #Add the constraits
    #Initialization
    mdl.addConstrs((t_phs_tes_t[t] == self.tbs.phs.storages["S_PHS"]["temp_set"] for t in K if t == 0))
    mdl.addConstrs((q_phs_tes_t[t] == 0 for t in K if t == 0))

    # Heat Flow in and heat flow out  of the tank
    mdl.addConstrs(qh_out_t[t] == heating_demand_t[t] for t in K)
    mdl.addConstrs(qh_in_t[t] == quicksum(self.tbs.phs.heat_pumps[h]["p_kw"] * self.tbs.phs.heat_pumps[h]["cop"]*
                                          N_h_t[h, t] for h in heaters) for t in K)

    #Energy balance equation for the change in teperature
    mdl.addConstrs(q_phs_tes_t[t+1] == q_phs_tes_t[t] + (qh_in_t[t]+q_chp_in[t]-qh_out_t[t])*(self.sim_time_step*60)
                                        for t in K if t<len(K)-1)

    #detla T of the tank
    mdl.addConstrs(delta_phs_t[t] == (q_phs_tes_t[t]) / (m_phs_tes * c_p_w) for t in K)
    mdl.addConstrs(t_phs_tes_t[t+1] == t_phs_tes_t[t]+delta_phs_t[t] for t in K if t<len(K)-1)

    #Temperature regulation
    mdl.addConstrs(t_phs_tes_t[t] >= self.tbs.phs.storages["S_PHS"]["temp_min"] for t in K)
    mdl.addConstrs(t_phs_tes_t[t] <= self.tbs.phs.storages["S_PHS"]["temp_max"] for t in K)

    #Calculate the electrical power
    mdl.addConstrs(Pel_h[h,t] == self.tbs.phs.heat_pumps[h]["p_kw"]*N_h_t[h,t] for h in heaters for t in K)
    #Total PHS Power of the system
    mdl.addConstrs(p_phs_t[t] == quicksum(Pel_h[h, t] for h in heaters) for t in K)

    return p_phs_t,Pel_h,t_phs_tes_t,qh_in_t,qh_out_t,q_chp_in,heaters,N_h_t,delta_phs_t,q_phs_tes_t


def phs_opt_formulation_energy_based(self,mdl,K,x_o,mode,from_agent,heating_demand_t,q_chp_in,Qh_hvac,hvac_opt):

    #Get the storage parameters
    if hvac_opt==True:
        T_amb = self.tbs.hvac.t_amb
    else:
        T_amb = [0]*len(K)
    T_tes_ini = self.tbs.phs.storages["S_PHS"]["temp_set"] #initial starting temperature of the tank
    V_tes = self.tbs.phs.storages["S_PHS"]["volume"] #m3
    K_v = self.tbs.phs.storages["S_PHS"]["k_v"]
    #Constants
    row_tes = 998 #Kg/m3
    c_tes   = 4.19 #kJ/Kg.K
    K_v = 0.0534/100 # thermal storage loss factor
    T_max = self.tbs.phs.storages["S_PHS"]["temp_max"]
    T_min = self.tbs.phs.storages["S_PHS"]["temp_min"]


    # Get the list of heaters
    heaters = list(self.tbs.phs.heat_pumps.keys())
    heating_demand = heating_demand_t

    #Get the set of Heater with Binary Control
    H_binary = [x for x in list(self.tbs.phs.heat_pumps.keys()) if self.tbs.phs.heat_pumps[x]["control"] == "Binary"]
    N_h_b_t = mdl.addVars([(h, t) for h in heaters for t in K], vtype=GRB.BINARY,ub=1,lb=0, name="N_h_t")

    # Get the set of Heater with Continuous
    H_cont = [x for x in list(self.tbs.phs.heat_pumps.keys()) if self.tbs.phs.heat_pumps[x]["control"] == "Continuous"]
    N_h_c_t = mdl.addVars([(h, t) for h in heaters for t in K], vtype=GRB.CONTINUOUS, ub=1, lb=0, name="N_h_c_t")

    #Add the overal control Variable for heater
    N_h_t = mdl.addVars([(h, t) for h in heaters for t in K], vtype=GRB.CONTINUOUS,ub=1,lb=0, name="N_h_t")
    #Equate this heater control to both binary and continuous
    mdl.addConstrs(N_h_t[h,t] == N_h_b_t[h,t] for h in heaters for t in K if h in H_binary)
    mdl.addConstrs(N_h_t[h, t] == N_h_c_t[h, t] for h in heaters for t in K if h in H_cont)

    # ON/OFF variables
    z_on = mdl.addVars([(h, t) for h in (H_binary) for t in  K],
                       vtype=GRB.BINARY)
    z_off = mdl.addVars([(h, t) for h in (H_binary) for t in K],
                        vtype=GRB.BINARY)

    #Add the Sotrage Internal Energy Variable
    H_test_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=0, ub=100000, name="H_tes_t")
    T_tes_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=-1000, ub=100000, name="T_tes_t")

    #Heating power in by heaters
    Qh_in_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=0, ub=100000, name="Qh_in_c_t")
    Qchp_in_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=-10000, ub=100000, name="Qh_in_t")
    Qmfs_out_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="Qmfs_out_t")
    Qh_hvac_out = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="Qh_hvac_out")


    #Electrical Power of the heaters
    Pel_h = mdl.addVars([(h, t) for h in heaters for t in K], vtype=GRB.CONTINUOUS, name="Pel_heaters")
    # Total electrical power of the PHS
    p_phs_t = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, name="p_phs_t")


    #Add the constraits
    #Initialization
    mdl.addConstrs((H_test_t[t] == (row_tes*V_tes)*c_tes*(T_tes_ini-T_amb[t])/3600 for t in K if t == 0)) #in KWH
    mdl.addConstrs(T_tes_t[t] == T_tes_ini for t in K if t==0)


    # Heat Energy  Flow in and heat flow out  of the tank
    #From MFS
    mdl.addConstrs(Qmfs_out_t[t] == heating_demand_t[t]  for t in K)
    #From Heat Pumps
    mdl.addConstrs(Qh_in_t[t] == quicksum(self.tbs.phs.heat_pumps[h]["p_kw"] * self.tbs.phs.heat_pumps[h]["cop"] *
                                            N_h_t[h, t] for h in heaters) for t in K)  # kwh

    #From CHP
    mdl.addConstrs(Qchp_in_t[t] == q_chp_in[t]  for t in K) #kwh


    #Energy balance equation for the change in teperature

    mdl.addConstrs(H_test_t[t+1] ==H_test_t[t] + (Qh_in_t[t]+q_chp_in[t]-Qmfs_out_t[t]-Qh_hvac[t])*(self.sim_time_step/60)
                                        for t in K if t<len(K)-1)

    #Calculate the temperature of the tank
    mdl.addConstrs(T_tes_t[t] == T_amb[t] + (H_test_t[t]/(row_tes*V_tes*c_tes))*3600  for t in K if t>0)

    #Constraints on the tank
    # mdl.addConstrs(H_test_t[t]<=200 for t in K)
    # mdl.addConstrs(H_test_t[t] >= 0 for t in K)

    # #Temperature regulation
    mdl.addConstrs(T_tes_t[t] >= T_min for t in K)
    mdl.addConstrs(T_tes_t[t] <= T_max for t in K)

    #Calculate the electrical power
    mdl.addConstrs(Pel_h[h,t] == self.tbs.phs.heat_pumps[h]["p_kw"]*N_h_t[h,t] for h in heaters for t in K)
    #Total PHS Power of the system
    mdl.addConstrs(p_phs_t[t] == quicksum(Pel_h[h, t] for h in heaters) for t in K)


    #%%Run time constraints

    for h in H_binary:
        mdl.addConstrs(N_h_t[h, t] - N_h_t[h, t - 1] == z_on[h, t] - z_off[h, t]
                       for t in  K if
                       t > 0)

        mdl.addConstrs(z_on[h, t] + z_off[h, t] <= 1 for t in K)

    # Implement MO and MR times
    for h in H_binary:
            MR = list(range(0, int(self.tbs.phs.heat_pumps[h]["t_on"])))
            MO = list(range(0, int(self.tbs.phs.heat_pumps[h]["t_off"])))
            # mdl.addConstrs(quicksum(z_on[k, t - j] for j in MR) <= 1  for t in K)
            mdl.addConstrs(quicksum(z_on[h, t - j] for j in MR if t-j>0) <= 1 - z_off[h, t] for t in K)
            mdl.addConstrs(quicksum(z_off[h, t - j] for j in MO if t-j>0) <= 1 - z_on[h, t] for t in K)

    return N_h_t,p_phs_t,Pel_h,H_test_t,T_tes_t,Qh_in_t,Qmfs_out_t,Qchp_in_t


def phs_opt_results(self,K,p_phs_t,Pel_h,N_h_t,t_phs_tes_t,qh_out_t,qh_in_t,q_chp_in,q_phs_tes_t,Qh_hvac):
    heaters = list(self.tbs.phs.heat_pumps.keys())

    #%%Append Total and individual load profiles of the compressors
    self.tbs.phs.res["Load_profiles"] = pd.DataFrame(columns=["KW_Total"])
    self.tbs.phs.res["Load_profiles"]["KW_Total"] = [p_phs_t[t].x for t in K]

    for h in heaters:
        self.tbs.phs.res["Load_profiles"][h] = [Pel_h[h,t].x for t in K]

    #%%Append the State Variables of the CAS
    self.tbs.phs.res["States"] = pd.DataFrame(columns=["Qh_out","Q_in","Q_chp_in","T_s_phs","E_PHS","Price"])
    self.tbs.phs.res["States"]["Qh_out_KW"] = [qh_out_t[t].x for t in K]
    self.tbs.phs.res["States"]["Q_phs_KW"] =  [qh_in_t[t].x for t in K]
    self.tbs.phs.res["States"]["Q_chp_KW"] = [q_chp_in[t].x for t in K]
    self.tbs.phs.res["States"]["T_s_phs"] = [t_phs_tes_t[t].x for t in K]
    self.tbs.phs.res["States"]["E_PHS"] = [(q_phs_tes_t[t].x)/3600 for t in K]
    self.tbs.phs.res["States"]["Price"] = self.day_ahead_prices

def phs_opt_results_energy_based(self,K,p_phs_t, Pel_h, H_test_t, T_tes_t, Qh_in_t,Qchp_in_t,Qmfs_out_t,Qh_hvac):
    heaters = list(self.tbs.phs.heat_pumps.keys())

    #%%Append Total and individual load profiles of the compressors
    self.tbs.phs.res["Load_profiles"] = pd.DataFrame(columns=["KW_Total"])
    self.tbs.phs.res["Load_profiles"]["KW_Total"] = [p_phs_t[t].x for t in K]

    for h in heaters:
        self.tbs.phs.res["Load_profiles"][h] = [Pel_h[h,t].x for t in K]

    #%%Append the State Variables of the CAS
    self.tbs.phs.res["States"] = pd.DataFrame(columns=["H_tes_kwh","T_s","T_amb","Qmfs_out","Qh_hvac","Qh_in_kw","Qchp_in_kw","Price"])
    self.tbs.phs.res["States"]["H_tes_kwh"] = [H_test_t[t].x for t in K]
    self.tbs.phs.res["States"]["T_s"] =  [T_tes_t[t].x for t in K]
    self.tbs.phs.res["States"]["T_amb"] = self.tbs.hvac.t_amb
    self.tbs.phs.res["States"]["Qmfs_out"] = [Qmfs_out_t[t].x for t in K]
    self.tbs.phs.res["States"]["Qh_hvac"] = [Qh_hvac[t].x for t in K]
    self.tbs.phs.res["States"]["Qh_in_kw"] = [Qh_in_t[t].x for t in K]
    self.tbs.phs.res["States"]["Qchp_in_kw"] = [Qchp_in_t[t].x for t in K]
    self.tbs.phs.res["States"]["Price"] = self.day_ahead_prices



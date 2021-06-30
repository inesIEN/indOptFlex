from gurobipy import Model,GRB,quicksum
import pandas as pd

def hvac_opt_formulation_mpc(self,mdl,K,x_o,mode,n):

    #Add the model variable

    T_r = mdl.addVars(len(K), vtype=GRB.CONTINUOUS,lb=-1000,ub=1000, name="t_room_t")
    T_amb = self.tbs.hvac.t_amb
    #T_amb = mdl.addVars(len(K),vtype=GRB.CONTINUOUS,name="T_amb")
    Qc_hvac = mdl.addVars(len(K), vtype=GRB.CONTINUOUS,lb=-0,ub=1000, name="Qc_hvac") # Cooling Power
    Qh_hvac = mdl.addVars(len(K), vtype=GRB.CONTINUOUS, lb=-0, ub=1000, name="Qh_hvac") #Heating Power

    #Get the constants for the room

    C_r = self.tbs.hvac.thermal_capacity # in kJ/K
    U_r = self.tbs.hvac.heat_loss_coeff  # in kW/K

    #set the initial room temperature:
    t_r_ini = self.tbs.hvac.t_set

    #Modelling equation
    #Initialization
    if mode == "MILP":
        mdl.addConstrs(T_r[t] == t_r_ini for t in K if t==0)
        mdl.addConstrs(Qc_hvac[t] == 0 for t in K if t==0)
        mdl.addConstrs(Qh_hvac[t] == 0 for t in K if t == 0)
    elif mode == "MPC":
        mdl.addConstrs(T_r[t] == x_o["T_r"] for t in K if t==0)

    #Equate the Tamb (Optimization Variable to Parameter)
    #mdl.addConstrs(T_amb[t] == T_ambient[t+n] for t in K)
    #State Equation
    mdl.addConstrs(T_r[t+1] == T_r[t] + (U_r/C_r)*((T_amb[t+n]-T_r[t])*self.sim_time_step*60) +
                                1/C_r*(Qh_hvac[t]-Qc_hvac[t])*self.sim_time_step*60 for t in K if t < len(K) - 1)


    mdl.addConstrs(T_r[t] >= self.tbs.hvac.t_set_min for t in K)
    mdl.addConstrs(T_r[t] <= self.tbs.hvac.t_set_max for t in K)


    return T_r, T_amb, Qc_hvac, Qh_hvac


def hvac_opt_results(self,K,Qh_hvac,Qc_hvac,T_amb,T_r):
    self.tbs.hvac.res["States"] = pd.DataFrame(columns=["Qh_hvac","Qc_hvac","T_amb","T_r", "Price"])
    self.tbs.hvac.res["States"]["Qh_hvac"] = [Qh_hvac[t].x for t in K]
    self.tbs.hvac.res["States"]["Qc_hvac"] = [Qc_hvac[t].x for t in K]
    self.tbs.hvac.res["States"]["T_amb"] = T_amb
    self.tbs.hvac.res["States"]["T_r"] = [T_r[t].x for t in K]
    self.tbs.hvac.res["States"]["Price"] = self.day_ahead_prices

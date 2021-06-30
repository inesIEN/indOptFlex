import pandas as pd


def mfs_opt_results(self,K,p_mfs_t,p_r_t,R_r_t,N_i_t,R_s_t,mode):

    #Get the requried data
    machines = list(self.mfs.machines.keys())
    storages = list(self.mfs.storage_facilities.keys())
    products = list(self.mfs.products.keys())
    raw_materials = list(self.mfs.raw_materials.keys())
    resources = storages + raw_materials + products
    tasks = list(self.mfs.tasks.keys())

    #%%Append Total and individual load profiles of the Machines
    self.mfs.res["Load_profiles"] = pd.DataFrame(columns=["mfs_kw_total"])
    if mode == "MPC":
        self.mfs.res["Load_profiles"]["KW_Total"] = p_mfs_t
    elif mode == "MILP":
        self.mfs.res["Load_profiles"]["KW_Total"] = [p_mfs_t[t].x for t in K]

    for m in machines:
        self.mfs.res["Load_profiles"][m] = [p_r_t[m,t].x for t in K]

    #%%Append the Task schedule
    self.mfs.res["Tasks"] = pd.DataFrame(columns=[i for i in tasks])
    for i in tasks:
        self.mfs.res["Tasks"][i] = [N_i_t[i,t].x for t in K]

    #%%Append the Storages Results
    self.mfs.res["Storages"] = pd.DataFrame(columns=[s for s in storages])
    for s in storages:
        self.mfs.res["Storages"][s] = [R_s_t[s,t].x for t in K]

    # %%Append the Product Results
    self.mfs.res["Products"] = pd.DataFrame(columns=[p for p in products])
    for p in products:
        self.mfs.res["Products"][p] = [R_r_t[p, t].x for t in K]

    # %%Append the RM
    self.mfs.res["Raw-materials"] = pd.DataFrame(columns=[r_m for r_m in raw_materials])
    for r_m in raw_materials:
        self.mfs.res["Raw-materials"][r_m] = [R_r_t[r_m, t].x for t in K]

def cas_opt_results(self, K, p_cas_t, N_k_t, Pel_k, v_stor_t, v_in, v_out, p_t,n=0):
    #Get the list of compressors
    compressors = list(self.tbs.cas.compressors.keys())
    #Append the total power of CAS system
    self.tbs.cas.res_df.loc[self.tbs.cas.res_df.index[n:n + len(K)], "p_cas_t"] = [p_cas_t[t].x for t in K]
    #Append the power of each compressor
    for k in compressors:
        # Electrical Power of Each Compressor "k"
        self.tbs.cas.res_df.loc[self.tbs.cas.res_df.index[n:n + len(K)], k] = [Pel_k[k, t].x for t in K]
        # N_k_t for each compressor "k"
        self.tbs.cas.res_df.loc[self.tbs.cas.res_df.index[n:n + len(K)], "N_k_t-" + k] = [N_k_t[k, t].x for t in K]

    #Append the pressure
    self.tbs.cas.res_df.loc[self.tbs.cas.res_df.index[n:n + len(K)], "p_t"] = [p_t[t].x for t in K]



def pcs_opt_results(self,K,p_pcs_t,N_c_t,Pel_c,t_pcs_t,qc_out_t,qc_in_t,Qc_hvac,n=0):
    # Get the list of chillers
    chillers = list(self.tbs.pcs.chillers.keys())

    # Append the total power of PCS system
    self.tbs.pcs.res_df.loc[self.tbs.pcs.res_df.index[n:n + len(K)], "p_pcs_t"] = [p_pcs_t[t].x for t in K]
    # Append the power of each compressor
    for c in chillers:
        # Electrical Power of Each Compressor "k"
        self.tbs.pcs.res_df.loc[self.tbs.pcs.res_df.index[n:n + len(K)], c] = [Pel_c[c, t].x for t in K]
        # N_c_t for each chiller "c"
        self.tbs.pcs.res_df.loc[self.tbs.pcs.res_df.index[n:n + len(K)], "N_c_t-" + c] = [N_c_t[c, t].x for t in K]

    # Append the Temperature of the thermal storage tank
    self.tbs.pcs.res_df.loc[self.tbs.pcs.res_df.index[n:n + len(K)], "t_pcs_t"] = [t_pcs_t[t].x for t in K]

    #Append the Heat and Cooling In and Out and Qc for HVAC
    self.tbs.pcs.res_df.loc[self.tbs.pcs.res_df.index[n:n + len(K)], "qc_out_t"] = [qc_out_t[t].x for t in K]
    self.tbs.pcs.res_df.loc[self.tbs.pcs.res_df.index[n:n + len(K)], "qc_in_t"] = [qc_in_t[t].x for t in K]
    self.tbs.pcs.res_df.loc[self.tbs.pcs.res_df.index[n:n + len(K)], "Qc_hvac"] = [Qc_hvac[t].x for t in K]
    
    
def phs_opt_results(self,K,p_phs_t, Pel_h,N_h_t, H_test_t, T_phs_t, Qh_in_t,Qchp_in_t,Qmfs_out_t,Qh_hvac,n=0):
    
    #Get the list of heaters
    heaters = list(self.tbs.phs.heat_pumps.keys())

    # Append the total power of PCS system
    self.tbs.phs.res_df.loc[self.tbs.phs.res_df.index[n:n + len(K)], "p_phs_t"] = [p_phs_t[t].x for t in K]
    # Append the power of each compressor
    for h in heaters:
        # Electrical Power of each heater "h"
        self.tbs.phs.res_df.loc[self.tbs.phs.res_df.index[n:n + len(K)], h] = [Pel_h[h, t].x for t in K]
        # N_h_t for each heater "h"
        self.tbs.phs.res_df.loc[self.tbs.phs.res_df.index[n:n + len(K)], "N_h_t-" + h] = [N_h_t[h, t].x for t in K]

    # Append the Temperature of the thermal storage tank
    self.tbs.phs.res_df.loc[self.tbs.phs.res_df.index[n:n + len(K)], "t_phs_t"] = [T_phs_t[t].x for t in K]

    # Append the Heat and Cooling In and Out and Qc for HVAC
    self.tbs.phs.res_df.loc[self.tbs.phs.res_df.index[n:n + len(K)], "Qh_in_t"] = [Qh_in_t[t].x for t in K]
    self.tbs.phs.res_df.loc[self.tbs.phs.res_df.index[n:n + len(K)], "Qchp_in_t"] = [Qchp_in_t[t].x for t in K]
    self.tbs.phs.res_df.loc[self.tbs.phs.res_df.index[n:n + len(K)], "Qmfs_out_t"] = [Qmfs_out_t[t].x for t in K]
    self.tbs.phs.res_df.loc[self.tbs.phs.res_df.index[n:n + len(K)], "Qh_hvac"] = [Qh_hvac[t].x for t in K]


def es_bess_opt_results(self,K,E_batt,p_batt_ch,p_batt_disch,p_batt_sold,p_batt_use,n=0):


    for b in list(self.es.ess.keys()):
        self.es.ess[b]["res_df"].loc[self.es.ess[b]["res_df"].index[n:n + len(K)], "E_batt-"+b] = [E_batt[t].x for t in K]
        self.es.ess[b]["res_df"].loc[self.es.ess[b]["res_df"].index[n:n + len(K)], "p_batt_ch-"+b] = [p_batt_ch[t].x for t in K]
        self.es.ess[b]["res_df"].loc[self.es.ess[b]["res_df"].index[n:n + len(K)], "p_batt_disch-"+b] = [p_batt_disch[t].x for t in K]
        self.es.ess[b]["res_df"].loc[self.es.ess[b]["res_df"].index[n:n + len(K)], "p_batt_sold-"+b] = [p_batt_sold[t].x for t in K]
        self.es.ess[b]["res_df"].loc[self.es.ess[b]["res_df"].index[n:n + len(K)], "p_batt_use-"+b] = [p_batt_use[t].x for t in K]

def es_chp_opt_results(self,K,p_chp_use,p_chp_sold,q_chp_in,chp_cost,n=0):

    for j in list(self.es.chp.keys()):
        self.es.chp[j]["res_df"].loc[self.es.chp[j]["res_df"].index[n:n + len(K)], "p_chp_use-"+j] = [p_chp_use[t].x for t in K]
        self.es.chp[j]["res_df"].loc[self.es.chp[j]["res_df"].index[n:n + len(K)], "p_chp_sold-"+j] = [p_chp_sold[t].x for t in K]
        self.es.chp[j]["res_df"].loc[self.es.chp[j]["res_df"].index[n:n + len(K)], "q_chp_in-"+j] = [q_chp_in[t].x for t in K]
        self.es.chp[j]["res_df"].loc[self.es.chp[j]["res_df"].index[n:n + len(K)], "chp_cost-"+j] = [chp_cost[t].x for t in K]


def es_pv_results(self,K,pv_prod,p_pv_use,p_pv_sold,n=0):
    for j in list(self.es.pv_plants.keys()):
        if n==0:
            #Only for the parameters, as they don't change after every MPC Loop
            self.es.pv_plants[j]["res_df"]["pv_prod-" + j] = pv_prod
        self.es.pv_plants[j]["res_df"].loc[self.es.pv_plants[j]["res_df"].index[n:n + len(K)], "p_pv_use-"+j] = [p_pv_use[t].x for t in K]
        self.es.pv_plants[j]["res_df"].loc[self.es.pv_plants[j]["res_df"].index[n:n + len(K)], "p_pv_sold-"+j] = [p_pv_sold[t].x for t in K]


def hvac_opt_results(self,K,Qh_hvac,Qc_hvac,T_amb,T_r,n=0):
    if n==0:
        self.tbs.hvac.res_df["T_amb"] = T_amb
    self.tbs.hvac.res_df.loc[self.tbs.hvac.res_df.index[n:n + len(K)], "Qh_hvac"] = [Qh_hvac[t].x for t in K]
    self.tbs.hvac.res_df.loc[self.tbs.hvac.res_df.index[n:n + len(K)], "Qc_hvac"] = [Qc_hvac[t].x for t in K]
    self.tbs.hvac.res_df.loc[self.tbs.hvac.res_df.index[n:n + len(K)], "T_r"] = [T_r[t].x for t in K]


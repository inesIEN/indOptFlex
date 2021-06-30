from _collections import defaultdict
from gurobipy import  GRB,quicksum
import pandas as pd
#%% Prepare the dataset
def mfs_opt_formulation(self,mdl,T):
    #Get the lists of machines and resources from KMU Manufacturing systems
    machines = list(self.mfs.machines.keys())
    storages = list(self.mfs.storage_facilities.keys())
    products = list(self.mfs.products.keys())
    raw_materials = list(self.mfs.raw_materials.keys())
    resources = storages + raw_materials + products
    materials = raw_materials + products
    tasks = list(self.mfs.tasks.keys())

    # Prepare the task-resource relationship parameters dict
    mue_i_r = defaultdict(lambda: defaultdict)

    for i in tasks:
            mue_i_r[i] = {}

    #Convert processing rate (units/hr) to time step
    scaler = (60/self.mfs.sim_time_step)

    #Create the resource interaction dict based on the processing_rate
    for i in tasks:
        for m in materials:
            #If material is being consumed
            if m in (self.mfs.tasks[i].task_resource_consumption_dict):
                mue_i_r[i].update({m: (-self.mfs.machines[[self.mfs.tasks[i].machine][0]]["processing_rate"])/scaler})

            elif m in (self.mfs.tasks[i].task_resource_production_dict):
                mue_i_r[i].update({m: (self.mfs.machines[[self.mfs.tasks[i].machine][0]]["processing_rate"]) / scaler})
                #Find the corresponding stroage id for the material and also mue_i_r for the storage

            else:
                mue_i_r[i].update({m: 0})

            # for s in storages:
            #     if m in self.mfs.storage_facilities[s]["materials"]:
            #         mue_i_r[i].update((s:))
            #
            #
            #
            #     for s in self.mfs.storage_facilities:
            #      if m in self.mfs.storage_facilities[s]["materials"]:
            #         mue_i_r[i].update({s: (-self.mfs.machines[[self.mfs.tasks[i].machine][0]]["processing_rate"])/scaler})
            #      else:
            #          mue_i_r[i].update({s: (-self.mfs.machines[[self.mfs.tasks[i].machine][0]]["processing_rate"])/scaler})
            #
            # elif m in (self.mfs.tasks[i].task_resource_production_dict):
            #     mue_i_r[i].update({m: (self.mfs.machines[[self.mfs.tasks[i].machine][0]]["processing_rate"])/scaler})
            #
            #     # Find the corresponding stroage id for the material and also mue_i_r for the storage
            #     for s in self.mfs.storage_facilities:
            #         if m in self.mfs.storage_facilities[s]["materials"]:
            #             mue_i_r[i].update(
            #                 {s: (self.mfs.machines[[self.mfs.tasks[i].machine][0]]["processing_rate"]) / scaler})


        for m in machines:
            if m == self.mfs.tasks[i].machine:
                mue_i_r[i].update({m: 1})
            else:
                mue_i_r[i].update({m: 0})


    #Define the variables
    # All the resource except the machines
    R_r_t = mdl.addVars([(r, t) for r in materials  for t in T], lb=0, vtype=GRB.CONTINUOUS)

    R_s_t = mdl.addVars([(s, t) for s in storages  for t in T], vtype=GRB.CONTINUOUS)

    # Binary variable about the task ON/off time
    N_i_t = mdl.addVars([(i, t) for i in tasks for t in T], vtype=GRB.BINARY)

    #Binary variable for switching of the task (1 when start and ends, 0 otherwise)
    S_i_t = mdl.addVars([(i, t) for i in tasks for t in T], vtype=GRB.INTEGER, lb=0)

    # Binary variable showing when the task start at time "t"
    Y_i_t = mdl.addVars([(i, t) for i in tasks for t in T], vtype=GRB.BINARY)

    #Binary variable showing when the task ends
    R_i_t = mdl.addVars([(i, t) for i in tasks for t in T], vtype=GRB.BINARY)

    # Varibale for no. of tasks being executed in each time step
    count_i_t = mdl.addVars(len(T), vtype=GRB.CONTINUOUS,lb=0,ub=100)

    # variable for task being ON or OFF - Only for the flexible tasks
    # z_on = mdl.addVars([(i, t) for i in (tasks) if self.mfs.tasks[i].task_type != "NS"
    #                     for t in T], vtype=GRB.BINARY)
    # z_off = mdl.addVars([(i, t) for i in (tasks) if self.mfs.tasks[i].task_type != "NS"
    #                      for t in T], vtype=GRB.BINARY)

    z_on = mdl.addVars([(i, t) for i in tasks for t in T] , vtype=GRB.BINARY)
    z_off = mdl.addVars([(i, t) for i in tasks for t in T] , vtype=GRB.BINARY)

    # power of the machine at time step "t" - Only for MACHINES
    p_r_t = mdl.addVars([(m, t) for m in (machines) for t in T], vtype=GRB.CONTINUOUS,lb=0, name='p_r_t')


    #total power of mfs machines
    p_mfs_t = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="p_mfs_t",ub=1000,lb=0)

    #%% Add the model constraints

    # 1. Resources
    #1.1 Initialization of the materials and evolution of the materials
    mdl.addConstrs(R_r_t[r, t] == self.mfs.Ro[r] for r in (materials) for t in T if t == 0)
    mdl.addConstrs(R_s_t[s, t] == self.mfs.Ro[s] for s in (storages) for t in T if t == 0)

    mdl.addConstrs(R_r_t[r, t] == R_r_t[r, t - 1] + (quicksum(mue_i_r[i][r] * N_i_t[i, t] for i in tasks))
                   for r in materials for t in T if t > 0)


    #1.2 Final Product Quantity must be produced
    for p in products:
        if self.mfs.products[p]["prod_type"] == "final":
            mdl.addConstrs(R_r_t[p,t] >= self.mfs.products[p]["required_quanitity"] for t in T if t==T[-1])

    #1.3. Minimum and maximum products and storages quantities
    mdl.addConstrs(
        R_s_t[s, t] == quicksum(R_r_t[r, t] for r in self.mfs.storage_facilities[s]["materials"]) for s in storages
        for t in T if t > 0)
    mdl.addConstrs(R_s_t[r, t] <= self.mfs.storage_facilities[r]["max"] for r in storages for t in T if t > 0)
    mdl.addConstrs(R_s_t[r, t] >= self.mfs.storage_facilities[r]["min"] for r in storages for t in T if t > 0)



    #%%2. Tasks

    #2.1 Tasks shall not proceed in first and last steps
    mdl.addConstrs(N_i_t[i, t] == 0 for i in tasks for t in T if t == 0 if self.mfs.tasks[i].task_type == "FLEX" )
    mdl.addConstrs(N_i_t[i, t] == 0 for i in tasks for t in T if t == T[-1] if self.mfs.tasks[i].task_type == "FLEX" )

    #2.2 Constant Tasks shall work continuously
    mdl.addConstrs(N_i_t[i, t] == 1 for i in tasks for t in T if self.mfs.tasks[i].task_type == "NS")


    #2.2 Calculate the switching varibale, whenever task starts and ends
    mdl.addConstrs(N_i_t[i,t-1]-N_i_t[i,t]-S_i_t[i,t] <= 0 for i in tasks for t in T if t>0)
    mdl.addConstrs(-N_i_t[i,t - 1] + N_i_t[i,t] - S_i_t[i, t] <= 0 for i in tasks for t in T if t > 0)


    #2.4 Calculate corresponding ON/OFF variables

    mdl.addConstrs(N_i_t[i, t] - N_i_t[i, t - 1] == z_on[i, t] - z_off[i, t] for i in tasks for t in T if t>0)

    #2.5 Ensure that both swithch on and switch off doesn't happen at the same time
    mdl.addConstrs(z_on[i, t] + z_off[i, t] <= 1 for i in tasks for t in T)

    #2.6 MO, MR and No. of Switching constraints
    for i in tasks:
        if self.mfs.tasks[i].task_type == "FLEX":
            MO = (list(range(1,int(self.mfs.tasks[i].regeneration_duration/self.mfs.sim_time_step))))
            MR = (list(range(1,int(self.mfs.tasks[i].active_duration/self.mfs.sim_time_step))))
            mdl.addConstrs(quicksum(z_on[i,t-j] for j in MR if t-j>0 ) <= 1-z_off[i,t] for t in T)
            mdl.addConstrs(quicksum(z_off[i, t - j] for j in MO if t - j > 0 ) <= 1 - z_on[i, t] for t in T)
            mdl.addConstr(quicksum(z_on[i,t] for t in T) <= int(self.mfs.tasks[i].usage_number))
            mdl.addConstr(quicksum(z_off[i,t] for t in T) <= int(self.mfs.tasks[i].usage_number))
            #mdl.addConstr(quicksum(S_i_t[i,t] for t in T) <=5)

        if self.mfs.tasks[i].task_type == "NON-FLEX":
            mdl.addConstr(quicksum(z_on[i, t] for t in T if t > 0) <= 1)
            mdl.addConstr(quicksum(z_off[i, t] for t in T if t > 0) <= 1)

    # #2.3 Minimum Run time of the tasks
    # for i in tasks:
    #     if self.mfs.tasks[i].task_type == "FLEX":
    #         MR = list(range(0,int(self.mfs.tasks[i].active_duration/self.mfs.sim_time_step)))
    #         for t in T:
    #             mdl.addConstr(quicksum(S_i_t[i, t - j] for j in MR if t - j > 0) <= 1)
    #         mdl.addConstr(quicksum(S_i_t[i,t] for t in T ) <=int(self.mfs.tasks[i].usage_number))


    # #2.6 No. of interuptions on the tasks shall be limited by usage number (only for flexibile tasks)
    # mdl.addConstrs(quicksum(z_on[i, t] for t in T if t > 0) <= int(self.mfs.tasks[i].usage_number)
    #                for i in tasks if self.mfs.tasks[i].task_type == "FLEX")

    # #2.6.1 No. of interuption on the tasks of type "NON-FLEX" is only 1
    # mdl.addConstrs(quicksum(z_on[i, t] for t in T if t > 0) <= 1
    #                for i in tasks if self.mfs.tasks[i].task_type == "NON-FLEX")
    # mdl.addConstrs(quicksum(z_off[i, t] for t in T if t > 0) <= 1
    #                for i in tasks if self.mfs.tasks[i].task_type == "NON-FLEX")

    # #2.7 Invalidity interval - All the tasks should not proceed in the invalidity interval
    # mdl.addConstrs(N_i_t[i, t] == 0 for i in tasks for t in T if t in self.mfs.tasks[i].invalidity_interval
    #                if self.mfs.tasks[i].task_type == "FLEX")

    #2.8 Interlocking between tasks that use similar machines
    for m in machines:
        self.mfs.machines[m]["tasks"] = []
        for  i in tasks:
            if self.mfs.tasks[i].machine == m:
                self.mfs.machines[m]["tasks"].append(i)
        mdl.addConstrs(quicksum(N_i_t[i,t] for  i in self.mfs.machines[m]["tasks"]) <=1 for t in T)



    #mdl.addConstrs(N_i_t["CLS-2",t]+N_i_t["CRS",t] <=1 for t in T if t>0)

    #2.8. At any time more than 4 tasks can not be operated parellely:
    mdl.addConstrs(count_i_t[t] == quicksum(N_i_t[i, t] for i in tasks) for t in T if t > 0)
    #mdl.addConstrs(count_i_t[t] <= 3 for t in T if t > 0)


    #%% 3. Power (KW) of the Machines/MFS Network
    #3.1. calculate the p_r_t "only for machines"
    mdl.addConstrs(p_r_t[m, t] == quicksum(mue_i_r[i][m] * self.mfs.machines[m]["p_rated"] * N_i_t[i, t]
                                           for i in tasks) for m in machines for t in T)


    #3.2. Calculation of power profile of whole RTN network in SME
    mdl.addConstrs(p_mfs_t[t] == quicksum(p_r_t[m, t] for m in machines) for t in T)

    #3.3 Limit the p_mfs_t to some value
    #mdl.addConstrs(p_mfs_t[t] <= 150 for t in T)

    # %%Calculate the TBS profiles based on the running tasks:
    air_demand_r_t = mdl.addVars([(m, t) for m in (machines) for t in T], vtype=GRB.CONTINUOUS, name='air_r_t')
    air_demand_t = mdl.addVars(len(T), vtype=GRB.CONTINUOUS)
    q_cool_demand_t = mdl.addVars(len(T), vtype=GRB.CONTINUOUS)
    q_heat_demand_t = mdl.addVars(len(T), vtype=GRB.CONTINUOUS)

    # calculate these demand profile using constraints

    for t in T:
        mdl.addConstr(air_demand_t[t] == (quicksum(mue_i_r[i][m] * self.mfs.machines[m]["air_demand"] * N_i_t[i, t]
                                                        for i in tasks for m in machines)))

    mdl.addConstrs(q_cool_demand_t[t] == quicksum(quicksum(mue_i_r[i][m] * self.mfs.machines[m]["cooling_demand"] * N_i_t[i, t]
                                                for i in tasks) for m in machines) for t in T)

    mdl.addConstrs(
        q_heat_demand_t[t] == quicksum(quicksum(mue_i_r[i][m] * self.mfs.machines[m]["heating_demand"] * N_i_t[i, t]
                                                for i in tasks) for m in machines) for t in T)

    return p_mfs_t,air_demand_t,q_cool_demand_t,q_heat_demand_t,tasks,machines,products,storages,p_r_t,N_i_t,Y_i_t,R_i_t,S_i_t,R_r_t,R_s_t

def mfs_opt_results(self,K,p_mfs_t,p_r_t,R_r_t,N_i_t,R_s_t):
    #%%Get the requried data
    machines = list(self.mfs.machines.keys())
    storages = list(self.mfs.storage_facilities.keys())
    products = list(self.mfs.products.keys())
    raw_materials = list(self.mfs.raw_materials.keys())
    resources = storages + raw_materials + products
    tasks = list(self.mfs.tasks.keys())

    #%%Append Total and individual load profiles of the compressors
    self.mfs.res["Load_profiles"] = pd.DataFrame(columns=["KW_Total"])
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

import random
from collections import defaultdict
import numpy as np
import pandas as pd
import pylab as pl
from gurobipy import Model, GRB, quicksum


class stee_plan_opt_CS():

    def __init__(self,id,desc):
        self.id = id
        self.desc = desc
        self.day_ahead_prices = None



    def get_day_ahead_prices(self,file_path="external_model_inputs/market_price.xlsx"):

        prices = pd.read_excel(file_path)
        prices = (prices["price"]).values.tolist()
        self.day_ahead_prices  = prices


    def mfs_steel_task_optimization(self,show_figure=True):

        #%%Create the optimization model framework
        mdl = Model('flex_start')
        mdl.__len__ = 1

        # Number of time steps
        t_hr = [t for t in self.hr]  # time step in hr
        T = self.t  # time step in 0,1,2,3,..

        #Get the tasks
        tasks = list(self.mfs.tasks_new.keys())

        #Get the list of storages
        storages = list(self.mfs.storage_new.keys())

        #Get the list of storage
        storages_rm = [i for i in list(self.mfs.storage_new.keys()) if self.mfs.storage_new[i].type=="RM" ]
        storages_int = [i for i in list(self.mfs.storage_new.keys()) if self.mfs.storage_new[i].type == "INT"]
        storages_final = [i for i in list(self.mfs.storage_new.keys()) if self.mfs.storage_new[i].type == "FINAL"]


        #%% Declare the model variables
        #Task Operation Binary variable
        N_i_t = mdl.addVars([(i, t) for i in tasks for t in T], vtype=GRB.BINARY)

        #Storage Variable
        S_s_t = mdl.addVars([(s,t) for s in storages for t in T], vtype=GRB.CONTINUOUS,lb=0)
        P_s_i_t = mdl.addVars([(s,i,t) for s in storages for i in tasks for t in T],vtype=GRB.CONTINUOUS)
        C_s_i_t = mdl.addVars([(s, i, t) for s in storages for i in tasks for t in T], vtype=GRB.CONTINUOUS)

        #Electricity variables
        p_i_t = mdl.addVars([(i,t) for i in tasks for t in T], vtype=GRB.CONTINUOUS)
        p_t = mdl.addVars(len(T), vtype=GRB.CONTINUOUS, name="p_t")
        #%%Add the constraints

        #Storage constraints

        # Production and consumption rate of the task
        # for s in tasks:
        #     #Get the
        #     s_prod = (self.mfs.tasks_new[i].storage_prod)
        #     s_cons = (self.mfs.tasks_new[i].storage_cons)
        #
        #     mdl.addConstrs(P_s_i_t[s_prod, i, t] == N_i_t[i, t] * self.mfs.tasks_new[i].processing_rate for t in T)
        #     mdl.addConstrs(C_s_i_t[s_cons, i, t] == N_i_t[i, t] * self.mfs.tasks_new[i].processing_rate for t in T)

        for s in storages:
            #get the tasks:
            if self.mfs.storage_new[s].type=="RM":
                #get the task that consumes this storage
                i_cons = self.mfs.storage_new[s].task_cons
                mdl.addConstrs(P_s_i_t[s,i_cons,t] == 0 for t in T)
                mdl.addConstrs(C_s_i_t[s,i_cons,t] == N_i_t[i_cons,t] * self.mfs.tasks_new[i_cons].processing_rate for t in T)

            elif self.mfs.storage_new[s].type=="INT":
                i_cons = self.mfs.storage_new[s].task_cons
                i_prod = self.mfs.storage_new[s].task_prod

                mdl.addConstrs(C_s_i_t[s, i_cons, t] == N_i_t[i_cons, t] * self.mfs.tasks_new[i_cons].processing_rate for t in T)
                mdl.addConstrs(P_s_i_t[s, i_prod, t] == N_i_t[i_prod, t] * self.mfs.tasks_new[i_prod].processing_rate for t in T)

            elif self.mfs.storage_new[s].type=="FINAL":
                i_prod = self.mfs.storage_new[s].task_prod
                mdl.addConstrs(C_s_i_t[s, i_prod, t] == 0 for t in T)
                mdl.addConstrs(P_s_i_t[s, i_prod, t] == N_i_t[i_prod, t] * self.mfs.tasks_new[i_prod].processing_rate for t in T)

        #Material storage constraints
        mdl.addConstrs(S_s_t[s,t] == (self.mfs.storage_new[s].capacity_ini)  for s in storages for t in T if t==0)

        for s in storages:
            for i in tasks:
                mdl.addConstrs(S_s_t[s,t]==S_s_t[s,t-1] + P_s_i_t[s,i,t] - C_s_i_t[s,i,t] for t in T if t>0)

        # mdl.addConstrs(S_s_t[s,t] == S_s_t[s,t-1] + quicksum(P_s_i_t[s,i,t] for s in storages ) -
        #                                             quicksum(C_s_i_t[s,i,t] for s in storages)
        #                                                         for s in storages
        #                                                         for i in tasks
        #                                                         for t in T if t>0)

        mdl.addConstrs(S_s_t[s,t]>=0 for s in storages for t in T)
        mdl.addConstrs(S_s_t[s, t] <= self.mfs.storage_new[s].capacity_max for s in storages for t in T)


        #Power Requirement constraints
        mdl.addConstrs(p_i_t[i,t] == N_i_t[i,t]*self.mfs.tasks_new[i].power_consumption for i in tasks for t in T)
        mdl.addConstrs(p_t[t] == quicksum(p_i_t[i, t] for i in tasks) for t in T )

        #Final product constraint, show that this product must be manufactured
        final_tasks = ["Deh"]
        for i in final_tasks:
            mdl.addConstr(quicksum(N_i_t[i,t]*self.mfs.tasks_new[i].processing_rate for t in T)>=90)


        #%%# %% Objective function
        mdl.setObjective(quicksum(p_t[t] for t in T))

        # Senseofthemode
        mdl.modelSense = GRB.MINIMIZE
        mdl.setParam("TimeLimit", 10.0)
        mdl.optimize()

        task_schedule = {}
        for i in tasks:
            task_schedule[i] = [N_i_t[i, t].x for t in T]

        self.mfs.opt_task_schedule = task_schedule



        def print_opt_results():
            import pandas as pd
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            columns_tasks = [x for x in tasks]
            columns_storage = [x for x in storages]
            res_tasks = pd.DataFrame(columns=columns_tasks)
            res_storage = pd.DataFrame(columns=columns_storage)

            res_tasks["HR"] = t_hr
            res_storage["HR"] = t_hr

            for i in tasks:
                res_tasks[i] = [N_i_t[i, t].x for t in T]

            for s in storages:
                res_storage[s] = [S_s_t[s,t].x for t in T]

            cols_task = res_tasks.columns.to_list()
            cols_task.remove("HR")

            cols_storage = res_storage.columns.to_list()
            cols_storage.remove("HR")

            cols_task_rev = cols_task[::-1]
            cols_storage_rev = cols_storage[::-1]

            fig1 = make_subplots(rows=len(cols_task), shared_xaxes=True, vertical_spacing=0.02, row_titles=cols_task_rev,
                                y_title="Values")

            for i in range(len(cols_task)):
                fig1.add_trace(go.Scatter(x=res_tasks.HR,
                                         y=res_tasks[cols_task[i]]),
                              row=len(cols_task) - i, col=1,
                              )
            fig1['layout'].update(height=1000)
            fig1.show()

            fig2 = make_subplots(rows=len(cols_storage), shared_xaxes=True, vertical_spacing=0.02,
                                 row_titles=cols_storage_rev,
                                 y_title="Values")

            for i in range(len(cols_storage)):
                fig2.add_trace(go.Scatter(x=res_storage.HR,
                                          y=res_storage[cols_storage[i]]),
                               row=len(cols_storage) - i, col=1,
                               )
            fig2['layout'].update(height=1000)
            fig2.show()
        if show_figure == True :
            print_opt_results()

































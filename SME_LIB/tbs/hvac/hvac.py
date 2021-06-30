from _collections import defaultdict
import pandas as pd

class HVAC():
    def __init__(self, id, desc,sim_time_step,space_area,heat_loss_coeff,thermal_capacity,
                 t_set_min,t_set,t_set_max,N_sim,**kwargs):
        self.id = id
        self.desc = desc
        self.sim_time_step = sim_time_step
        self.heat_loss_coeff = heat_loss_coeff
        self.space_area = space_area
        self.thermal_capacity = thermal_capacity
        self.t_set = t_set
        self.t_set_min = t_set_min
        self.t_set_max = t_set_max
        self.t_amb = None
        self.N_sim = N_sim


        #To store the results in DF
        self.res_df = pd.DataFrame(index=N_sim,columns=[])

        self.walls = defaultdict(lambda: defaultdict)
        self.windows = defaultdict(lambda: defaultdict)
        self.roof = defaultdict(lambda: defaultdict)
        self.floor = (lambda: defaultdict)

    def load_ambient_temperatures(self,file_path):
        t_amb = pd.read_excel(file_path,sheet_name="Sheet1")
        self.t_amb = list(t_amb["T_amb"])

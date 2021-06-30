import pandas as pd


def _create_time_grid(self,sim_time_step,sim_horizon):
    T = sim_horizon  # Total Number of Hours in a day
    #Time steps for minutes
    N_sim = T * 60 / sim_time_step
    #Create the time range
    hour_range = (pd.date_range('00:00', '23:59',
                                freq=str(self.sim_time_step) + 'min'))
    self.hr = hour_range
    hour_range = hour_range.to_list()
    self.N_sim = list(range(int(N_sim)))
    # time_grid = list(zip(hour_range, self.t))
    #
    # self.hr = [hr[0] for hr in time_grid]


    return
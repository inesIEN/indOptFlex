"""
This module calculate the the running parameters the tasks, which are required for the executions
"""

class Calculate_tasks():



    def calculte_task_durations(self):

        for task in self.tasks.keys():

            #Get the associated product
            product = self.tasks[task].product

            #Get the associated machine
            machine = self.tasks[task].machine
            #production rate of the machine
            prod_rate = self.machines[machine]["prod_rate"]

            #Get the required quanitity of the product
            req_quanitity = self.products[product]["required_quanitity"]

            #Calculate the task duration time
            duration = (req_quanitity / prod_rate)

            #Attach the duration inside the task data dict

            self.tasks[task].duration = duration * 60

    # def calculate_task_durataion_new(self):
    #     for


    def calculate_task_validity_interval(self):

        for task in self.tasks.keys():


            start_index = self.hr.index(self.tasks[task].validity.split(',')[0])
            end_index   = self.hr.index(self.tasks[task].validity.split(',')[1])
            validity_interval = list(range(start_index, end_index + 1))
            invalidity_interval = [item for item in self.t if not item in validity_interval]
            self.tasks[task].validity_interval = validity_interval
            self.tasks[task].invalidity_interval = invalidity_interval

            # elif self.tasks[task].task_type == "NON-FLEX":
            #     start_index = self.hr.index(self.tasks[task].start_time)
            #     end_index = start_index + int(self.tasks[task].duration/self.sim_time_step)
            #     validity_interval = list(range(start_index, end_index + 1))
            #     invalidity_interval = [item for item in self.t if not item in validity_interval]
            #     self.tasks[task].validity_interval = validity_interval
            #     self.tasks[task].invalidity_interval = invalidity_interval













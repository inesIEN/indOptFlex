from collections import defaultdict
import xlrd
import os
import pandas as pd

from mfs.calculate_tasks import Calculate_tasks



class MFS(Calculate_tasks):

    def __init__(self, id, desc, N_sim, hr, sim_time_step):
        self.id = id
        self.desc = desc
        self.machines = defaultdict(lambda: defaultdict)
        self.products = defaultdict(lambda: defaultdict)
        self.raw_materials = defaultdict(lambda : defaultdict)
        self.storage_facilities = defaultdict(lambda : defaultdict)
        self.tasks = defaultdict(lambda : defaultdict)
        self.tasks_new = defaultdict(lambda: defaultdict)
        self.storage_new = defaultdict(lambda: defaultdict)
        self.N_sim = N_sim
        self.hr = hr
        self.sim_time_step = sim_time_step
        self.Ro = defaultdict(lambda  : defaultdict)
        self.mue_i_r = defaultdict(lambda  : defaultdict)

        #Load Profile Dataframe
        self.ref_load_profile_df = None
        self.products_state_df = None
        self.storages_state_df = None

        #Append the results optimization in dictionary
        self.res = defaultdict(lambda: defaultdict)


    #Functions that uses the class of the MFS objects and save the recorded data intro "Default Dicts"



    def get_load_profile_from_forecast_data_pickle(self, file_path, measurement_names):
        forecast = pd.read_pickle(file_path)
        self.ref_load_profile_df = forecast[measurement_names]/1000

    def create_machine(self, id, name, p_kw, processing_rate, air_demand, cooling_demand,heating_demand):
        self.machines[id] = (
            self.Machine(id=id, name=name, p_rated=p_kw, processing_rate=processing_rate,
                         air_demand=air_demand, cooling_demand=cooling_demand,heating_demand=heating_demand)).__dict__


    def create_product(self, id, initial_quantitity, required_quanitity, prod_deadline,prod_type,storage_id):
        self.products[id] = (self.Product(id=id, initial_quantitity=initial_quantitity,
                                          required_quanitity=required_quanitity, prod_deadline=prod_deadline,
                                          prod_type=prod_type,storage_id=storage_id)).__dict__
        # Append the initial quantities
        self.Ro[id] = initial_quantitity

    def create_raw_material(self, id, initial_quantity, minimum_quantity, storage_id):
        self.raw_materials[id] = (self.Raw_material(id=id, initial_quantity=initial_quantity,
                                                    minimum_quantity=minimum_quantity,
                                                    storage_id=storage_id)).__dict__

        #Append the initial quantities
        self.Ro[id] = initial_quantity


    def create_storage_facility(self,id,capacity,materials,initial_quantity,min,max):
        self.storage_facilities[id] = (self.Storage_facility(id=id,capacity=capacity,materials=materials,
                                                             initial_quantity=initial_quantity,
                                                             min=min,max=max)).__dict__
        # Append the initial quantities
        self.Ro[id] = initial_quantity


    def create_tasks(self,id,name,product, machine, task_resource_consumption_dict,
                     task_resource_production_dict,task_type,minimum_run_time=None,minimum_off_time=None,
                     usage_number=None, validity=None):

        self.tasks[id] = (self.Task(id=id,name=name,
                         product=product, machine = machine, task_resource_consumption_dict=task_resource_consumption_dict,
                         task_resource_production_dict=task_resource_production_dict,
                         task_type=task_type, minimum_run_time = minimum_run_time,
                         minimum_off_time = minimum_off_time,
                         usage_number=usage_number, validity=validity))




    def create_mfs_from_file(self,file_path):
        #Load the workbook
        path = file_path
        # workbook = xlrd.open_workbook(file_path)


        #Create Machines
        df_machines = pd.read_excel(file_path,sheet_name="Machines")

        for i in range(len(df_machines)):
            self.create_machine(id=df_machines["ID"][i],name=df_machines["Name"][i],
                                    p_kw=df_machines["p_kw"][i], processing_rate=df_machines["processing_rate"][i],
                                    air_demand=df_machines["air_demand"][i],
                                    cooling_demand=df_machines["cooling_demand"][i],
                                    heating_demand = df_machines["heating_demand"][i])

        #Create storages
        df_storages = pd.read_excel(path,sheet_name="Storages")
        for i in range(len(df_storages)):
            self.create_storage_facility(id=df_storages["ID"][i],capacity=df_storages["Capacity"][i],
                                         materials=None,
                                         initial_quantity=df_storages["initial_quantitiy"][i],
                                         min=df_storages["min"][i], max=df_storages["max"][i])

        #Create Products
        df_products = pd.read_excel(path,sheet_name="Products")
        for i in range(len(df_products)):
            self.create_product(id=df_products["ID"][i],initial_quantitity=df_products["initial_quantity"][i],
                                required_quanitity=df_products["required_quantity"][i],
                                prod_deadline=df_products["prod_deadline"][i],prod_type=df_products["prod_type"][i],
                                storage_id=df_products["storage_id"][i])

        #Create Raw Materials
        df_raw_materials =pd.read_excel(path, sheet_name="Raw_materials")
        for i in range(len(df_raw_materials)):
            self.create_raw_material(id=df_raw_materials["ID"][i],initial_quantity=df_raw_materials["initial_quantity"][i]
                                     ,minimum_quantity=df_raw_materials["minimum_quantity"][i]
                                     ,storage_id=df_raw_materials["storage_id"][i])

        # Create Tasks
        df_tasks = pd.read_excel(path, sheet_name="Tasks")
        for i in range(len(df_tasks)):
            self.create_tasks(id=df_tasks["ID"][i],name=df_tasks["name"][i],
                              product=df_tasks["product"][i],machine=df_tasks["machine"][i],
                              task_resource_consumption_dict=df_tasks["cons_dict"][i].split(","),
                              task_resource_production_dict=df_tasks["prod_dict"][i].split(","),
                              task_type=df_tasks["task_type"][i],minimum_run_time=df_tasks["minimum_run_time"][i],
                              minimum_off_time=df_tasks["minimum_off_time"][i],
                              usage_number=df_tasks["usage_number"][i],validity=df_tasks["validity"][i])


        #Append the Materials inside the storage based on the given inputs
        storages = list(self.storage_facilities.keys())

        for s in storages:
            materials = []
            #Products
            for p in list(self.products.keys()):
                if s in self.products[p]["storage_id"]:
                    materials.append(p)

            for rm in list(self.raw_materials.keys()):
                if s in self.raw_materials[rm]["storage_id"]:
                    materials.append(rm)

            #Append the material dict now in storage facility

            #Raw Materials
            self.storage_facilities[s]["materials"] = materials



    #Declare the Classes of the resources inside the Manufacturing System

    class Machine:
        def __init__(self, id, name, p_rated, processing_rate, air_demand, cooling_demand,heating_demand):
            self.id = id
            self.name = name
            self.p_rated = p_rated
            self.processing_rate = processing_rate
            self.air_demand = air_demand
            self.cooling_demand = cooling_demand
            self.heating_demand = heating_demand
            self.tasks = defaultdict

    class Product:
        def __init__(self, id, initial_quantitity, required_quanitity, prod_deadline, prod_type,storage_id):
            self.id = id
            self.initial_quantitity = initial_quantitity
            self.required_quanitity = required_quanitity
            self.prod_deadline = prod_deadline
            self.prod_type = prod_type
            self.storage_id = storage_id

    class Raw_material:
        def __init__(self,id,initial_quantity, minimum_quantity, storage_id):
            self.id = id
            self.initial_quantitiy = initial_quantity
            self.min_quantity = minimum_quantity
            self.storage_id = storage_id

    class Storage_facility:
        def __init__(self,id,capacity,materials,initial_quantity,min,max):
            self.id = id
            self.capacity = capacity
            self.materials = materials
            self.min = min
            self.max = max
            self.initial_quantity = initial_quantity

    #Declare the classes for the tasks
    class Task:
        def __init__(self,id,name,product,machine, task_resource_consumption_dict,
                     task_resource_production_dict, task_type,
                     minimum_run_time=None,minimum_off_time=None,usage_number=None,
                     duration=None,validity=None,validity_interval=None,invalidity_interval=None):
            self.id = id
            self.name = name
            self.product = product
            self.machine = machine
            self.task_resource_consumption_dict = task_resource_consumption_dict
            self.task_resource_production_dict = task_resource_production_dict
            self.task_type = task_type
            self.minimum_run_time = minimum_run_time
            self.minimum_off_time = minimum_off_time
            self.usage_number=usage_number
            self.duration = duration
            self.validity = validity
            self.validity_interval = validity_interval
            self.invalidity_interval = invalidity_interval




import pandas as pd
from collections import defaultdict

from cs.cs import CS
#from cs.cs import CS
from es.es import ES
from mfs.mfs import MFS
from market.market import Market
from tbs.CAS.cas import CAS
from tbs.tbs import TBS
from utilities.plotter_function import Plots
from utilities.time_grid import _create_time_grid


class SME(CS,CAS,ES,Plots,Market):

    """
    This represent the main SME class, which when called creates an empty SME object. This represents the
    top layer of the SME and it contains generic attributes for identifcation and also attributes as containers
    for the sub-objects

    Attributes to be provided
    ------------------------
    id : str
         id of the SME
    name : str
         name of the SME
    ort : str
        place of the SME

    sim_time_step : int
        discrete time step of the simulation in minutes

    sim_horizon : int

        No. of time time steps in simulation horizon. For e.g If a simulation is needed to be performed for 24 HRS with
        a time step of 15 minutes, then sim_horizon = 24* (60/15) = 96

    Attributes embedded
    ------------------------------------
    t = list
        time grid points of simulation [0,1,2,3,.....]

    sme_load_profiles : container variable to hold any load profiles (Mostly Übergabeyähler)

    mfs : sub-class
         contains MFS object
    tbs : sub-class
        contains TBS object
    es : sub-class
        containing ES object
    market: sub-class
        contains market object
    """

    def __init__(self,id,name,ort,sim_time_step=15,sim_horizon=24):
        """
        Constructs all the necessary attributes for the SME object
        :param id:
        :param name:
        :param ort:
        :param sim_time_step:
        :param sim_horizon:
        """
        self.id = id
        self.name = name
        self.ort = ort

        #Simulation time step and horizon paeameters for the SME
        self.sim_time_step = sim_time_step #minutes
        self.sim_horizon = sim_horizon  # hours

        self.N_sim = None
        self.hr = None

        self.sme_load_profiles = None

        #Create the time grid
        _create_time_grid(self,self.sim_time_step,self.sim_horizon)

        #Define the technical units
        self.mfs = None
        self.tbs = None
        self.es  =  None

        #Define the control systems of SME
        self.cs = None

        #Define the external market with which SME is connected
        self.markets = defaultdict(lambda: defaultdict)

    #Get load profile from local folders or saved excel data
    def get_load_profiles(self,data_folder_path,date):
        """
        This function helps to load the data from the folder containing CSV Files
        :param data_folder_path: str
        :param date:  "DD.MM.YY"
        :return: Appends the load profiles in sme_load_profiles
        """
        self.sme_load_profiles = get_load_profiles(data_folder_path,date)
        #Harcode
        total = (list(self.sme_load_profiles.sum(axis=1)))
        #self.sme_load_profiles["Total"] = total

    #Get the Load Profiles from ENIT Agent - from and to based on RESR API
    def get_load_profiles_from_enit_agent(self,URL,FROM,TO,ID_List,username,password,desc):
        """
        This function is used to load the Profiles from ENIT Agent directly

        :param URL: str - link of the ENIT Agent of the company provided from ENIT GmBH
        :param FROM: str-  Time and date from which the data needs to be retrieved, format : "2021-04-04T22:00:00"
        :param TO: str-  Time and date till which the data needs to be retrieved, format : "2021-04-05T21:45:00"
        :param ID_List: list - List of IDs of all the meters for which data needs to be retrieved
        :param username: str - username
        :param password: str - password
        :param desc: str - any text or description about the data to be retrieved
        :return: dataframe - load profile dataframe
        """
        self.sme_load_profiles = get_load_profiles_from_enit_agent(URL=URL,FROM=FROM,TO=TO,ID_list=ID_List,
                                                                   time_step=self.sim_time_step,desc=desc,
                                                                   username=username,password=password)

    #Get the Load Profiles for clustered data or typical - based on F.IPA Input
    def get_load_profile_clustered(self,cluster_id,file_path):
        """
        This function is written for the purpose of loading the profiles provided by Alejandro from IPA
        which he clustered during his work.
        :param cluster_id:  ID of the cluster
        :param file_path:  Path where the file is located
        :return:
        """
        self.sme_load_profiles = get_load_profile_from_clustered_profiles(path=file_path,cluster_id=cluster_id)

    def get_load_profile_from_forecast_data_pickle(self, file_path, measurement_names):
        """
        This function is written for the purpose of load the data directly from .pkl file as forecast provided by
        Thilo Walser from Fraunhofer IPA

        :param file_path: path of the file
        :param measurement_names: list of string - for the measurements requested
        :return:
        """
        forecast = pd.read_pickle(file_path)
        self.sme_load_profiles = forecast[measurement_names] / 1000
        self.sme_load_profiles["total"] = self.sme_load_profiles.sum(axis=1)


    #Create further objects functions
    def create_mfs(self,id,desc,sim_time_step=15):

        """
        This function is used to created and append the MFS object in SME object

        """

        mfs = MFS(id=id, desc=desc, N_sim=self.N_sim, hr=self.hr, sim_time_step=sim_time_step)
        self.mfs = mfs

    def create_tbs(self,id,desc):
        """
        This function is used to created and append the TBS object in SME object

        """
        tbs = TBS(id=id,desc=desc,sim_time_step=self.sim_time_step,N_sim=self.N_sim)
        self.tbs = tbs

    def create_es(self, id, desc):
        """
             This function is used to create and append the ES object in SME object

        """
        es = ES(id=id, desc=desc,sim_time_step=self.sim_time_step,N_sim=self.N_sim)
        self.es = es


    def create_market(self,id,type,desc,price_buy,price_sell,electiricity_tax,demand_rate,energy_rate,eeg_surcharge,kwk_surcharge,
                     offshore_surcharge,abschalt_surcharge,tax,pltw,retailer_cut_da):
        """
        This function is used to create a market object that is appended to SME object

        :param id: str - id
        :param type: str - Fixed, Day_ahead or Capacity
        :param desc: str - any description to be provided
        :param price_buy: float - price of buying electricity which is fixed
        :param price_sell: float - price of selling electricity
        :param electiricity_tax: - float - Electricity tax in percentage
        :param demand_rate:  - float - Anual Demand rate in Euor/KW - #todo : For Atypical Net-usage
        :param energy_rate:  - float - Energy Rate in Euro/kWH #todo: for Atypical Net-usage
        :param eeg_surcharge: - float
        :param kwk_surcharge: - float
        :param offshore_surcharge: - float
        :param abschalt_surcharge: - float
        :param tax: float - General sales (VAT ) tax in percentage
        :param pltw: - List of two numbers for e.g [24,36] - 06.00 AM - 09.00 AM represents peak-load time window
                       #todo : Shall be converted to strings in DATE TIME FORMAT
        :param retailer_cut_da: Percentage of Retailer Profits
        :return: object (Market)
        """
        self.markets[id] = (Market(id=id,type=type,desc=desc,price_buy=price_buy,price_sell=price_sell,electiricity_tax=electiricity_tax
                           ,demand_rate=demand_rate,energy_rate=energy_rate,eeg_surcharge=eeg_surcharge,
                            kwk_surcharge=kwk_surcharge,offshore_surcharge=offshore_surcharge,
                            abschalt_surcharge=abschalt_surcharge,tax=tax,pltw=pltw,retailer_cut_da=retailer_cut_da))








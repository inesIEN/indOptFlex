# Generic Industrial Enterprise Model Library - 
Developed by : Research group - Intelligent Energy Network , Institute of Energy Systems Technology, Offenburg University of Applied Sceinces. 
\
Authors and Credits: 
\
Rahul Khatri,  
Michael Schmidt,
Rainer Gasper

---------------------------------------
Required Pacakges to be installed
---------------------------------------
- pandas 
- numpy 
- gurobipy (Please also install the Gurobi Optimizer locally to your computer 
  and also the license key . 
  Details : https://www.gurobi.com/academia/academic-program-and-licenses/)


--------------------------------
Project Description
--------------------------------
This project is developed for the purpose of modelling of small and medium-sized enterprise
to analyse their active participation in energy markets with variable electricity prices.
Using the developed model, a small and medium-sized enterprise can be modelled and optimal control can be applied
with different objective functions related to electricity buying and selling costs.

Following Technical Units can be modelled :
1. Manunfacturing Systems (MFS)
    Based on the Resource (Raw Material, Storage Facilitiy, Machines, Products) and Task (Manufacturing Jobs)

2. Technical Building Services (TBS)\
    Compressed Air System (CAS) \
    Process Cooling and Heating Systems \
    HVAC 
    
3. Energy Systems (ES) \
   PV , Battery , CHP 
   
Once the SME object can be defined, different units can be written and also the corresponding data
can be loaded. 

-------------------------------------
Energy management and optimal control
-------------------------------------
The model library is also equiped with MILP and MPC based optimal control which on one hand maps
the physical boundary conditions in form of model constraints, and participation mechansim (
the way in which a SME buys electricity from market (retailer/aggregators) ) in terms of cost objective
functions). The corresponding control algorithms are written in SME_LIB/cs directory. In "cs" directory two formulation
are written, one for the MILP based problem and one for the MPC based problem.


----------------------------------------
Execution of the Files
----------------------------------------
For the modelling and execution of scripts, it is recommended to create a new directory under "modelled_smes", where 
a new SME can be created and simulation can be performed.

Example can be found in "Steel_plant" under modelled SMEs


----------------------------------------
External Data
----------------------------------------
The external data can be put in folder "external_model_inputs". 
It include weather data (radiation factor for PV, ambient temperature) and electricity price data.

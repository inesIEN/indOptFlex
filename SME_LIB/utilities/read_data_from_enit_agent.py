import pandas as pd
import requests

x = pd.read_excel()

"""
example to call import functions starting at line 87; set your username in line 87 and your password in line 88
"""



def parse_json_records_auth(url,username, password):
    """
    Parser for extracting records from a JSON file which is given by an URL
    Authentification included
    Args:
        url: URL from REST API

    Returns:
        Records as a list

    """
    r = requests.get(url=url, auth=(username, password))
    #print(r)
    data = r.json()
    #print(data)
    records = data["records"]
    time_series_name = data["timeSeriesName"]
    group = data["timeSeriesGroupName"]
    unit = data["unit"]
    #print("Records rausholen")
    #print(unit, time_series_name)
    return records, time_series_name, unit, group

def get_dataframe_from_records(records, number, time_series_name, group, unit):
    """

    Parameters
    ----------
    records
    number
    time_series_name
    group
    unit

    Returns
    -------

    """
    df_records = pd.DataFrame(records)
    time = pd.to_datetime(df_records["timestamp"], unit="ms")
    df_records["timestamp"] = time
   # df_records["timestamp"] = df_records["timestamp"].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
    #df_records["timestamp"] = df_records["timestamp"].dt.tz_localize(None)

    df = df_records.set_index("timestamp")

    df = df.rename(columns={'value':  group +'_' + time_series_name + "_Unit_" + unit})
    return df


def import_mech_tron_time(FROM,TO,ID,username, password):
    """
    import mechtron data from enit agent (for other companies: change url)
    :param FROM:
    :param TO:
    :param ID: ID of time series to be imported
    (see: \ownCloud\GaIN Filesharing\30 Arbeitspakete\AP3\Datenübergabe\DataSourceDefinition_mech-tron_20200515.xlsx)
    :param username: username for enit agent
    :param password: password for enit agent
    :return: pandas dataframe with measurements
    """
    url = "http://apgq7t5153.agent.enit.io/api/export/timeseries/"+str(ID)+"/from/"+FROM+"/to/"+TO
    records, time_series_name, unit, group = parse_json_records_auth(url, username, password)
    df = get_dataframe_from_records(records, ID, time_series_name, group, unit)
    return df

def import_mech_multiple_time(FROM, TO, ID_list, username, password):
    frames = []
    for i in ID_list:
        url = "http://apgq7t5153.agent.enit.io/api/export/timeseries/"+str(i)+"/from/"+FROM+"/to/"+TO
        print(url)
        records, time_series_name, unit, group = parse_json_records_auth(url, username, password)
        df = get_dataframe_from_records(records, i, time_series_name, group, unit)
        frames.append(df)
    df = pd.concat(frames, axis=1)
    return df

username_agent = ""         # set your username here
password_agent = ""         # set your password here
FROM = '2020-06-01T00:00:00'  # start time for import, make sure data format is "YYYY-MM-DDTHH:MM:SS"
TO = '2020-06-30T00:00:00'  # end time for import, make sure data format is "YYYY-MM-DDTHH:MM:SS"
# for ID declaration see: \ownCloud\GaIN Filesharing\30 Arbeitspakete\AP3\Datenübergabe\DataSourceDefinition_mech-tron_20200515.xlsx
ID = 1          # ID, if only one signal is imported
ID_list = [1,2,3]   # ID list, if multiple signals are imported
# function to import only one signal
dataframe = import_mech_tron_time(FROM, TO, ID=110, username="rahul.rahul@hs-offenburg.de",
                                           password="Somersett@1992")
#function to import multiple signals
dataframe_multiple_variables = import_mech_multiple_time(FROM, TO, ID_list=ID_list, username=username_agent,
                                           password=password_agent)

## to save pandas dataframe as csv see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html

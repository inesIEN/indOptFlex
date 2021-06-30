from collections import defaultdict
import pandas as pd


class Market():
    def __init__(self, id, type, desc, price_buy, price_sell, electiricity_tax, demand_rate, energy_rate, eeg_surcharge,
                 kwk_surcharge,
                 offshore_surcharge, abschalt_surcharge, tax, pltw,retailer_cut_da):
        self.id = id
        self.type = type
        self.desc = desc
        self.price_buy = price_buy
        self.price_sell = price_sell
        self.electiricity_tax = electiricity_tax
        self.demand_rate = demand_rate
        self.energy_rate = energy_rate
        self.eeg_surcharge = eeg_surcharge
        self.kwk_surcharge = kwk_surcharge
        self.offshore_surcharge = offshore_surcharge
        self.abschalt_surcharge = abschalt_surcharge
        self.tax = tax
        self.pltw = pltw
        self.retailer_cut_da = retailer_cut_da
        self.prices_variable = pd.DataFrame(columns=["Price"])
        self.prices_fixed = pd.DataFrame(columns=["Price"])

    def get_day_ahead_prices_clustered_price(self,market_id, file_path, cluster_id):
        pkl_file = pd.read_pickle(file_path)
        day_ahead_prices = pkl_file[cluster_id]["Average"]
        prices_original = day_ahead_prices

        prices = pd.DataFrame(columns=["price"])

        # Create a dataframe based on simulation time_step, For e.g 15min--->sim_time_step

        prices = pd.DataFrame(columns=["price"], index=self.sme_load_profiles.index)
        prices = prices[0:int(24 * (60 / self.sim_time_step))]

        # range_x = int((len(prices) * self.sim_time_step) / len(prices_original))

        range_x = int(len(prices) / len(prices_original))

        for p in range(len(prices_original)):
            p1 = p * range_x
            p2 = p1 + range_x

            prices["price"][p1:p2] = prices_original[p] / 1000

            # for i in range(range_x):
            #     prices["price"].iloc[p+i] = prices_original["price"][p]/100

        prices = (prices["price"]).values.tolist()

        # prices = (prices["price"]).values.tolist()
        # prices = prices*n_days
        self.markets[market_id].price_variable = prices


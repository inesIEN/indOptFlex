from matplotlib import pyplot as plt



class Plots():

    def plot_load_profile(self):
        #For Manufacturing System
        load_profile_mfs = self.mfs.load_profile_df
        cols2select = [col for col in load_profile_mfs.columns.tolist() if col not in ['total']]
        load_profile_mfs[cols2select].plot(x="Hour",kind="bar",stacked="True",
                                           title="Load Profile - Manufacturing System(MFS)")
        plt.xlabel("Time ")
        plt.ylabel("Power (KW)")


    def plot_optimized_load_profiles(self):

        fig = plt.figure()


        ax1 = fig.add_subplot(211)
        plt.plot(self.sme_load_profiles)
        plt.legend(list(self.sme_load_profiles.columns))
        ax1.set_xlabel('Time Stamp')
        ax1.set_ylabel('€/MWH')

        ax2 = fig.add_subplot(212)
        plt.plot(self.sme_load_profiles.index, self.day_ahead_prices)
        ax2.set_xlabel('Time Stamp')
        ax2.set_ylabel('€/MWH')

        plt.show()


import xarray as xr
import numpy as np
import pandas as pd
from random import shuffle
from matplotlib import pyplot as plt
from collections import defaultdict
import math
import statistics as st


class VisualStimData:
    """
    Data and methods for the visual stimulus ePhys experiment.
    The data table itself is held in self.data, an `xarray` object.
    Inputs:
        data: xr.DataArray or xr.Dataset
    Methods:
         plot_electrode: plotting the measured voltage of given subject, repetition and electrodes.
         experimenter_bias: plotting and presenting the stats of the average measurements conducted by different experimenters.
    """
    def __init__(self, data, ):
        self.data = data
        # Checking that the data is xarray as it should be
        if not isinstance(self.data, (xr.DataArray, xr.Dataset)):
            raise TypeError("The data inserted must be an xarray DataArray or Dataset")

    def plot_electrode(self, rep_number: int, rat_id: int, elec_number: tuple=(0,)):
        """
        Plots the voltage of the electrodes in "elec_number" for the rat "rat_id" in the repetition
        "rep_number". Shows a single figure with subplots.
        """
        # Get the relevant data:
        rat_data = self.data[rat_id].sel(repetition=rep_number).sel(electrode=list(elec_number))
        # Plotting the data
        fig, axes = plt.subplots(len(elec_number))
        for i in range(len(elec_number)):
            axes[i].plot(rat_data.coords['time'].values,rat_data.sel(electrode=elec_number[i]).values)
            if i == 0:
                axes[i].set(title=f'Subject: {rat_id}    Repetition index: {rep_number}\nElectrode {elec_number[i]}', ylabel='Voltage', xlabel='Time in trial (secs)')
            else:
                axes[i].set(title=f'Electrode {elec_number[i]}', ylabel='Voltage', xlabel='Time in trial (secs)')
        plt.show()

    def experimenter_bias(self):
        """ Shows the statistics of the average recording across all experimenters """
        # Initialize the dict to store the data
        averages_dict = defaultdict(list)
        # collect the data
        for subject in self.data.data_vars:
            sub_data = self.data[subject]
            averages_dict[sub_data.Experimenter].append(sub_data.values.mean())

        # Creating a data frame for the statistical data:
        df_stats = pd.DataFrame(index=averages_dict.keys(), columns=['mean', 'std', 'median'])
        df_stats.index.name = 'Experimenter'

        # Calculating and filling the stats data in the data frame
        for key in averages_dict:
            if len(averages_dict[key]) <= 1:
                raise ValueError('At least one experimenter has no more than one data point, thus statistics can not be calculated')
            df_stats['mean'][key] = st.mean(averages_dict[key])
            df_stats['std'][key] = st.stdev(averages_dict[key])
            df_stats['median'][key] = st.median(averages_dict[key])

        # ploting:
        fig, ax = plt.subplots()
        ax.bar(df_stats.index, df_stats['mean'],  yerr=df_stats['std'], capsize=4)
        ax.set(title=f'Averaged voltage measurements for each experimenter', ylabel='Voltage', xlabel='Experimenter')
        ax.plot(df_stats.index, df_stats['median'], 'o', color='red')
        ax.legend(['Median', 'Mean'], loc='upper right')
        plt.show()

        # printing the stats:
        print(f'\n\nGeneral statistics for the averaged data measured by each experimenter:\n'
              f'-----------------------------------------------------------------------\n'
              f'{df_stats}')


def mock_stim_data(n=50) -> VisualStimData:
    """ Creates a new VisualStimData instance with mock data """
    # parameters:
    n_elctrodes = 10
    n_smaples_per_electrode = 10000
    n_repetitions = 4
    trial_t = 2
    pre_t = 1
    during_t = 0.1
    post_t = 0.9

    # initiate optional IDs, genders, experimenters:
    optional_IDs = list(range(999, 100, -1))
    genders = ['male', 'female']
    experimenters = ['A', 'B', 'C', 'D']

    # handling time points:
    time_points = np.linspace(0, trial_t, num=n_smaples_per_electrode)
    # add the stimulus type (-1 is pre, 0 is during, 1 is post) accordingly:
    stim_type = np.concatenate((np.full((int(math.ceil(pre_t/trial_t*n_smaples_per_electrode))), -1), np.full((int(math.ceil(during_t/trial_t*n_smaples_per_electrode))), 0), np.full(n_smaples_per_electrode - (int(math.ceil(pre_t/trial_t*n_smaples_per_electrode))+int(math.ceil(during_t/trial_t*n_smaples_per_electrode))), 1)))

    # Initialize a dictionary to gather the data arrays
    dict_of_data ={}
    for i in range(n):
        # mock the data:
        mocked_voltage_data = np.random.uniform(-80, 40, (n_repetitions, n_smaples_per_electrode, n_elctrodes))
        # mock the attributes:
        ID = optional_IDs.pop()
        shuffle(genders)
        gender = genders[0]
        temp = np.random.normal(25, 3)
        humidity = np.random.normal(50, 15)
        shuffle(experimenters)
        experimenter = experimenters[0]

        # Creating the data:
        dims = ('repetition', 'time', 'electrode')
        coords = {'repetition': np.arange(n_repetitions),
                  'time': time_points,
                  'electrode': np.arange(n_elctrodes)}
        data = xr.DataArray(mocked_voltage_data, dims=dims, coords=coords, attrs= {'Rat_ID': ID, 'Rat_Gender': gender, 'Room_Temp' :temp, 'Room_Humid': humidity, 'Experimenter': experimenter})

        # adding stim_type indication
        data['stim_type'] = ('time', stim_type)
        # store it in the designated dictionary from which the Dataset will be created and returned
        dict_of_data[data.attrs['Rat_ID']] = data

    return VisualStimData(xr.Dataset(dict_of_data, attrs={'Experiment': 'Home_work', 'Data_Source': 'Mocked'}))


if __name__ == '__main__':
    stim_data = mock_stim_data()  # default is 50 subjects.
    stim_data.plot_electrode(rep_number=2, rat_id=101, elec_number=(1, 4, 7))  # add necessary vars
    stim_data.experimenter_bias()


import xarray as xr
import numpy as np
import pandas as pd
from random import shuffle

class VisualStimData:
    """
    Data and methods for the visual stimulus ePhys experiment.
    The data table itself is held in self.data, an `xarray` object.
    Inputs:
        data: xr.DataArray or xr.Dataset
        ...
    Methods:
         ...
    """
    def __init__(self, data, ):
        self.data = data
        if not isinstance(self.data,(xr.DataArray, xr.Dataset)):
            raise TypeError("The data inserted must be an xarray DataArray or Dataset")

    def plot_electrode(self, rep_number: int, rat_id: int, elec_number: tuple=(0,)):
        """
        Plots the voltage of the electrodes in "elec_number" for the rat "rat_id" in the repetition
        "rep_number". Shows a single figure with subplots.
        """
        pass

    def experimenter_bias(self):
        """ Shows the statistics of the average recording across all experimenters """
        pass


def mock_stim_data() -> VisualStimData:
    """ Creates a new VisualStimData instance with mock data """
    n_elctrodes = 10
    n_smaples_per_electrode = 10000
    n_repetitions = 4
    trial_t = 2
    pre_t = 1
    during_t = 0.1
    post_t = 0.9

    mocked_elcrodes_data = np.random.uniform(-80, 40, (n_repetitions, n_smaples_per_electrode, n_elctrodes))

    # Creating the data:
    dims = ('repetition', 'time', 'electrode')
    coords = {'repetition': np.arange(n_repetitions),
              'time': np.linspace(0, trial_t, num=n_smaples_per_electrode),
              'electrode': np.arange(n_elctrodes)}
    data = xr.DataArray(mocked_elcrodes_data, dims=dims, coords=coords)
    pass


if __name__ == '__main__':
    stim_data = mock_stim_data()
    stim_data.plot_electrode()  # add necessary vars
    stim_data.experimenter_bias()


np.random.uniform(-80, 40, (2, 4))
np.linspace(0, 2, num=10000)

if not isinstance(a, (xr.DataArray, xr.Dataset)):
    raise TypeError("The data inserted must be an xarray DataArray or Dataset")

--

def mock_stim_data(n=10) -> VisualStimData:
    """ Creates a new VisualStimData instance with mock data """
    n_elctrodes = 10
    n_smaples_per_electrode = 10000
    n_repetitions = 4
    trial_t = 2
    pre_t = 1
    during_t = 0.1
    post_t = 0.9

    # initiate a ID optional numbers:
    optional_IDs = list(range(999, 100, -1))
    genders = ['male', 'female']
    experimenters = ['A', 'B', 'C', 'D']

    # handling time points and stim type:
    time_points = np.linspace(0, trial_t, num=n_smaples_per_electrode)
    stim_type = np.concatenate((np.full((int(pre_t/trial_t*n_smaples_per_electrode)),-1), np.full((int(during_t/trial_t*n_smaples_per_electrode)),0), np.full((int(post_t/trial_t*n_smaples_per_electrode)),1)))


    dict_of_data ={}
    for i in range(n):
        mocked_voltage_data = np.random.uniform(-80, 40, (n_repetitions, n_smaples_per_electrode, n_elctrodes))

        # mock the attributes:
        ID = optional_IDs.pop()
        shuffle(genders)
        gender = genders[0]
        temp = np.random.normal(25,3)
        humidity = np.random.normal(50,15)
        shuffle(experimenters)
        experimenter = experimenters[0]

        # add the stim_type (-1 is pre, 0 is during, 1 is post)
        # Creating the data:
        dims = ('repetition', 'time', 'electrode')
        coords = {'repetition': np.arange(n_repetitions),
                  'time': time_points,
                  'electrode': np.arange(n_elctrodes)}
        data = xr.DataArray(mocked_voltage_data, dims=dims, coords=coords, attrs= {'Rat_ID': ID, 'Rat_Gender': gender, 'Room_Temp' :temp, 'Room_Humid': humidity, 'Experimenter': experimenter})

        # adding stim_type indication
        data['stim_type'] = ('time', stim_type)
        dict_of_data[data.attrs['Rat_ID']] = data

    return  VisualStimData(xr.Dataset(dict_of_data, attrs={'Experiment': 'Home_work', 'Data_Source': 'Mocked'}))



stim_type
"""Samples 2D analysis result

With this class, you can sample linearly interpolated where ever you want.
And you can save the original and sampled data.
In Ansys, you have to export whole value and selection info can be smaller 
regeon than value. Value txt file has inlcude the regeon of selection info.

Typical usage:

With Numpy result:
    sampler = ResultSampler()
    sampler.load_numpy_result(xs, ys,
        {'key1':values1, 'key2':values2, ...}) # generate pandas dataframe

    print(sampler.get_result_df())
    sampled_result = sampler.sample_result(xs, ys, 'key1', save_path='./result.csv') # linear interpolate and sample


With .csv Ansys result:
    sampler = ResultSampler()
    sampler.load_ansys_result(
        coordinate_path= './coord.txt',
        {
            'disp_x':'./displacement_x.txt',
            'disp_y':'./displacement_y.txt',
            'stress_x':'./ stress_x.txt',...
        }
    )

    print(sampler.get_result_df())
    sampled_result = sampler.sample_result(xs, ys, 'key1', save_path) # linear interpolate and sample
"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata

class ResultSampler:
    """ Sample analysis result and visualize 
    
    Currently only varified to work on 2D analys.
    Basically this class read result into Pandas DataFrame as an intermediate
    data. Keys of DataFrame will be 'Node ID','x', 'y', 'user_def1', 'user_def2', ...
    """

    def __init__(self):
        self.result_df = pd.DataFrame()

    
    def load_ansys_result(self, coord_file_path, value_path_dict ,is_3d=False):
        """ Load Ansys result txt.

        Load csv and create pandas data frame. First, load selected coordinate
        (the Ansys 'selection information'). Next, read values e.g displacement,
        and append values to self.result_df, only where selected, which means
        only corresponding value to the coordinate.

        Args:
            coord_file_path: str. Path to the txt result file.
            value_path_dict: dictionary containing 'key' and corresponding file
            path string. Example:
                    {'disp_x':'./result/displacement_x',
                     'disp_y':'./result/displacement_y'}
        Returns:
            None
        """
        self.result_df = pd.DataFrame()

        coord_df = pd.read_csv(coord_file_path, delimiter="\t")
        coord_df = coord_df.iloc[:, [4, 1, 2]] # 4:Node ID, 1:x, 2:y
        coord_df = coord_df.drop_duplicates(subset='Node ID')
        coord_df = coord_df.sort_values(by=['Node ID'])
        coord_df = coord_df.set_axis(['Node ID', 'x', 'y'], axis='columns')
        coord_df.set_index(keys='Node ID', drop=False, inplace=True)

        self.result_df = coord_df


        # Node Number == Node ID in Ansys
        for key_i, path_i in value_path_dict.items():
            df = pd.read_csv(path_i, delimiter="\t")
            df = df.drop_duplicates(subset='Node Number')
            df.set_index(keys='Node Number', inplace=True) # column as index

            # Save only selected data in coord df (in Ansys, selection info)
            selected_id_mask = coord_df.loc[:, 'Node ID']
            df = df.loc[selected_id_mask, :] # mask with pands series
            df = df.sort_values(by=['Node Number'])
            df = df.set_axis([key_i], axis='columns')
            
            self.result_df[key_i] = df # append data


        # print(self.result_df.head())
    def load_numpy_result(self, xs, ys, value_dict, is_3d=False):
        """ Load numpy result
        Load numpy and create pandas data frame for later use, and save it to
        self.result_df.

        Args:
            xs: array like. shape (n,). Original position of x
            ys: array like. shape (n,). Original position of y
            value_path_dict: dictionary containing 'key' and corresponding numpy
            Example:
                    {'disp_x': u_pred_np,
                     'disp_y': v_pred_np}
        Returns:
            None
        """
        self.result_df = pd.DataFrame()
        coord_df = pd.DataFrame({'x':xs, 'y':ys})
        values_df = pd.DataFrame(value_dict)

        self.result_df = pd.concat((coord_df, values_df), axis=1)
        # print(self.result_df)



    def get_result_df(self):
        """Getter of result data frame"""
        return self.result_df
            

    def sample_result(self, xs, ys, key, save_path=''):
        """Sample interpolated result from original data.
        
        This method uses scipy.interpolate.griddata function for interpolation.
        griddata function returns nan value if sampled point is out of bounds.
        This sampling can be 2D area or 1D line. Just specfy where you want to
        sample by (xs, ys).

        Args:
            xs:  Array like. 1D array. shape=(n, ).
            ys:  Array like. 1D array. shape=(n, ).
            key: str, key name specified while loading data.
            save_path: str, if you want to save sampled result, then give path.
                if you specify nothing, then no saving happen.

        Returns:
            values: Array like. 1D array. shape=(n, ).
        """
        orig_xs = self.result_df.loc[:, 'x']
        orig_ys = self.result_df.loc[:, 'y']
        orig_values = self.result_df.loc[:, key]


        interpolated_value = griddata((orig_xs, orig_ys),
                                      orig_values,
                                      (xs, ys), # sample position
                                      method='linear')
        
        
        if save_path != '':
            self.__save_sampled_result(xs, ys, interpolated_value, key, save_path)

        return interpolated_value

    
    def __save_sampled_result(self, xs, ys, values, key, save_path):
        """Save sampled result using pandas DataFrame.
        """
        df = pd.DataFrame({'x':xs, 'y':ys, key:values})
        print(df)
        df.to_csv(save_path, index=False)
        print("Sampled data was saved to '{}'".format(save_path))
     

    def save_original(self, file_path):
        """ Save original data frame """
        self.result_df.to_csv(file_path, index=False)
        print("Original data was save to '{}'".format(file_path))


        
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sampler = ResultSampler()
    sampler.load_ansys_result(
        coord_file_path='./result_whole/coord.txt',
        value_path_dict={
            'disp_x' : './result_whole/disp_x.txt',
            'disp_y' : './result_whole/disp_y.txt',
            'stress_x':'./result_whole/stress_x.txt',
            'stress_y':'./result_whole/stress_y.txt',
            }
    )

    # sampler.save_original('./result_whole/result_Ansys.csv')


    df = sampler.get_result_df()

    xs = df.loc[:, 'x'].to_numpy()
    ys = df.loc[:, 'y'].to_numpy()
    u_pred = df.loc[:, 'disp_x'].to_numpy()
    v_pred = df.loc[:, 'disp_y'].to_numpy()
    sigma_y_pred = df.loc[:, 'stress_y'].to_numpy()

    sampler_np = ResultSampler()
    sampler_np.load_numpy_result(
        xs, ys, 
        value_dict={
            'disp_x':u_pred,
            'disp_y':v_pred,
            'stress_y':sigma_y_pred
            })

    
    # x_grid = np.linspace(1.5, 5, num=50)
    # y_grid = np.linspace(-1, 1, num=20)

    # X,Y = np.meshgrid(x_grid, y_grid)
    # xs = X.flatten()
    # ys = Y.flatten()

    xs = np.linspace(1.5, 5, num=50)
    ys = np.zeros_like(xs)

    values = sampler.sample_result(xs, ys, 'stress_y', 
                save_path='./result_whole/sampled_stress_y_np.csv')

    # b = plt.scatter(xs, ys, c=values)


    # fig = plt.figure(figsize=(6,4))
    # ax = fig.add_subplot()

    # ax.axis('equal')
    # ax.grid(ls=':')

    # b=ax.scatter(xs, ys, c=values)
    # plt.colorbar(b)
    
    # plt.plot(xs, values)
    # plt.show()
    # print(sampler.get_result_df())










"""
    import sys
    sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\ReadExcel2D")
 
    from VizExcel import ExcelViewer, PlotExcel

    file_path = '../results.xlsx'

    excel_result = ExcelViewer(file_path)
    print(excel_result.GetDataFRame())

    plotter = PlotExcel()
    #plotter.plotResult(excel_result, key='S11', hole_radius=1.0)
    plotter.plotResult(excel_result, key='S11', hole_radius=None)
"""
# 2D Excel contour Plotter with arbitrary xyz value.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri



class ExcelViewer():
    def __init__(self, file_path):
        """ Open excel file and view 
        Arguments:
            - file_path: str, file_path
        """
        self.file_path = file_path
        self.resultDataFrame = pd.read_excel(file_path)




     
    def GetDataFRame(self):
        """ Returns DataFrame """
        return self.resultDataFrame






class PlotExcel():
    def __init__(self):
        # self.file_dir = file_dir
        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)


    def plotResult(self, excelViewer, key='S11', hole_radius=None, contour_num=20):
        """ 
        Arguments
        - excel Viewer: ExcelViewer object,
        - key: str, header name for coloring the plot.
        - hole_radius: float, this is applyed only for center hole models. 
                       Set None for non hole object.
        """
        result_DataFrame = excelViewer.GetDataFRame()

        x = result_DataFrame.loc[:, 'x'].to_numpy()
        y = result_DataFrame.loc[:, 'y'].to_numpy()
        value = result_DataFrame.loc[:, key].to_numpy()

        # triangles to render for surface plot
        triang = tri.Triangulation(x, y)

        # If you did not set hole_radius, then no masking happens.
        if hole_radius is not None:
            tri_center_x = x[triang.triangles].mean(axis=1)
            tri_center_y = y[triang.triangles].mean(axis=1)
            mask = np.where(((tri_center_x**2+tri_center_y**2)<hole_radius**2), 1, 0)
            triang.set_mask(mask)
        

        # self.ax.tricontourf(x,y,value, 20)
        contour = self.ax.tricontourf(triang, value, contour_num)
        self.ax.tricontour(triang, value, contour_num, colors='w', linewidths=0.5, linestyles='solid')
        self.ax.set_aspect('equal')
        self.ax.set_title(f'{key}')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        plt.colorbar(contour, shrink=1, pad=0.05)
        plt.tight_layout()
        plt.show()



def debug_visualize_mesh(x, y, value, contour_num=20):
    """ This function visualize data using triang.

    Args: 
        x    : array like. 
        y    : array like.
        value: array like.
        contour_num: int. The number of contour line to be drawn.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    
    triang = tri.Triangulation(x, y)

    

    contour = ax.tricontourf(triang, value, contour_num)
    ax.tricontour(triang, value, contour_num, colors='w', linewidths=0.5, linestyles='solid')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.colorbar(contour, shrink=1, pad=0.05)
    plt.tight_layout()
    plt.show()
 
    


if __name__ == '__main__':
    # file_path = './exact.xlsx'
    # file_path = './results.xlsx'
    file_path = '../results.xlsx'

    excel_result = ExcelViewer(file_path)
    # print(excel_result.GetDataFRame())

    plotter = PlotExcel()
    plotter.plotResult(excel_result, key='S11', hole_radius=1.0)
    # plotter.plotResult(excel_result, key='S11', hole_radius=None)


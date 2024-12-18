import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
import os


def plot_info_plane(args):
    inputs_outputs_arr = np.load(args.input_output_MI_path
                                  , allow_pickle=True)
    Y_outputs_arr = np.load(args.output_modelOutput_MI_path
                             , allow_pickle=True)

    # print(inputs_outputs_arr_clean1.shape, Y_outputs_arr_clean1.shape)  # (11, ) (11, 100)
    info_plane = np.empty([33, 2])
    for idx in range(33):
        info_plane[idx, 0] = np.mean(inputs_outputs_arr[idx][-5:])
        info_plane[idx, 1] = np.mean(Y_outputs_arr[idx][-5:])
    # x轴: inputs, Y轴: outputs
    df = pd.DataFrame(columns=['Epoch', 'I(X;T)', 'I(T;Y)'])
    df['Epoch'] = np.arange(0, 99, 3)
    df['I(X;T)'] = info_plane[:, 0]
    df['I(T;Y)'] = info_plane[:, 1]
    fig, ax = plt.subplots()
    sca = ax.scatter(x=df['I(X;T)'], y=df['I(T;Y)'], c=df['Epoch'], cmap='summer')
    ax.set_xlabel('I(X;T)')
    ax.set_ylabel('I(T;Y)')
    fig.colorbar(sca, label="Epoch", orientation="vertical")

    # Generate save path
    input_dir = os.path.dirname(args.input_output_MI_path)
    input_filename = os.path.basename(args.input_output_MI_path)
    class_info = input_filename.split('_class_')[-1].split('.')[0] if '_class_' in input_filename else ''
    save_filename = f"info_plane{f'_class_{class_info}' if class_info else ''}.png"
    save_path = os.path.join(input_dir, save_filename)

    fig.savefig(save_path, dpi=300)
    print(f"Figure saved to: {save_path}")


def test():
    arr = np.load(r'figs-inputs-vs-outputs/infoNCE.npy')
    print(arr.shape)
    pass


def manovaAnalyze():
    # 读取数据
    data = pd.read_csv('poison_clean.csv')
    print(data.dtypes)

    exog = data[['label']]
    endog = data[['intial_x', 'initial_y', 'turning_x', 'turning_y', 'conv_x', 'conv_y']].astype(float)

    # 进行MANOVA分析
    maov = MANOVA(endog, exog)
    print(maov.mv_test())


# def annovaAnalyze():
#     # Annova
#     data = pd.read_csv('annova.csv')
#     print(data.dtypes)
#     # annova
#     f_statistic, p_value = stats.f_oneway(data['intial_x'], data['initial_y'], data['turning_x'], data['turning_y'], data['conv_x'], data['conv_y'])
#
#     print('F statistic:', f_statistic)
#     print('P value:', p_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_output_MI_path', type=str, default='results/ob_infoNCE_06_22', help='output_dir')
    parser.add_argument('--output_modelOutput_MI_path', type=str, default='results/ob_infoNCE_06_22', help='output_dir')
    args = parser.parse_args()
    plot_info_plane(args)
    # test()
   # manovaAnalyze()
 #   annovaAnalyze()
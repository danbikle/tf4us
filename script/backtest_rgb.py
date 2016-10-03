# ~/reg4us/script/backtest_rgb.py

# This script should plot an RGB visualization of predictions.

# This script should be called from ~/reg4us/script/backtest.bash

# Demo:
# ~/anaconda3/bin/python backtest_rpt.py ../public/csv/backtest_all.csv

import numpy  as np
import pandas as pd
import pdb

# I should check cmd line args
import sys
if (len(sys.argv) != 2):
  print('You typed something wrong:')
  print('Demo:')
  print("~/anaconda3/bin/python ~/anaconda3/bin/python backtest_rpt.py ../public/csv/backtest_all.csv")
  sys.exit()

csv_in = sys.argv[1]
bt_df  = pd.read_csv(csv_in)

rgb0_df          = bt_df[:-1][['cdate','cp']]
rgb0_df['cdate'] = pd.to_datetime(rgb0_df['cdate'], format='%Y-%m-%d')
rgb0_df.columns  = ['cdate','Long Only']
# I should create effectiveness-line for Linear Regression predictions.
# I have two simple rules:
# 1. If blue line moves 1%, then model-line moves 1%.
# 2. If model is True, model-line goes up.
len_i      = len(rgb0_df)
blue_l     = [cp       for cp       in bt_df.cp]
pred_linr_l = [pred_linr for pred_linr in bt_df.pred_linr]
linr_l     = [blue_l[0]]
for row_i in range(len_i):
  blue_delt = blue_l[row_i+1]-blue_l[row_i]
  linr_delt = np.sign(pred_linr_l[row_i]) * blue_delt
  linr_l.append(linr_l[row_i]+linr_delt)
rgb0_df['Linear Regression'] = linr_l[:-1]

# I should create effectiveness-line for Logistic Regression predictions.
pred_logr_l = [pred_logr for pred_logr in bt_df.pred_logr]
logr_l     = [blue_l[0]]
for row_i in range(len_i):
  blue_delt = blue_l[row_i+1]-blue_l[row_i]
  logr_delt = np.sign(pred_logr_l[row_i]-0.5) * blue_delt
  logr_l.append(logr_l[row_i]+logr_delt)
rgb0_df['Logistic Regression'] = logr_l[:-1]

import matplotlib
matplotlib.use('Agg')
# Order is important here.
# Do not move the next import:
import matplotlib.pyplot as plt

rgb1_df = rgb0_df.set_index(['cdate'])
rgb1_df.plot.line(title="RGB Effectiveness Visualization", figsize=(11,7))
plt.savefig('../public/backtest_rgb.png')
plt.close()
'bye'

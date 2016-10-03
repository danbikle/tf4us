# backtest_rpt.py

# This script should report on results of a backtest.

# This script should be called from ~/tf4us/script/backtest.bash

# Demo:
# ~/anaconda3/bin/python backtest_rpt.py ../public/csv/backtest_2001.csv

import numpy  as np
import pandas as pd
import pdb

# I should check cmd line args
import sys
if (len(sys.argv) != 2):
  print('You typed something wrong:')
  print('Demo:')
  print("~/anaconda3/bin/python ~/anaconda3/bin/python backtest_rpt.py ../public/csv/backtest_2001.csv")
  sys.exit()
  
csv_in = sys.argv[1]
bt_df  = pd.read_csv(csv_in)

# I should report long-only-effectiveness:
eff_lo_f = np.sum(bt_df.pctlead)

# I should report Linear-Regression-Effectiveness:
eff_sr     = bt_df.pctlead * np.sign(bt_df.pred_linr)
bt_df['eff_linr'] = eff_sr
eff_linr_f        = np.sum(eff_sr)

# I should report Logistic-Regression-Effectiveness:
eff_sr     = bt_df.pctlead * np.sign(bt_df.pred_logr - 0.5)
bt_df['eff_logr'] = eff_sr
eff_logr_f        = np.sum(eff_sr)

# I should report tf10-Effectiveness:
eff_sr     = bt_df.pctlead * np.sign(bt_df.tf10 - 0.5)
bt_df['eff_tf10'] = eff_sr
eff_tf10_f        = np.sum(eff_sr)

# I should report tf11-Effectiveness:
eff_sr     = bt_df.pctlead * np.sign(bt_df.tf11 - 0.5)
bt_df['eff_tf11'] = eff_sr
eff_tf11_f        = np.sum(eff_sr)

print('csv_in: '+csv_in)
print('Long-Only-Effectiveness: '          +str(eff_lo_f))
print('Linear-Regression-Effectiveness: '  +str(eff_linr_f))
print('Logistic-Regression-Effectiveness: '+str(eff_logr_f))
print('tf10-Effectiveness: '+str(eff_tf10_f))
print('tf11-Effectiveness: '+str(eff_tf11_f))

'bye'

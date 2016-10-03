# learn_tst_rpt.py

# This script should learn from observations in ~/tf4us/public/csv/feat.csv

# Then it should test its learned models on observations later than the training observations.

# Next it should report effectiveness of the models.

# Demo:
# ~/anaconda3/bin/python learn_tst_rpt.py TRAINSIZE=25 TESTYEAR=2000

# Above demo will train from years 1985 through 2014 and predict each day of 2000

import numpy  as np
import pandas as pd
import pdb

# I should check cmd line args
import sys
if (len(sys.argv) != 3):
  print('You typed something wrong:')
  print('Demo:')
  print("~/anaconda3/bin/python genf.py TRAINSIZE=30 TESTYEAR=2015")
  sys.exit()

# I should get cmd line args:
trainsize     = int(sys.argv[1].split('=')[1])
testyear_s    =     sys.argv[2].split('=')[1]
train_end_i   = int(testyear_s)
train_end_s   =     testyear_s
train_start_i = train_end_i - trainsize
train_start_s = str(train_start_i)
# train and test observations should not overlap:
test_start_i  = train_end_i
test_start_s  = str(test_start_i)
test_end_i    = test_start_i+1
test_end_s    = str(test_end_i)

feat_df  = pd.read_csv('../public/csv/feat.csv')
train_sr = (feat_df.cdate > train_start_s) & (feat_df.cdate < train_end_s)
test_sr  = (feat_df.cdate > test_start_s)  & (feat_df.cdate < test_end_s)
train_df = feat_df[train_sr]
test_df  = feat_df[test_sr]

# I should build a Linear Regression model from slope columns in train_df:
x_train_a = np.array(train_df)[:,3:]
y_train_a = np.array(train_df.pctlead)
from sklearn import linear_model
linr_model = linear_model.LinearRegression()
# I should learn:
linr_model.fit(x_train_a, y_train_a)
# Now that I have learned, I should predict:
x_test_a       = np.array(test_df)[:,3:]
predictions_a  = linr_model.predict(x_test_a)
predictions_df = test_df.copy()
predictions_df['pred_linr'] = predictions_a.reshape(len(predictions_a),1)

# I should build a Logistic Regression model.
logr_model    = linear_model.LogisticRegression()
# I should get classification from y_train_a:
# Should I prefer median over mean?:
# class_train_a = (y_train_a > np.median(y_train_a))
class_train_a = (y_train_a > np.mean(y_train_a))
# I should learn:
logr_model.fit(x_train_a, class_train_a)
# Now that I have learned, I should predict:
predictions_a               = logr_model.predict_proba(x_test_a)[:,1]
predictions_df['pred_logr'] = predictions_a.reshape(len(predictions_a),1)

#
# I should build a TensorFlow model
import tensorflow as tf
sess10 = tf.InteractiveSession()

# tf wants class training values to be 1 hot encoded.
class_train1h_l = [[0,1] if cl else [1,0] for cl in class_train_a]
# [0,1] means up-observation
# [1,0] means down-observation
ytrain1h_a = np.array(class_train1h_l)
learning_rate   = 0.001

# I declare 2d Tensors.
# I should use 0th row of x_train_a to help shape xvals:
fnum_i  = len(x_train_a[0, :])
label_i = len(ytrain1h_a[0,:]) # Should usually be 2.

xvals = tf.placeholder(tf.float32, shape=[None, fnum_i], name='x-input')
weight1 = tf.Variable(tf.zeros([fnum_i, label_i]))
weight0 = tf.Variable(tf.zeros([label_i]))
yhat    = tf.nn.softmax(tf.matmul(xvals, weight1) + weight0)
# Define loss and optimizer
yactual       = tf.placeholder(tf.float32, [None, label_i])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(yactual * tf.log(yhat), reduction_indices=[1]))
# http://www.google.com/search?q=tensorflow+GradientDescentOptimizer+vs+AdamOptimizer
# http://sebastianruder.com/optimizing-gradient-descent/
#train_step    = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step    = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# Train
training_steps_i = 555
tf.initialize_all_variables().run()
for i in range(training_steps_i):
  train_step.run({xvals: x_train_a, yactual: ytrain1h_a})
prob_a = sess10.run(yhat, feed_dict={xvals: x_test_a})
# I should collect the tf predictions
predictions_df['tf10'] = prob_a[:,1]
# tensorflow sess10 should be done for now.
#

#
# I should build a TensorFlow model
print('sess11 VERY busy ...')
sess11 = tf.InteractiveSession()
learning_rate     = 0.001
training_steps_i  = 2123
layer1_input_dim  = fnum_i
layer1_output_dim = label_i

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  And adds a number of summary ops.
  """
  # layer_name should be used in later version.
  # I should hold the state of the weights for the layer.
  weights     = weight_variable([input_dim, output_dim])
  biases      = bias_variable([output_dim])    
  preactivate = tf.matmul(input_tensor, weights) + biases
  # I should use act() passed in via arg.
  # Default: act=tf.nn.relu
  activations = act(preactivate, 'activation')
  return activations
# I should declare some placeholder-variables:
xvals     = tf.placeholder(tf.float32, shape=[None, fnum_i] , name='x-input')
yactual   = tf.placeholder(tf.float32, shape=[None, label_i], name='y-input')
keep_prob = tf.placeholder(tf.float32, name='probability2keep-not-drop')
yhat = nn_layer(xvals, layer1_input_dim, layer1_output_dim, 'layer1', act=tf.nn.softmax)

cross_entropy = -tf.reduce_mean(yactual * tf.log(yhat))
train_step    = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# tensorflow sess11 should be done for now.
#


# I should create a CSV to report from:
predictions_df.to_csv('../public/csv/tf4.csv', float_format='%4.6f', index=False)

# I should report long-only-effectiveness:
eff_lo_f = np.sum(predictions_df.pctlead)
print('Long-Only-Effectiveness:')
print(eff_lo_f)

# I should report Linear-Regression-Effectiveness:
eff_sr     = predictions_df.pctlead * np.sign(predictions_df.pred_linr)
predictions_df['eff_linr'] = eff_sr
eff_linr_f                 = np.sum(eff_sr)
print('Linear-Regression-Effectiveness:')
print(eff_linr_f)

# I should report Logistic-Regression-Effectiveness:
eff_sr     = predictions_df.pctlead * np.sign(predictions_df.pred_logr - 0.5)
predictions_df['eff_logr'] = eff_sr
eff_logr_f                 = np.sum(eff_sr)
print('Logistic-Regression-Effectiveness:')
print(eff_logr_f)

# I should report tf10-Effectiveness:
eff_sr     = predictions_df.pctlead * np.sign(predictions_df.tf10 - 0.5)
predictions_df['eff_tf10'] = eff_sr
eff_tf10_f                 = np.sum(eff_sr)
print('tf10-Effectiveness:')
print(eff_tf10_f)

# I should use html to report:
model_l = ['Long Only', 'Linear Regression', 'Logistic Regression']
eff_l   = [eff_lo_f, eff_linr_f, eff_logr_f]

rpt_df                  = pd.DataFrame(model_l)
rpt_df.columns          = ['model']
rpt_df['effectiveness'] = eff_l
rpt_df.to_html(        '../app/views/pages/_agg_effectiveness.erb',      index=False)
predictions_df.to_html('../app/views/pages/_detailed_effectiveness.erb', index=False)

import matplotlib
matplotlib.use('Agg')
# Order is important here.
# Do not move the next import:
import matplotlib.pyplot as plt

rgb0_df          = predictions_df[:-1][['cdate','cp']]
rgb0_df['cdate'] = pd.to_datetime(rgb0_df['cdate'], format='%Y-%m-%d')
rgb0_df.columns  = ['cdate','Long Only']
# I should create effectiveness-line for Linear Regression predictions.
# I have two simple rules:
# 1. If blue line moves 1%, then model-line moves 1%.
# 2. If model is True, model-line goes up.
len_i       = len(rgb0_df)
blue_l      = [cp       for cp       in predictions_df.cp]
pred_linr_l = [pred_linr for pred_linr in predictions_df.pred_linr]
linr_l      = [blue_l[0]]
for row_i in range(len_i):
  blue_delt = blue_l[row_i+1]-blue_l[row_i]
  linr_delt = np.sign(pred_linr_l[row_i]) * blue_delt
  linr_l.append(linr_l[row_i]+linr_delt)
rgb0_df['Linear Regression'] = linr_l[:-1]

# I should create effectiveness-line for Logistic Regression predictions.
pred_logr_l = [pred_logr for pred_logr in predictions_df.pred_logr]
logr_l      = [blue_l[0]]
for row_i in range(len_i):
  blue_delt = blue_l[row_i+1]-blue_l[row_i]
  logr_delt = np.sign(pred_logr_l[row_i]-0.5) * blue_delt
  logr_l.append(logr_l[row_i]+logr_delt)
rgb0_df['Logistic Regression'] = logr_l[:-1]

rgb1_df = rgb0_df.set_index(['cdate'])
rgb1_df.plot.line(title="RGB Effectiveness Visualization "+testyear_s, figsize=(11,7))
plt.grid(True)
plt.savefig('../public/rgb.png')
plt.close()

'bye'

#!/bin/bash

# ~/tf4us/script/night.bash

# I should run this script at 8pm Calif-time.

# This script should generate some static content each night
# after the market has closed and the most recent GSPC-closing-price
# is available from Yahoo.

# The static content should help me compare effectiveness of GSPC
# predictions computed from Linear Regression and Logistic Regression.

# If you have questions, e-me: bikle101@gmail.com

# I should cd to the right place:
cd ~/tf4us/script/

# I should create a folder to hold CSV data:
mkdir -p ~/tf4us/public/csv/

# I should get prices from Yahoo:
## debug /usr/bin/curl http://ichart.finance.yahoo.com/table.csv?s=%5EGSPC > ~/tf4us/public/csv/gspc.csv

# I should extract two columns and also sort:
echo cdate,cp                                                              > ~/tf4us/public/csv/gspc2.csv
sort ~/tf4us/public/csv/gspc.csv|awk -F, '{print $1"," $5}'|grep -v Date >> ~/tf4us/public/csv/gspc2.csv

# I should compute features from the prices:
~/anaconda3/bin/python genf.py SLOPES='[2,3,4,5,6,7,8,9]'

# I should learn, test, and report:
~/anaconda3/bin/python learn_tst_rpt.py TRAINSIZE=25 TESTYEAR=2016

exit

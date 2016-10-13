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
# cd ~/tf4us/script/
export TF4US=`dirname $0`/../
cd    $TF4US
# I should create a folder to hold CSV data:
mkdir -p public/csv/
cd       public/csv/
# I should get prices from Yahoo:
/usr/bin/curl http://ichart.finance.yahoo.com/table.csv?s=%5EGSPC > gspc.csv

# I should extract two columns and also sort:
echo cdate,cp                                                              > gspc2.csv
sort ~/tf4us/public/csv/gspc.csv|awk -F, '{print $1"," $5}'|grep -v Date >> gspc2.csv

# I should compute features from the prices:
cd $TF4US/script/
python genf.py SLOPES='[2,3,4,5,6,7,8,9]'

# I should learn, test, and report:
python learn_tst_rpt.py TRAINSIZE=25 TESTYEAR=2016

exit

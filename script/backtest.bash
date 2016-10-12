#!/bin/bash

# ~/tf4us/script/backtest.bash

# This script should backtest Linear Regression and Logistic Regression with GSPC prices.

# If you have questions, e-me: bikle101@gmail.com

# I should cd to the right place:
cd ~/tf4us/script/

# I should create a folder to hold CSV data:
mkdir -p ~/tf4us/public/csv/

# I should get prices from Yahoo:
/usr/bin/curl http://ichart.finance.yahoo.com/table.csv?s=%5EGSPC > ~/tf4us/public/csv/gspc.csv

# I should extract two columns and also sort:
echo cdate,cp                                                              > ~/tf4us/public/csv/gspc2.csv
sort ~/tf4us/public/csv/gspc.csv|awk -F, '{print $1"," $5}'|grep -v Date >> ~/tf4us/public/csv/gspc2.csv

# I should compute features from the prices:
python genf.py SLOPES='[2,3,4,5,6,7,8,9]'
rm -f /tmp/learn_tst_rpt.py.txt
rm -f ../public/csv/backtest_*csv
rm -f ../public/backtest_$yr.png
thisyr=`date +%Y`
for (( yr=2000; yr<=${thisyr}; yr++ ))
do
    echo Busy...
    echo backtesting: $yr                                             >> /tmp/learn_tst_rpt.py.txt
    python learn_tst_rpt.py TRAINSIZE=25 TESTYEAR=$yr >> /tmp/learn_tst_rpt.py.txt 2>&1
    mv ../public/csv/tf4.csv ../public/csv/backtest_$yr.csv
    mv ../public/rgb.png ../public/backtest_$yr.png
    python backtest_rpt.py ../public/csv/backtest_${yr}.csv > /tmp/backtest_rpt_${yr}.py.txt 2>&1
done

cat ../public/csv/backtest_*.csv | sed -n 1p           > /tmp/backtest_all.csv
cat ../public/csv/backtest_*.csv | sort|grep -v cdate >> /tmp/backtest_all.csv
cp /tmp/backtest_all.csv ../public/csv/
python backtest_rpt.py ../public/csv/backtest_all.csv
python backtest_rgb.py ../public/csv/backtest_all.csv

# backtest.bash clobbers data created by night.bash
# I should fix that:
./night.bash

exit

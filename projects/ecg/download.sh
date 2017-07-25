#!/bin/sh
# source https://physionet.org/physiobank/database/mitdb/

# first dataset
for i in `seq -w 0 24`;
do
    wget https://physionet.org/physiobank/database/mitdb/1$i.atr
    wget https://physionet.org/physiobank/database/mitdb/1$i.hea
    wget https://physionet.org/physiobank/database/mitdb/1$i.dat
done   

# second dataset
for i in `seq -w 0 34`;
do
    wget https://physionet.org/physiobank/database/mitdb/2$i.atr
    wget https://physionet.org/physiobank/database/mitdb/2$i.hea
    wget https://physionet.org/physiobank/database/mitdb/2$i.dat
done   


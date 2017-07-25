#!/bin/bash
PATH=$PATH:../../bin

function extract_ann {
rdann -a atr -r mitdb/$1 -v | tr -s " #" "," | cut -d"," -f2- | gzip > $2.gz
} 

function extract_sig {
rdsamp -r mitdb/$1 -v | tr -s "\t #" "," | cut -d"," -f2- | gzip > $2.gz
}

function process {
    for i in `seq -w 0 $2`
    do
        rec=$1$i       
        if [ -f $rec.hea ]
        then
            echo Processing record $rec
            extract_ann $rec extracted/$c.ann
            extract_sig $rec extracted/$c.sig
            let c=c+1    
        else        
            echo Record $rec does not exists
        fi           
    done
}

c=0
process 1 24
process 2 34

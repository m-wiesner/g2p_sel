#! /bin/bash
cut -f2- $1 | LC_ALL= sed 's/\t//g' |\
  paste <(cut -f1 $1 | LC_ALL= sed 's/./\L&/g;s/_//g') - |\
  LC_ALL=C sort -u 

#
# this project validates partitioned models produced by circle-partitioner
# with circle-part-driver and two scripts; part_eval_all.sh and part_eval_one.py
#

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

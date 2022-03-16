NAME=$1

mv ${NAME}/channel ${NAME}/temp

mv ${NAME}/layer ${NAME}/channel
mv ${NAME}/channel/uint8 ${NAME}/channel/int16

mv ${NAME}/temp ${NAME}/layer
mv ${NAME}/layer/int16 ${NAME}/layer/uint8


StateArray=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)
# StateArray=(12 18 23 27)
RunArray=(0 1 2)
for state in ${StateArray[*]};do
	for run in ${RunArray[*]};do
            python tract_main.py --random-state $state --runtime $run
done;done

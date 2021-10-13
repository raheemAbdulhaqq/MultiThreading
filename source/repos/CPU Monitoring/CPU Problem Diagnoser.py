import psutil
import time
import statistics

max_time = 120
start_time = time.time()  # remember when we started
usageForTwoMinutes = []
while (time.time() - start_time) < max_time:
    intervalUsage = psutil.cpu_percent(interval=5)
    usageForTwoMinutes.append(intervalUsage)
print(usageForTwoMinutes)
currentUsage = psutil.cpu_percent(interval=5)
usageForTwoMinutes.pop(0)
usageForTwoMinutes.append(currentUsage)
avgUsage = statistics.mean(usageForTwoMinutes)
print(usageForTwoMinutes)
print(avgUsage)
if(currentUsage > 90):
    print('CPU usage is high')
elif(currentUsage > avgUsage*2):
    print('Usage level is more than double average in the last 2 minutes')

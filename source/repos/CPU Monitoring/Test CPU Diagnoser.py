import wmi
import psutil
import time
import statistics

f = wmi.WMI()
max_time = 10
start_time = time.time()  # remember when we started
usageForTwoMinutes = []
listOfProcesses = []
while (time.time() - start_time) < max_time:
    intervalUsage = psutil.cpu_percent(interval=5)
    usageForTwoMinutes.append(intervalUsage)
    for process in f.Win32_Process():
        listOfProcesses.append(process.ProcessId)
numberOfProcess = len(listOfProcesses)
print(usageForTwoMinutes)
currentUsage = psutil.cpu_percent(interval=5)
usageForTwoMinutes.pop(0)
usageForTwoMinutes.append(currentUsage)
avgUsage = statistics.mean(usageForTwoMinutes)
print(usageForTwoMinutes)
print(avgUsage)
if(currentUsage > 0):
    print("CPU usage is high because:")
    currentListOfProcess = []
    print("pid Process name")
    for process in f.Win32_Process():
        p = psutil.Process(process.ProcessId)
        currentListOfProcess.append(p)
        if(p.cpu_percent(interval=1.0) > currentUsage/2):
            #print("This process is using more than half of the CPU at this particular time"f"{process.ProcessId:<10} {process.Name}")
            print(process.Name)
    if(len(currentListOfProcess) > 0):
        print("There are many more process running than normal")
    else:
        print("Your CPU usage is high for an unknown reason.")
elif(currentUsage > avgUsage*2):
    print('Usage level is more than double average in the last 2 minutes')
    
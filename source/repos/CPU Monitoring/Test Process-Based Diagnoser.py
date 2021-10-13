import wmi
import psutil
import time

fiveMinutes = 30
start_time = time.time()
listOfProcessesId = []
listOfProcessesUsage = []
f = wmi.WMI()
print("PID  Usage")
for process in f.Win32_Process():
    p = psutil.Process(process.ProcessId)
    x = p.cpu_percent()
    print(p.pid, x)
    listOfProcessesId.append(process.ProcessId)
    listOfProcessesUsage.append(x)
    dictofProcesses = dict(zip(listOfProcessesId, listOfProcessesUsage))
print(listOfProcessesId)
print(listOfProcessesUsage)
print(dictofProcesses)

#after five minutes
while(time.time() - start_time) == fiveMinutes:
    for process in f.Win32_Process():
        q = psutil.Process(process.ProcessId)
        y = q.cpu_percent(interval=1.0)
        if(q == dictofProcesses.p & y > dictofProcesses.values(q)*10):
            print("This process has increased it's usage to over times from 5 minutes earlier!")

        

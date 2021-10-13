import psutil

while True:
    cpu_usage = psutil.cpu_percent(interval = 5)
    print(cpu_usage)
    if cpu_usage > 50:
        print("Your CPU usage is high!")
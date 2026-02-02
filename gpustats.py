from gpustat import GPUStatCollection

def monitor_gpu():
    # Fetch latest snapshots of all GPUs
    gpu_stats = GPUStatCollection.new_query()
    
    for gpu in gpu_stats:
        print(f"GPU [{gpu.index}]: {gpu.name}")
        print(f"  - Health: {'OK' if gpu.temperature < 85 else 'CRITICAL (High Temp)'}")
        print(f"  - Temperature: {gpu.temperature}°C")
        print(f"  - Utilization: {gpu.utilization}%")
        print(f"  - Memory: {gpu.memory_used}/{gpu.memory_total} MB")
        
        # Optional: Print power draw if available
        if gpu.power_draw is not None:
            print(f"  - Power Draw: {gpu.power_draw}W")
        
        # List processes running on this specific GPU
        print("  - Active Processes:")
        for proc in gpu.processes:
            print(f"    * PID {proc['pid']} ({proc['command']}): {proc['gpu_memory_usage']} MB")
        print("-" * 30)

if __name__ == "__main__":
    try:
        monitor_gpu()
    except Exception as e:
        print(f"Error querying GPU: {e}")

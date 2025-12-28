import subprocess
import sys

# Command to run vLLM
command = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2.5-32B-Instruct",
    "--quantization", "bitsandbytes", 
    "--load-format", "bitsandbytes",
    "--dtype", "half", 
    "--gpu-memory-utilization", "0.90",
    "--port", "8000",
    "--api-key", "EMPTY"
]

print("🚀 Starting vLLM Server...")
print("⚠️ KEEP THIS NOTEBOOK RUNNING! Do not close it.")
print("-" * 50)

# Run server
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Print logs
try:
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
except KeyboardInterrupt:
    print("\n Stopping server...")
    process.terminate()
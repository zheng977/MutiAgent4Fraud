#!/bin/bash
# Filename: llm_deploy.sh
# Description: Deploy vLLM inference server with port forwarding

MODEL_NAME="/path/to/your/model"  # e.g., /home/user/models/Qwen/Qwen2.5-32B-Instruct


PORT_NUM_PER_INSTANCE=40
# Main port for each vLLM instance
MAIN_PORTS=(
    10000
)
# Tensor parallelism degree (adjust with gpus-per-task)
TENSOR_PARALLEL_SIZE=4
# Port forwarding script path
PORT_FORWARD_SCRIPT="utils/port_forward.py"
# Port availability check script path
PORT_AVALIABLE_SCRIPT="utils/port_avaliable.py"
# Swap space size (GB)
SWAP_GB=2
# Log directory
LOG_DIR="./log"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/vllm_batch_$(date +%Y%m%d_%H%M%S).log"
# ===============================================
if [[ $SLURM_JOB_ID == "" ]]; then
  echo "Warning: not running under Slurm; continuing on current GPU node..." >&2
fi
# Generate unique temporary file
PORT_RECORD=$(mktemp -p /tmp port_record_XXXXXX.py)

# Stage 1: Launch all vLLM instances
run_vllm() {
    local port=$1
    echo "Starting vLLM instance (Port: $port, Log: $LOG_FILE)" | tee -a "$LOG_FILE"
    TORCH_COMPILE_DISABLE=1 VLLM_DISABLE_COMPILE=1 python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_NAME \
        --dtype auto \
        --port $port \
        --trust-remote-code \
        --block-size 32 \
        --swap-space $SWAP_GB \
        --gpu-memory-utilization 0.8 \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --disable-log-stats \
        >> "$LOG_FILE" 2>&1 &   
    echo "vLLM instance started on port: $port" | tee -a "$LOG_FILE"
}

for main_port in "${MAIN_PORTS[@]}"; do
    run_vllm $main_port
done

# Stage 2: Port allocation within safe range
declare -a ALL_PORTS
declare -a FORWARD_PIDS

for main_port in "${MAIN_PORTS[@]}"; do
    required_ports=$((PORT_NUM_PER_INSTANCE - 1))
    start_range=$((main_port + 1))
    end_range=$((main_port + PORT_NUM_PER_INSTANCE - 1))
    collected=0
    port_list=()
    
    # Probe within specified range
    for (( port=start_range; port<=end_range; port++ )); do
        # Enhanced port detection
        if python $PORT_AVALIABLE_SCRIPT $port
        then
            python $PORT_FORWARD_SCRIPT $port $main_port >> "$PORT_RECORD" 2>&1 &
            FORWARD_PIDS+=($!)
            ALL_PORTS+=($port)
            port_list+=($port)
            ((collected++))
            # echo "[Allocated] $port -> $main_port" >&2
        else
            echo "[Port Conflict] Port $port is already in use" | tee -a "$LOG_FILE"
            # Extend search range
            ((end_range++))
        fi
    done

    # Strict quantity validation
    if [ $collected -ne $required_ports ]; then
        echo "Fatal Error: Only found $collected/$required_ports available ports in range $start_range-$end_range" | tee -a "$LOG_FILE"
        echo "Conflicting ports may come from other instances or external services. Please check:" | tee -a "$LOG_FILE"
        ss -tulpn | grep -E ":($(seq -s '|' $start_range $end_range))" | tee -a "$LOG_FILE"
        exit 1
    fi
done

# Format output function
format_list() {
    local arr=("$@")
    printf "[%s]" "$(printf "%s, " "${arr[@]}" | sed 's/, $//')"
}

# Get public IP
PUBLIC_IP=$(
    ip -4 addr show eth0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1 ||
    ip -4 addr | grep -v '127.0.0.1' | awk '/inet / {print \$2}' | cut -d/ -f1 | head -1
)

# Summary output
echo "================ Final Configuration ================" | tee -a "$LOG_FILE"
echo "Service IP: $PUBLIC_IP" | tee -a "$LOG_FILE"
echo "Main Ports: $(format_list "${MAIN_PORTS[@]}")" | tee -a "$LOG_FILE"
echo "Forwarded Ports: $(format_list "${ALL_PORTS[@]}")" | tee -a "$LOG_FILE"
echo "All Available Ports: $(format_list "${MAIN_PORTS[@]}" "${ALL_PORTS[@]}")" | tee -a "$LOG_FILE"
echo "=====================================================" | tee -a "$LOG_FILE"

cleanup() {
    echo "Cleaning up resources..." | tee -a "$LOG_FILE"
    # Terminate all port forwarding processes
    for pid in "${FORWARD_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null
        fi
    done

    # Terminate all vLLM processes
    pkill -P $$ 2>/dev/null
    
    # Remove temporary files
    rm -f $PORT_RECORD
    
    echo "Resource cleanup completed" | tee -a "$LOG_FILE"
    exit 0
}

# Register signal handlers
trap cleanup SIGTERM SIGINT SIGHUP EXIT

# Wait for all background processes
wait

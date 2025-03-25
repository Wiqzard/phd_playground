ps aux | grep train.py | grep -v grep | while read -r line; do
    pid=$(echo "$line" | awk '{print $2}')
    echo "Force killing process: $line"
    kill -9 "$pid"
done
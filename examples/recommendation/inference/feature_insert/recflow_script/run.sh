
# create proto
protoc --python_out=. ./examples/recommendation/inference/feature_insert/protos/recflow.proto
echo "recflow.proto created !!!"

# insert data
python ./examples/recommendation/inference/feature_insert/recflow_script/insert_redis.py
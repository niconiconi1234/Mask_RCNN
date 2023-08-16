To compile maskrcnn.proto, please use the following command in `samples/maskrcnn-demo-server` directory:
```bash
python -m grpc_tools.protoc -I ./protos --python_out=. --grpc_python_out=. ./protos/maskrcnn.proto
```
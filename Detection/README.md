
```
docker run -it -v $(pwd)/flask:/tmp/src -p 8888:8888 -p 5000:5000 -e KERAS_BACKEND=tensorflow irobert0126/flask_keras bash
root@0ddf35e67dfd:/tmp/src# python3 main.py

Test:
curl -X POST -F model=@example/case1/nin_trojan_yellow_square_2_1.h5 'http://localhost:5000/predict'
```


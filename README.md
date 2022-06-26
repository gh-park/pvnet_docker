
# pvnet_docker
```bash
   bash build.sh
   bash start.sh
```
# 학습결과 복사
```bash

   docker cp 보낼파일.pth  pvnet:/root/workspace/pvnet_smc/data/model/pvnet/mycat

```
# 도커내부
```bash
   ./make_dataset.sh
   ./start_training.sh
   ./test.sh
   ./start_realsense.sh
```


# PILLOW버전 안맞는경우
```bash
   pip3 install Pillow==6.1
```

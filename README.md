
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
# OPENCV 에러
```bash
   pip3 uninstall opencv-python; 
   pip3 install opencv-python;
```
# 학습실행 시 ubuntu시스템의 priority 높여서 실행
```bash
   nice -n 99999 COMMAND
```

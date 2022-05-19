#######Example command#######
# sudo docker run -i -d --name dexnet --gpus all --network host \
#     --mount type=bind,source=/home/elevenjiang/Documents/Project/ObjectSynthesis/Code,target=/home/Project/Code \
#     --privileged -v /dev:/dev -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
#     -v /tmp/.X11-unix:/tmp/.X11-unix\
#     nvidia/cudagl:10.0-devel-ubuntu16.04


sudo docker run -i -d --name dexnet --gpus all --network host \
    --mount type=bind,source=/Change/to/Your/own/code!!!,target=/home/Project/Code \
    --privileged -v /dev:/dev -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix\
    nvidia/cudagl:10.0-devel-ubuntu16.04

sudo xhost +local:`docker inspect --format='{{ .Config.Hostname }}' dexnet`
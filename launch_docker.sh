container_name=$1
xhost +local:
XAUTH=/home/tribhi/.docker.xauth
touch $XAUTH
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

docker run -it --net=host\
    --user=$(id -u) \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="USER=$USER" \
    --workdir=/home/$USER/ \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="${PWD}:/home/$USER" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    -v "/etc/passwd:/etc/passwd:rw" \
    --gpus all \
    --runtime nvidia \
    --security-opt seccomp=unconfined \
    --name=${container_name} \
    medirl:noetic

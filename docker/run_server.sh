if [ -z $1 ];
then
    port=8080
else
    port=$1
fi
echo "port is set to $port"

docker run --rm -it -p $port:8080 hifuku:latest \
    /bin/bash -i -c \
    '
    cd ~/scikit-robot && git remote add h-ishida git@github.com:HiroIshida/scikit-robot.git && git fetch h-ishida && git checkout h-ishida/master && pip3 install -e . ;\
    pip3 install selcol --upgrade ;\
    cd ~/scikit-motionplan && git fetch origin && git checkout origin/master && pip3 install -e . ;\
    cd ~/rpbench && git fetch origin && git checkout origin/master && pip3 install -e . ;\
    cd ~/hifuku && git fetch origin && git checkout origin/master && pip3 install -e . ;\
    pip3 freeze ;\
    python3 -m hifuku.datagen.http_datagen.server'

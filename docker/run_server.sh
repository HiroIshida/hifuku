if [ -z $1 ];
then
    port=8080
else
    port=$1
fi
echo "port is set to $port"

docker run --rm -it -p $port:8080 hifuku:latest \
    /bin/bash -i -c \
    'cd ~/scikit-plan && git fetch origin && git checkout origin/master && pip3 install -e . ;\
    cd ~/scikit-robot && git fetch origin && git checkout origin/master && pip3 install -e . ;\
    cd ~/hifuku && git fetch origin && git checkout origin/master && pip3 install -e . ;\
    pip3 freeze ;\
    python3 -m hifuku.http_datagen.server'

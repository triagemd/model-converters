FROM debian:latest

RUN apt-get -y update && \
    apt-get -y install python-pip python3-pip && \
    pip2 install --upgrade pip && \
    pip3 install --upgrade pip && \
    # we install these packages at runtime in the entrypoint, but might as well fetch the latest version into the container
    mkdir /pip && \
    pip2 download -d /pip tensorflow keras h5py model-converters && \
    pip3 download -d /pip tensorflow keras h5py model-converters

ADD entrypoint.sh /opt/

CMD ["/opt/entrypoint.sh"]

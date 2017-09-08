FROM debian:latest

RUN apt-get -y update && \
    apt-get -y install python-pip python3-pip && \
    pip2 install --upgrade pip && \
    pip3 install --upgrade pip

# we install these packages at runtime in the entrypoint, but might as well fetch the latest version into the container
RUN mkdir /pip && \
    pip2 download -d /pip tensorflow keras h5py && \
    pip3 download -d /pip tensorflow keras h5py

RUN pip2 install model-converters && \
    pip3 install model-converters && \
    pip install stored

ADD entrypoint.sh /opt/

CMD ["/opt/entrypoint.sh"]

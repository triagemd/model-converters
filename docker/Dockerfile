FROM debian:latest

RUN apt-get -y update && \
    apt-get -y install python-pip python3-pip && \
    pip2 install --upgrade pip && \
    pip3 install --upgrade pip && \
    pip install stored>=0.0.29

# we install these packages at runtime in the entrypoint, but might as well fetch the latest version into the container
RUN mkdir /pip && \
    pip2 download -d /pip tensorflow keras h5py model-converters && \
    pip3 download -d /pip tensorflow keras h5py model-converters
# note: don't install model-converters here, since it declares a dependency on tensorflow and keras (so those will be installed too)
#       which is a waste of space since they get reinstalled in the entrypoint

ADD entrypoint.sh /opt/

CMD ["/opt/entrypoint.sh"]

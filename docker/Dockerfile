FROM debian:latest

# --upgrade-strategy=eager is because apparently the version of urllib that ships in stable Debian has a bug that breaks retries
RUN apt-get -y update && \
    apt-get -y install python-pip python3-pip && \
    apt-get -y clean && \
    pip2 install --upgrade --upgrade-strategy=eager pip stored && \
    pip3 install --upgrade --upgrade-strategy=eager pip stored

# we install these packages at runtime in the entrypoint, but might as well fetch the latest version into the container
RUN mkdir /pip && \
    pip2 download -d /pip tensorflow keras h5py model-converters && \
    pip3 download -d /pip tensorflow keras h5py model-converters
# note: don't install model-converters here, since it declares a dependency on tensorflow and keras (so those will be installed too)
#       which is a waste of space since they get reinstalled in the entrypoint

ADD entrypoint.sh /opt/

CMD ["/opt/entrypoint.sh"]

FROM debian:latest

# --upgrade-strategy=eager is because apparently the version of urllib that ships in stable Debian has a bug that breaks retries
RUN apt-get -y update && \
    apt-get -y install python-pip python3-pip && \
    apt-get -y clean && \
    python2 -m pip install --upgrade --upgrade-strategy=eager --no-cache-dir pip stored && \
    python3 -m pip install --upgrade --upgrade-strategy=eager --no-cache-dir pip stored

# we install these packages at runtime in the entrypoint, but might as well fetch the latest version into the container
RUN mkdir /pip && \
    python2 -m pip download -d /pip tensorflow keras h5py model-converters && \
    python3 -m pip download -d /pip tensorflow keras h5py model-converters
# note: don't install model-converters here, since it declares a dependency on tensorflow and keras (so those will be installed too)
#       which is a waste of space since they get reinstalled in the entrypoint

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ADD entrypoint.sh /opt/

CMD ["/opt/entrypoint.sh"]

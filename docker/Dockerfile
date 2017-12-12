FROM alpine:latest

# --upgrade-strategy=eager is because there exist older versions of urllib that have a bug that breaks retries
RUN apk --no-cache add python py-pip python3 && \
    python -m pip install --upgrade --upgrade-strategy=eager pip stored>=0.0.29 && \
    python3 -m ensurepip && \
    python3 -m pip install --upgrade --upgrade-strategy=eager pip stored>=0.0.29

# we install these packages at runtime in the entrypoint, but might as well fetch the latest version into the container
RUN mkdir /pip && \
    python -m pip download -d /pip tensorflow keras h5py model-converters && \
    python3 -m pip download -d /pip tensorflow keras h5py model-converters
# note: don't install model-converters here, since it declares a dependency on tensorflow and keras (so those will be installed too)
#       which is a waste of space since they get reinstalled in the entrypoint

ADD entrypoint.sh /opt/

CMD ["/opt/entrypoint.sh"]

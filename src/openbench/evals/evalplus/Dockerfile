FROM python:3.11-slim@sha256:94c50be2dc994b873b55bc123e95e6dbade08095b3dfd790f51c34de3f08cbb7

RUN pip install --no-cache-dir numpy==2.2.6

RUN useradd --create-home --uid 1000 sandbox \
    && mkdir --parents /workspace \
    && chown sandbox:sandbox /workspace

USER sandbox
WORKDIR /workspace

CMD ["tail", "-f", "/dev/null"]

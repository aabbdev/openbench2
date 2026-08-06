FROM python:3.12-slim@sha256:646fb0bca3dd3ea1bcc6feb72c17ed16eed6e10cffc732fcc1478bd3e7f02d7b

RUN useradd --create-home --uid 1000 sandbox \
    && mkdir --parents /workspace \
    && chown sandbox:sandbox /workspace

USER sandbox
WORKDIR /workspace

CMD ["tail", "-f", "/dev/null"]

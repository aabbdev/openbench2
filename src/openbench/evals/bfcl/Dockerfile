FROM python:3.11-slim@sha256:94c50be2dc994b873b55bc123e95e6dbade08095b3dfd790f51c34de3f08cbb7

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git ca-certificates \
    && pip install --no-cache-dir --no-deps \
      "bfcl_eval @ git+https://github.com/ShishirPatil/gorilla.git@6ea57973c7a6097fd7c5915698c54c17c5b1b6c8#subdirectory=berkeley-function-call-leaderboard" \
    && pip install --no-cache-dir mpmath==1.3.0 \
    && apt-get purge --yes --auto-remove git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 sandbox \
    && mkdir --parents /workspace \
    && chown sandbox:sandbox /workspace

COPY runner.py /opt/openbench/bfcl_runner.py

USER sandbox
WORKDIR /workspace

CMD ["tail", "-f", "/dev/null"]

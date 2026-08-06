FROM python:3.11-slim@sha256:94c50be2dc994b873b55bc123e95e6dbade08095b3dfd790f51c34de3f08cbb7

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git ca-certificates \
    && pip install --no-cache-dir \
      build==1.5.0 setuptools==83.0.0 setuptools-scm==10.2.1 wheel==0.47.0 \
    && pip install --no-cache-dir --no-build-isolation --no-deps \
      "bfcl_eval @ git+https://github.com/ShishirPatil/gorilla.git@6ea57973c7a6097fd7c5915698c54c17c5b1b6c8#subdirectory=berkeley-function-call-leaderboard" \
    && pip install --no-cache-dir \
      filelock==3.20.0 mpmath==1.3.0 numpy==1.26.4 overrides==7.7.0 \
      rank-bm25==0.2.2 \
    && apt-get purge --yes --auto-remove git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 sandbox \
    && mkdir --parents /workspace \
    && chown sandbox:sandbox /workspace

COPY runner.py /opt/openbench/bfcl_runner.py

USER sandbox
WORKDIR /workspace

CMD ["tail", "-f", "/dev/null"]

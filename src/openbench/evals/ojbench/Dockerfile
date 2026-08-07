FROM python:3.11-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS builder

ARG DMOJ_REVISION=f098cd3a49a60186d1fadde5132329ec5f4f2213
ARG OJBENCH_REVISION=5e94480b1e135b98855cf5bc81213c256aff5b17

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential git libseccomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
        Cython==3.0.10 setuptools==69.5.1 wheel==0.43.0 \
    && git clone https://github.com/DMOJ/judge-server.git /src/dmoj \
    && git -C /src/dmoj checkout "${DMOJ_REVISION}" \
    && test "$(git -C /src/dmoj rev-parse HEAD)" = "${DMOJ_REVISION}" \
    && git clone https://github.com/He-Ren/OJBench.git /src/ojbench \
    && git -C /src/ojbench checkout "${OJBENCH_REVISION}" \
    && test "$(git -C /src/ojbench rev-parse HEAD)" = "${OJBENCH_REVISION}" \
    && python -m pip wheel --no-build-isolation --no-cache-dir --wheel-dir /wheels \
        filelock==3.18.0 loguru==0.7.2 PyYAML==6.0.2 setuptools==69.5.1 \
        tqdm==4.66.4 watchdog==4.0.1 termcolor==2.4.0 Pygments==2.18.0 \
        setproctitle==1.3.3 pylru==1.2.1 /src/dmoj \
    && python -m pip wheel --no-build-isolation --no-cache-dir --no-deps \
        --wheel-dir /wheels /src/ojbench

FROM python:3.11-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

RUN apt-get update \
    && apt-get install --yes --no-install-recommends g++ libseccomp2 pypy3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 judge \
    && mkdir --parents /workspace /problems /opt/ojbench \
    && chown judge:judge /workspace

COPY --from=builder /wheels /wheels
COPY --from=builder /src/ojbench/ojbench /opt/ojbench/package/ojbench
COPY --from=builder /src/ojbench/ojbench/config.yaml /opt/ojbench/config.yaml
COPY --from=builder /src/ojbench/ojbench/runtime.yaml /opt/ojbench/runtime.yaml
COPY --from=builder /src/ojbench/LICENSE /usr/share/doc/ojbench/LICENSE

RUN python -m pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

ENV OJBENCH_CONFIG_PATH=/opt/ojbench/config.yaml \
    OJBENCH_RUNTIME_PATH=/opt/ojbench/runtime.yaml \
    HOME=/home/judge \
    PYTHONPATH=/opt/ojbench/package \
    PYTHONUNBUFFERED=1

USER judge
WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]

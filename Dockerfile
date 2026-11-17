# Build from PowerShell (Windows):
#   $uvGitShas = (python .\resolve_uv_git_shas.py pyproject.docker)
#   docker build --build-arg UV_GIT_SHAS=$uvGitShas --build-arg PACKAGE_VERSION=0.1.0 -t oxdna:latest .
#
# Build from bash:
#   docker build --build-arg UV_GIT_SHAS="$(python resolve_uv_git_shas.py pyproject.docker)" --build-arg PACKAGE_VERSION=0.1.0 -t oxdna:latest .
FROM mambaorg/micromamba:latest

USER root
ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base -c conda-forge setuptools python=3.12 && \
    micromamba clean --all --yes

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/usr/local/bin:/opt/conda/bin:/root/.local/bin:$PATH"

# oxDNA v3.8.1 includes the DNA2/VMMC fix from 3.7.0. CPU only: this image
# has no CUDA toolkit. VMMC is CPU; MD uses md_backend from the system model.
RUN git clone --depth 1 --branch v3.8.1 https://github.com/lorenzo-rovigatti/oxDNA.git /tmp/oxDNA \
 && cmake -S /tmp/oxDNA -B /tmp/oxDNA/build \
        -DCUDA=OFF \
        -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /tmp/oxDNA/build -j"$(nproc)" \
 && install -m 0755 /tmp/oxDNA/build/bin/oxDNA /tmp/oxDNA/build/bin/DNAnalysis /usr/local/bin/ \
 && mkdir -p /usr/local/share/oxdna \
 && printf '3.8.1\n' > /usr/local/share/oxdna/VERSION \
 && rm -rf /tmp/oxDNA \
 && oxDNA >/tmp/oxdna-version.txt 2>&1 || true \
 && DNAnalysis >/tmp/dnanalysis-version.txt 2>&1 || true \
 && command -v oxDNA \
 && command -v DNAnalysis

ENV UV_PYTHON=/opt/conda/bin/python
ENV UV_PROJECT_ENVIRONMENT=/opt/conda
ENV PATH="/usr/local/bin:/opt/conda/bin:/root/.local/bin:$PATH"

WORKDIR /build/oxdna
COPY pyproject.docker README.md __init__.py ./
COPY nodes ./nodes
COPY models ./models
COPY util ./util
COPY part1_md ./part1_md
COPY part2_umbrella ./part2_umbrella
COPY templates ./templates
COPY testing ./testing
COPY data ./data

ARG UV_GIT_SHAS=unknown
ARG PACKAGE_VERSION=0.1.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${PACKAGE_VERSION}
ENV HATCH_VCS_PRETEND_VERSION=${PACKAGE_VERSION}

RUN echo "uv git sources ${UV_GIT_SHAS}" \
 && cp pyproject.docker pyproject.toml \
 && uv pip install --system "setuptools>=80.9.0" . \
 && python -c "import simstack, oxdna; from oxdna.nodes.workflow import fuse, umbrella_analysis; print(oxdna.__file__)"

WORKDIR /app
RUN mkdir -p /mnt /tmp/simstack && ln -sfn /tmp/simstack /root/simstack
ENTRYPOINT ["/opt/conda/bin/python", "-m", "simstack.core.run_node"]

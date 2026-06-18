# Runtime image for fiestaEM (fast EM-signal inference with JAX).
#
# Builds fiestaEM from THIS checkout (not PyPI) so the image matches the repo
# state, adds the blackjax sampler backend (used by fiesta's blackjax_smc /
# blackjax_nested_sampling samplers but not a hard install dep), and bakes the
# Bu2025_MLP kilonova surrogate so a read-only / offline worker doesn't need to
# download it at runtime.
#
# Build (linux/amd64; context = repo root):
#   docker build -f containers/Dockerfile -t fiestaem:latest .
#
# CI builds this on every push/PR to validate it (see .github/workflows/
# docker-build.yml); publishing/pushing to a registry is intentionally left
# manual.
#
# GPU variant: install the gpu extra instead — `uv pip install --system .[gpu]`
# (jax[cuda13]) — and run with a GPU available.
FROM ubuntu:24.04

ENV PIP_BREAK_SYSTEM_PACKAGES=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y python3 python3-pip git build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install uv

WORKDIR /src
COPY . /src

# fiestaEM (from this checkout) + the blackjax sampler backend.
RUN uv pip install --system --break-system-packages . blackjax

# Bake the KN surrogate (a CVMFS/read-only image can't download on a worker).
RUN python3 -c "from fiesta.surrogates.manage_surrogates import download_surrogate; download_surrogate('Bu2025_MLP')"

# Fail the build if the runtime isn't importable.
RUN python3 -c "import fiesta, blackjax, jax; print('fiestaEM + blackjax + jax', jax.__version__)"

CMD ["python3"]

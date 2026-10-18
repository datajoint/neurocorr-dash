# neurocorr-dash

A dashboard for [neurocorr](https://github.com/datajoint/neurocorr), the DataJoint pipeline that
reproduces the analysis of:

> Dimitri Yatsenko, Krešimir Josić, Alexander S. Ecker, Emmanouil Froudarakis, R. James Cotton,
> and Andreas S. Tolias (2015). **Improved estimation and interpretation of correlations in neural
> circuits.** *PLOS Computational Biology* 11(3): e1004083.
> [doi:10.1371/journal.pcbi.1004083](https://doi.org/10.1371/journal.pcbi.1004083)

The recordings and the original analysis code are in [atlab/cov-est](https://github.com/atlab/cov-est)
(MIT License, © 2014 Tolias Lab). The article is published under the Creative Commons Attribution 4.0
License.

## Pages

| Page | Shows |
|---|---|
| Overview | the sites and active cells, and how many estimates each method has computed |
| Sites | each correlation set: directions, cells, active cells, and its mean sample correlation |
| Site | a site's correlation matrix (sample or diag) and the positions of its active cells |
| Validation | each estimator's cross-validated loss relative to a reference, per site, as in the paper's Fig 4 |

The dashboard reads the pipeline's schema from the database with `dj.VirtualModule`, so it does not
need the pipeline's code installed.

## Run it

```bash
cp .env.example .env          # database host, user, password, and the schema prefix if any
uv venv && uv pip install -e .
uv run python app.py          # http://localhost:8052
```

Or in a container:

```bash
docker compose up --build     # http://localhost:8052
```

From inside a container, a database on the host is at `host.docker.internal`.

| Variable | Meaning |
|---|---|
| `DJ_HOST`, `DJ_PORT`, `DJ_BACKEND` | the database (`postgresql` or `mysql`) |
| `DJ_USER`, `DJ_PASS` | credentials with read access to the pipeline's schema |
| `DATABASE_PREFIX` | the prefix of the pipeline's schema names, if it was declared with one |
| `DASH_REQUESTS_PATHNAME_PREFIX`, `DASH_ROUTES_PATHNAME_PREFIX` | serve the app under a path, behind a proxy |

## Development

```bash
uv pip install -e . pytest ruff
pytest                        # figures from synthetic data; no database needed
ruff check . && ruff format .
```

## License

MIT; see [LICENSE](LICENSE).

### Notices

#### Mirrors

Repository:
- [Codeberg](https://codeberg.org/paveloom-university/Stellar-Astronomy-Laboratory-Workshop-S10-2022)
- [GitHub](https://github.com/paveloom-university/Stellar-Astronomy-Laboratory-Workshop-S10-2022)
- [GitLab](https://gitlab.com/paveloom-g/university/s10-2022/stellar-astronomy-laboratory-workshop)

#### Tectonic

The reports are expected to be compiled with [`tectonic`](https://tectonic-typesetting.github.io/en-US) as follows:

```bash
tectonic -X compile report.tex
```

#### Julia

This project provides [Julia](https://julialang.org) scripts. Make sure to use the project files (`Project.toml`) when running them:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. scripts/script.jl
```

Alternatively, you can use the `julia.bash` script, which starts a [daemon](https://github.com/dmolina/DaemonMode.jl) and runs scripts through it:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
./julia.bash scripts/script.jl
```

To kill the daemon run

```bash
./julia.bash kill
```

#### PGFPlotsX

For the time being, you should install the `texlive-luatex85` package to avoid [this](https://github.com/JuliaPlots/Plots.jl/issues/3319) error.

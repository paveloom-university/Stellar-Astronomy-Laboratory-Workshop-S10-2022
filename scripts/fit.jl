# This script generates catalogues of objects in disk galaxies
# with the model of uniform distribution in the symmetry plane
# and Laplace distribution on the vertical axis, and it fits
# the same model to the generated data, retrieving parameters
# and evaluating confidence intervals

println('\n', " "^4, "> Loading the packages...")

using Distributions # Probability distributions
using LaTeXStrings # LaTeX strings
using Optim # Optimization
using Plots # Plotting
using Random # Random numbers
using Roots # Finding roots
using Zygote # Derivatives

# Use the PGFPlotsX backend for plots
# (this `@eval` trick is necessary to fool `DaemonMode`)
@eval Main begin
    using Plots
    pgfplotsx()
end

# Change some of the default parameters for plots
default(fontfamily = "Computer Modern", dpi = 300, size = (400, 400), legend = :topright)

# Define the paths to output directories
CURRENT_DIR = @__DIR__
ROOT_DIR = basename(CURRENT_DIR) == "scripts" ? dirname(CURRENT_DIR) : CURRENT_DIR
PLOTS_DIR = joinpath(ROOT_DIR, "plots")
TRACES_DIR = joinpath(ROOT_DIR, "traces")

# Make sure the needed directories exist
mkpath(PLOTS_DIR)
mkpath(TRACES_DIR)

"Probability density function of the Laplace distribution"
laplace_pdf(x, μ, b) = 1 / (2 * b) * exp(-abs(x - μ) / b)

"""
Negative log-likelihood function of the model
(the model is a Laplace distribution)
"""
function nll(z::Vector{F}, θ::Vector{F})::F where {F <: Real}
    μ, b = θ
    return -sum(@. log(laplace_pdf(z, μ, b)))
end

"""
Calculate the minimum of the negative log-likelihood
function while one of the parameters is frozen
"""
function nll_frozen(
    idx::Int,
    value::F,
    z::Vector{F},
    θ₀::Vector{F},
    θₗ::Vector{F},
    θᵤ::Vector{F},
) where {F <: Real}
    # Exclude the frozen parameter from the active ones
    θ₀ = [θ₀[1:idx - 1]; θ₀[idx + 1:end]]
    θₗ = [θₗ[1:idx - 1]; θₗ[idx + 1:end]]
    θᵤ = [θᵤ[1:idx - 1]; θᵤ[idx + 1:end]]

    # Recreate the function, handling a frozen parameter
    function nll_frozen_inner(θ::Vector{F}) where {F <: Real}
        return nll(z, [θ[1:idx - 1]; value; θ[idx:end]])
    end

    # Optimize the new negative log likelihood function
    res = Optim.optimize(
        nll_frozen_inner,
        θ -> Zygote.gradient(nll_frozen_inner, θ)[1],
        θₗ,
        θᵤ,
        θ₀,
        Fminbox(LBFGS()),
        Optim.Options(
            show_trace=false,
            extended_trace=true,
            store_trace=true,
        );
        inplace=false,
    )

    return res.minimum
end

"Plot a histogram of the height sample"
function plot_histogram_height(z, μₜ, bₜ, μ, b, dir)
    # Plot the histogram of the height sample
    histogram(
        z;
        label = "",
        title = "",
        xlabel = L"z",
        ylabel = L"f(z)",
        normalize = true,
        yminorticks = 5,
    )

    # Plot the probability functions over the histogram
    plot!((x) -> laplace_pdf(x, μₜ, bₜ), label = "true", lw = 1.5)
    plot!((x) -> laplace_pdf(x, μ, b), label = "fit", lw = 1.5)

    # Save the figure as PDF and PNG
    savefig(joinpath(dir, "Histogram Z.pdf"))
end

"Create the objective function profiles for all of the parameters"
function profiles(
    z::Vector{F},
    L₀::F,
    θ::Vector{F},
    θ₀::Vector{F},
    θₗ::Vector{F},
    θᵤ::Vector{F},
    labels::Vector{Vector{String}},
    plots_dir::String,
) where {F <: Real}
    for (i, x₀, (label, tex_label)) in zip(1:length(θ), θ, labels)
        println(" "^6, "> Evaluating the confidence intervals of the $label parameter...")

        # Create an alias function for the frozen parameter
        nll_frozen_with(x) = nll_frozen(i, x, z, θ₀, θₗ, θᵤ)

        # Define a function whose roots define confidence intervals
        confidence_intervals(x) = nll_frozen_with(x) - L₀ - 0.5

        # Evaluate the confidence intervals
        r1, r2 = find_zeros(confidence_intervals, x₀ - 0.2, x₀ + 0.2)
        σr1, σr2 = x₀ - r1, r2 - x₀

        # Compute the mean of the sigmas
        σr = (σr1 + σr2) / 2

        # Print the results
        println(
            '\n',
            " "^8, "$label = $(x₀)\n",
            " "^8, "$(label)₋ₕ = $(r1)\n",
            " "^8, "$(label)₊h = $(r2)\n",
            " "^8, "σ$(label)₋ₕ = $(σr1)\n",
            " "^8, "σ$(label)₊ₕ = $(σr2)\n",
        )

        println(" "^6, "> Plotting the objective function profile for the $label parameter...")

        # Plot the profile
        plot(nll_frozen_with, r1 - σr1, r2 + σr2; label="", xlabel=latexstring(tex_label), ylabel=latexstring("L_p($tex_label)"))

        # Add vertical lines to the plot
        plot!([r1, r1], [L₀ - 0.2, L₀ + 0.5]; label="", linestyle=:dash)
        plot!([x₀, x₀], [L₀ - 0.2, L₀]; label="", linestyle=:dash)
        plot!([r2, r2], [L₀ - 0.2, L₀ + 0.5]; label="", linestyle=:dash)

        # Add points to the plot
        scatter!([r1], [L₀ + 0.5]; label="")
        scatter!([x₀], [L₀]; label="")
        scatter!([r2,], [L₀ + 0.5]; label="")

        # Add horizontal lines to the plot
        hline!([L₀ + 0.5]; label="", linestyle=:dash)
        hline!([L₀]; label="", linestyle=:dash)

        # Add annotations to the plot
        rounded = round.([r1, x₀, r2]; digits=2)
        annotate!(
            [
                (r1 + σr * 0.2, L₀ - 0.1, text("$(rounded[1])", 8, "Computer Modern")),
                (x₀, L₀ + 0.1, text("$(rounded[2])", 8, "Computer Modern")),
                (r2 + σr * 0.2, L₀ - 0.1, text("$(rounded[3])", 8, "Computer Modern")),
                (r1 - σr * 0.22, L₀ - 0.1, text("$(round(rounded[1] - rounded[2]; digits=2))", 8, "Computer Modern")),
                (r2 - σr * 0.25, L₀ - 0.103, text("+$(round(rounded[3] - rounded[2]; digits=2))", 8, "Computer Modern")),
                (x₀ - σr1 * 1.80, L₀ + 0.06, text(L"L_0", 8, "Computer Modern")),
                (x₀ - σr1 * (1.80 - 0.21), L₀ + 0.5 + 0.06, text(L"L_0 + 1/2", 8, "Computer Modern")),
            ]
        )

        # Save the figure
        savefig(joinpath(plots_dir, "$label profile.pdf"))
    end
end

# Define a set of initial parameters
θ₀ = [0.0, 0.5]

# Define the lower and upper boundaries, respectively
θₗ = [-Inf, 0]
θᵤ = [Inf, Inf]

"""
Generate a catalogue for a disk with uniform distribution in
the symmetry plane and Laplace distribution on the vertical axis
"""
function fit_uniform_disk(seed; N::Int = 1000, μₜ::F = 2.0, bₜ::F = 0.9) where {F <: Real}
    println(" "^4, "> Fitting a sample (seed = $seed, N = $N) of the uniform disk distribution...")

    # Set a seed for the random generator
    Random.seed!(seed)

    # Define the output directories
    name = "$seed, $N"
    plots_dir = joinpath(PLOTS_DIR, "fit_uniform_disk", name)
    traces_dir = joinpath(TRACES_DIR, name)

    # Make sure the needed directories exists
    mkpath(plots_dir)
    mkpath(traces_dir)

    # Generate a sample of heights from the Laplace distribution
    z = rand(Laplace(μₜ, bₜ), N)

    # Link the negative log-likelihood function with data
    nll_inner(θ) = nll(z, θ)

    # Optimize the negative log likelihood function
    res = Optim.optimize(
        nll_inner,
        θ -> Zygote.gradient(nll_inner, θ)[1],
        θₗ,
        θᵤ,
        θ₀,
        Fminbox(LBFGS()),
        Optim.Options(
            show_trace=false,
            extended_trace=true,
            store_trace=true,
        );
        inplace=false,
    )

    # Unpack the results
    θ = res.minimizer
    μ, b = res.minimizer
    L₀ = res.minimum

    # Save the trace and the results
    open(joinpath(traces_dir, "fit.trace"), "w") do io
        println(io, res.trace)
        println(
            io,
            " * Parameters:\n",
            "    μ = $(μ)\n",
            "    b = $(b)\n",
        )
        show(io, res)
    end

    # Print results
    println(
        '\n',
        " "^6, "μₜ = $(μₜ)\n",
        " "^6, "bₜ = $(bₜ)\n",
        " "^6, "μ = $(μ)\n",
        " "^6, "b = $(b)\n",
    )

    println(" "^6, "> Plotting the histogram...")

    # Plot the histogram of the height sample
    plot_histogram_height(z, μₜ, bₜ, μ, b, plots_dir)

    # Create the objective function profiles for all of the parameters
    profiles(z, L₀, θ, θ₀, θₗ, θᵤ, [["μ", "\\mu"], ["b", "b"]], plots_dir)

    print("\n")
    sleep(0.5)
end

# Plot samples from the uniform disk distribution
fit_uniform_disk(1, N=100)
fit_uniform_disk(1)
fit_uniform_disk(1, N=10000)
fit_uniform_disk(2)
fit_uniform_disk(3)
fit_uniform_disk(4)
fit_uniform_disk(5)

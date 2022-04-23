# This script generates catalogues
# of objects in disk galaxies

println('\n', " "^4, "> Loading the packages...")

using Distributions
using LaTeXStrings
using Plots
using Random

# Use the PGFPlotsX backend for plots
# (this `@eval` trick is necessary to fool `DaemonMode`)
@eval Main begin
    using Plots
    pgfplotsx()
end

# Change some of the default parameters for plots
default(fontfamily = "Computer Modern", dpi = 300)

# Define the paths to output directories
CURRENT_DIR = @__DIR__
ROOT_DIR = basename(CURRENT_DIR) == "scripts" ? dirname(CURRENT_DIR) : CURRENT_DIR
PLOTS_DIR = joinpath(ROOT_DIR, "plots")

# Make sure the needed directories exist
mkpath(PLOTS_DIR)

# Define the number of objects in a catalogue
N = 1000

# Define probability density functions for some of the distributions
normal_pdf(x, μ, σ) = 1 / (σ * √(2 * π)) * exp(-0.5 * ((x - μ) / σ)^2)
uniform_pdf(x, a, b) = a < x < b ? 1 / (b - a) : 0
uniform_radius_pdf(x, r) = 0 < x < r ? 2 * x / r^2 : 0
exponential_pdf(x, b) = x >= 0 ? 1 / b * exp(-x / b) : 0
exponential_radius_pdf(r, b) = r >= 0 ? 2 * π / b * r * exp(-π / b * r^2) : 0
laplace_pdf(x, μ, b) = 1 / (2 * b) * exp(-abs(x - μ) / b)

"Return a range of ticks that are dividable by π / denom, along with the labels"
function piticks(start, stop, denom)
    # Compute the integer bounds for a range
    # of values dividable by π / denom
    a = Int(cld(start, π / denom))
    b = Int(fld(stop, π / denom))
    # Compute the range
    ticks = range(a * π / denom, b * π / denom; step = π / denom)
    # Define the labels for each tick
    ticklabels = piticklabel.((a:b) .// denom)
    return ticks, ticklabels
end

"Return a label for a rational number which is a multiple of π"
function piticklabel(x::Rational)
    # Return zero if it's a zero
    iszero(x) && return L"0"
    # Define the sign of the rational
    sign = x < 0 ? "-" : ""
    # Get the numerator and the denominator
    n, d = abs(numerator(x)), denominator(x)
    # Add a multiple to π if necessary
    N = n == 1 ? "" : repr(n)
    # Return the label
    d == 1 ? L"%$sign%$N\pi" : L"%$sign\frac{%$N\pi}{%$d}"
end

function plot_distribution(sample::AbstractVector, pdf::Function, dir::AbstractString)
    println(" "^4, "> Plotting the $dir distribution...")

    # Define the output directory
    output_dir = joinpath(PLOTS_DIR, dir)

    # Make sure the directory exists
    mkpath(output_dir)

    # Plot the histogram of the distribution
    histogram(
        sample;
        label = "",
        title = "",
        xlabel = L"x",
        ylabel = L"f(x)",
        normalize = true,
        size = (400, 400),
        yminorticks = 5,
    )

    # Plot the probability function over the histogram
    plot!(pdf, label = "", lw = 1.5)

    # Save the figure as PDF and PNG
    savefig(joinpath(output_dir, "Histogram.pdf"))
    savefig(joinpath(output_dir, "Histogram.png"))
end

# Set a seed for the random number generator
Random.seed!(2)

# Plot a sample from the normal distribution with μ=1 and σ=2
plot_distribution(rand(Normal(1, 2), N), (x) -> normal_pdf(x, 1, 2), "normal")

# Plot a sample from the uniform distribution with a=1 and b=2
plot_distribution(rand(Uniform(1, 2), N), (x) -> uniform_pdf(x, 1, 2), "uniform")

# Plot a sample from the exponential distribution with b=0.5
plot_distribution(rand(Exponential(0.5), N), (x) -> exponential_pdf(x, 0.5), "exponential")

# Plot a sample from the Laplace distribution with μ = 0 and b=0.5
plot_distribution(rand(Laplace(0, 0.5), N), (x) -> laplace_pdf(x, 0, 0.5), "laplace")

"Plot a histogram of the radius sample"
function plot_histogram_radius(r, pdf::Function, dir)
    # Plot the histogram of the radius sample
    histogram(
        r;
        label = "",
        title = "",
        xlabel = L"r",
        ylabel = L"f(r)",
        normalize = true,
        size = (400, 400),
        yminorticks = 5,
    )

    # Plot the probability function over the histogram
    plot!(pdf, label = "", lw = 1.5)

    # Save the figure as PDF and PNG
    savefig(joinpath(dir, "Histogram R.pdf"))
    savefig(joinpath(dir, "Histogram R.png"))

end

"Plot a histogram of the angle sample"
function plot_histogram_angle(θ, dir)
    # Plot the histogram of the angle sample
    histogram(
        θ;
        label = "",
        title = "",
        xlabel = L"\theta",
        ylabel = L"f(\theta)",
        normalize = true,
        bins = range(0, stop = 2 * π, step = π / 4),
        size = (400, 400),
        xtick = piticks(0, 2 * π, 4),
        yminorticks = 5,
    )

    # Plot the probability function over the histogram
    plot!((x) -> uniform_pdf(x, 0, 2 * π), label = "", lw = 1.5)

    # Save the figure as PDF and PNG
    savefig(joinpath(dir, "Histogram θ.pdf"))
    savefig(joinpath(dir, "Histogram θ.png"))
end

"Plot a histogram of the height sample"
function plot_histogram_height(z, b, dir)
    # Plot the histogram of the height sample
    histogram(
        z;
        label = "",
        title = "",
        xlabel = L"z",
        ylabel = L"f(z)",
        normalize = true,
        size = (400, 400),
        yminorticks = 5,
    )

    # Plot the probability function over the histogram
    plot!((x) -> laplace_pdf(x, 0, b), label = "", lw = 1.5)

    # Save the figure as PDF and PNG
    savefig(joinpath(dir, "Histogram Z.pdf"))
    savefig(joinpath(dir, "Histogram Z.png"))
end

"Plot a heatmap of the symmetry plane"
function plot_heatmap_symmetry_plane(r, θ, dir; rₘ = 0.)
    # Switch to GR for heatmaps
    # (see https://github.com/JuliaPlots/Plots.jl/issues/3266)
    gr()

    # Plot the heatmap of the symmetry plane
    histogram2d(
        θ,
        r;
        proj = :polar,
        label = "",
        title = "",
        xlabel = L"\theta",
        yaxis = false,
        normalize = :probability,
        show_empty_bins = true,
        size = (525, 400),
        right_margin = 7.5 * Plots.mm,
        colorbar = :best,
        legend = false,
        nbins = 15,
    )

    # Add a radius error to show the scale
    θ₀ = 59 * π / 32
    if rₘ == 0
      rₘ = maximum(r)
    end
    xₘ = 1.1 * cos(θ₀ + π / 64)
    yₘ = 1.1 * sin(θ₀ + π / 64)
    plot!([0, θ₀], [0, rₘ], lw = 1.5, color = palette(:default)[3], arrow = true)
    annotate!(xₘ, yₘ, ("$(round(rₘ, digits = 4))", 10))

    # Save the figure as PNG
    savefig(joinpath(dir, "Heatmap Symmetry Plane.png"))

    # Switch back to PGFPlotsX
    pgfplotsx()
end

"Plot a heatmap of the vertical cut"
function plot_heatmap_vertical_cut(r, z, dir)
    # Plot the heatmap of a vertical cut
    histogram2d(
        r,
        z;
        label = "",
        title = "",
        xlabel = L"r",
        ylabel = L"z",
        normalize = :probability,
        size = (400, 400),
        nbins = 40,
    )

    # Save the figure as PDF and PNG
    savefig(joinpath(dir, "Heatmap Vertical Cut.pdf"))
    savefig(joinpath(dir, "Heatmap Vertical Cut.png"))
end

"""
Generate a catalogue for a disk with uniform distribution in
the symmetry plane and Laplace distribution on the vertical axis
"""
function plot_uniform_disk(seed, R::F = 2.0, b::F = 0.5) where {F <: Real}
    println(" "^4, "> Plotting the uniform disk (seed = $seed) distribution...")

    # Set a seed for the random generator
    Random.seed!(seed)

    # Define the output directory
    output_dir = joinpath(PLOTS_DIR, "uniform_disk", "$seed")

    # Make sure the directory exists
    mkpath(output_dir)

    # Generate samples from a uniformly distributed
    # circle with exponentially distributed heights
    # (see https://stackoverflow.com/a/50746409 and
    # https://stats.stackexchange.com/a/378957 for
    # explanation)
    r = sqrt.(rand(Uniform(0, 1), N)) * R
    θ = rand(Uniform(0, 2 * π), N)
    z = rand(Laplace(0, b), N)

    # Plot the histogram of the radius sample
    plot_histogram_radius(r, (x) -> uniform_radius_pdf(x, R), output_dir)

    # Plot the histogram of the angle sample
    plot_histogram_angle(θ, output_dir)

    # Plot the histogram of the height sample
    plot_histogram_height(z, b, output_dir)

    # Plot the heatmap of the symmetry plane
    plot_heatmap_symmetry_plane(r, θ, output_dir, rₘ = R)

    # Plot the heatmap of the vertical cut
    plot_heatmap_vertical_cut(r, z, output_dir)
end

# Plot samples from the uniform disk distribution
plot_uniform_disk(2)

"""
Generate a catalogue for a disk with exponential distribution in
the symmetry plane and Laplace distribution on the vertical axis
"""
function plot_exponential_disk(seed, b₁::F = 2.6, b₂::F = 0.9) where {F <: Real}
    println(" "^4, "> Plotting the exponential disk (seed = $seed) distribution...")

    # Set a seed for the random generator
    Random.seed!(seed)

    # Define the output directory
    output_dir = joinpath(PLOTS_DIR, "exponential_disk", "$seed")

    # Make sure the directory exists
    mkpath(output_dir)

    # Generate samples from an exponentially distributed
    # circle with exponentially distributed heights
    # (see https://stackoverflow.com/a/50746409 and
    # https://stats.stackexchange.com/a/378957 for
    # explanation)
    r = sqrt.(-b₁ / π .* log.(1 .- rand(Uniform(0, 1), N)))
    θ = rand(Uniform(0, 2 * π), N)
    z = rand(Laplace(0, b₂), N)

    # Plot the histogram of the radius sample
    plot_histogram_radius(r, (x) -> exponential_radius_pdf(x, b₁), output_dir)

    # Save the figure as PDF and PNG
    savefig(joinpath(output_dir, "Histogram R.pdf"))
    savefig(joinpath(output_dir, "Histogram R.png"))

    # Plot the histogram of the angle sample
    plot_histogram_angle(θ, output_dir)

    # Plot the histogram of the height sample
    plot_histogram_height(z, b₂, output_dir)

    # Plot the heatmap of the symmetry plane
    plot_heatmap_symmetry_plane(r, θ, output_dir)

    # Plot the heatmap of the vertical cut
    plot_heatmap_vertical_cut(r, z, output_dir)
end

# Plot samples from the exponential disk distribution
plot_exponential_disk(1)
plot_exponential_disk(2)
plot_exponential_disk(3)

println()

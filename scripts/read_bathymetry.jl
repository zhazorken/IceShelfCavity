using DataFrames
using Oceananigans.Units
using CSV

function get_bathymetry_from(filename_bathymetry)
    # Find the CSV whether files are laid out as scripts/ + data/ (cluster repo) or flat.
    candidates = (joinpath(@__DIR__, "..", "data", filename_bathymetry),
                  joinpath(@__DIR__, "data", filename_bathymetry),
                  joinpath(@__DIR__, filename_bathymetry))
    idx = findfirst(isfile, candidates)
    idx === nothing && error("Could not find bathymetry CSV '$filename_bathymetry'. Looked in:\n  " *
                             join(candidates, "\n  "))

    df = CSV.read(candidates[idx], DataFrame)

    x = df[!, "x_km"] * 1e3meters
    top = df[!, "topo"]     # z(x) of the ice-shelf base (ice underside)
    bottom = df[!, "bathy"] # z(x) of the sea floor

    return (; x, bottom, top)
end

filename_bathymetry = "pineislandbath.csv"

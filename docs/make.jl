using Documenter
using FileWatching.Pidfile

push!(LOAD_PATH, "../src")
using RungeKuttaToolKit
using RungeKuttaToolKit.RKCost
using RungeKuttaToolKit.RKParameterization

const WEBSITE_DIR = joinpath(homedir(),
    "Documents", "GitHub", "dzhang314.github.com")

trymkpidlock(".docslock") do
    makedocs(sitename="RungeKuttaToolKit.jl", format=Documenter.HTML(;
        mathengine=Documenter.KaTeX(Dict(:macros => Dict(
            "\\N" => "\\mathbb{N}",
            "\\R" => "\\mathbb{R}",
            "\\vf" => "\\mathbf{f}",
            "\\vy" => "\\mathbf{y}",
            "\\hash" => "\\mathbin{\\#}",
        )))
    ))
    if isdir(WEBSITE_DIR)
        cp(joinpath("docs", "build"),
            joinpath(WEBSITE_DIR, "RungeKuttaToolKit.jl");
            force=true)
    end
end

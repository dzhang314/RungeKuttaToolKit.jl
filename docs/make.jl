using Documenter
using FileWatching.Pidfile

push!(LOAD_PATH, "../src")
using RungeKuttaToolKit

const WEBSITE_DIR = joinpath(homedir(),
    "Documents", "GitHub", "dzhang314.github.com")

trymkpidlock(".docslock") do
    makedocs(sitename="RungeKuttaToolKit.jl Documentation")
    if isdir(WEBSITE_DIR)
        cp(joinpath("docs", "build"),
            joinpath(WEBSITE_DIR, "RungeKuttaToolKit.jl");
            force=true)
    end
end

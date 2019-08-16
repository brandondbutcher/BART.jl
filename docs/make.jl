using Documenter, BART

makedocs(;
    modules=[BART],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/brandondbutcher/BART.jl/blob/{commit}{path}#L{line}",
    sitename="BART.jl",
    authors="Brandon Butcher, University of Iowa",
    assets=String[],
)

deploydocs(;
    repo="github.com/brandondbutcher/BART.jl",
)

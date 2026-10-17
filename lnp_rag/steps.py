"""Preset LNP delivery-journey steps. Edit or add entries to match your review's outline."""

STEPS = {
    "uptake_pkpd": {
        "name": "Quantitative cellular uptake for PK/PD modeling",
        "description": (
            "quantitative, time-course or dose-dependent experimental data on LNP cellular uptake, including "
            "uptake concentrations, intracellular accumulation and endosomal escape, presented in accessible "
            "figures, tables or supplementary datasets"
        ),
        "guidance": (
            "The paper is useful only if its own experiments quantify LNP or cargo uptake in cells (e.g. MFI, % "
            "positive cells, amount per cell, colocalization, escape efficiency) at two or more time points OR two "
            "or more doses/concentrations, so the data could be digitized to fit a PK/PD model. The paper's topic "
            "or therapeutic application does not matter; uptake experiments are often a supporting figure in an "
            "application paper and still count. A single-time, single-dose image without quantification is "
            "'borderline' at best."
        ),
    },
    "protein_corona": {
        "name": "Protein corona formation",
        "description": "adsorption of plasma/serum proteins such as ApoE onto LNPs in biological fluids and its effect on LNP fate",
        "guidance": "",
    },
    "circulation_biodistribution": {
        "name": "Circulation and biodistribution",
        "description": "blood circulation, pharmacokinetics and organ/tissue distribution of LNPs after administration",
        "guidance": "",
    },
    "cellular_uptake": {
        "name": "Cellular uptake",
        "description": "binding to and internalization of LNPs by target cells, including endocytic pathways and uptake kinetics",
        "guidance": "",
    },
    "cellular_uptake_in_vitro": {
        "name": "Cellular uptake (in vitro)",
        "description": "binding to and internalization of LNPs by cells in vitro, including endocytic pathways and uptake kinetics",
        "guidance": (
            "This section of the review covers in vitro work only. The paper must report cellular uptake "
            "experiments in cultured cells, cell lines, primary cells, organoids or ex vivo tissue; papers "
            "whose uptake data come only from living animals or patients are 'exclude'."
        ),
    },
    "endosomal_escape": {
        "name": "Endosomal escape",
        "description": "release of LNP cargo from endosomes/lysosomes into the cytosol, including endosomal trafficking and escape efficiency",
        "guidance": "",
    },
    "cargo_expression": {
        "name": "Cargo release and expression",
        "description": "intracellular cargo release and resulting protein expression or gene silencing over time",
        "guidance": "",
    },
    "clearance": {
        "name": "Degradation and clearance",
        "description": "metabolism, degradation and elimination of LNP components and cargo from cells and the body",
        "guidance": "",
    },
}

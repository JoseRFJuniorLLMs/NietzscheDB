#!/usr/bin/env python3
"""
Insert Science & Engineering Galaxies into NietzscheDB.

Chemistry, Metallurgy, Genomics, Pharmacology, Engineering,
Earth Sciences, Oceanography, Medicine.

Usage:
  python scripts/insert_science_galaxies.py [--host HOST:PORT] [--collection NAME]
"""

import grpc
import json
import uuid
import math
import hashlib
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))
from grpc_tools import protoc
import importlib


def ensure_proto_compiled(repo_root=None):
    if repo_root is None:
        for candidate in [
            os.path.join(os.path.dirname(__file__), '..'),
            '/home/web2a/NietzscheDB',
            os.path.expanduser('~/NietzscheDB'),
        ]:
            if os.path.isdir(os.path.join(candidate, 'crates', 'nietzsche-api', 'proto')):
                repo_root = candidate
                break
        if repo_root is None:
            raise RuntimeError("Cannot find NietzscheDB repo root.")
    proto_dir = os.path.join(repo_root, 'crates', 'nietzsche-api', 'proto')
    out_dir = '/tmp/_nietzsche_gen'
    os.makedirs(out_dir, exist_ok=True)
    pb2_file = os.path.join(out_dir, 'nietzsche_pb2.py')
    if not os.path.exists(pb2_file):
        print(f"[proto] Compiling nietzsche.proto...")
        protoc.main(['grpc_tools.protoc', f'-I{proto_dir}',
            f'--python_out={out_dir}', f'--grpc_python_out={out_dir}', 'nietzsche.proto'])
        with open(os.path.join(out_dir, '__init__.py'), 'w') as f: f.write('')
    sys.path.insert(0, out_dir)
    return importlib.import_module('nietzsche_pb2'), importlib.import_module('nietzsche_pb2_grpc')


GALAXIES = {

    # ─── GALAXY 1: CHEMISTRY ───────────────────────────────────────────
    "Chemistry": {
        "depth": 0.06,
        "stars": {
            "General & Physical Chemistry": {
                "depth": 0.16,
                "concepts": [
                    "Periodic Table Elements", "Atomic Theory Dalton",
                    "Electron Configuration", "Quantum Numbers Chemistry",
                    "Chemical Bonding Covalent Ionic", "Molecular Orbital Theory",
                    "VSEPR Theory Shape", "Hybridization sp sp2 sp3",
                    "Electronegativity Pauling", "Lewis Structures",
                    "Thermochemistry Enthalpy", "Hess Law",
                    "Chemical Equilibrium Le Chatelier", "Equilibrium Constants",
                    "Acids and Bases pH", "Buffer Solutions",
                    "Oxidation Reduction Reactions", "Electrochemistry Nernst",
                    "Chemical Kinetics Rates", "Activation Energy Arrhenius",
                    "Catalysis Homogeneous Heterogeneous", "Gibbs Free Energy Chemistry",
                ]
            },
            "Organic Chemistry": {
                "depth": 0.20,
                "concepts": [
                    "Carbon Chemistry Backbone", "Functional Groups Organic",
                    "Alkanes Alkenes Alkynes", "Aromatic Compounds Benzene",
                    "Stereochemistry Chirality", "Enantiomers Diastereomers",
                    "SN1 SN2 Reaction Mechanisms", "Elimination Reactions E1 E2",
                    "Electrophilic Addition", "Nucleophilic Substitution",
                    "Grignard Reagents", "Wittig Reaction",
                    "Aldol Condensation", "Diels-Alder Reaction",
                    "Cross-Coupling Reactions Suzuki", "Organometallic Chemistry",
                    "Retrosynthetic Analysis Corey", "Total Synthesis Natural Products",
                    "Polymer Chemistry Polymerization", "Supramolecular Chemistry",
                    "Click Chemistry Sharpless", "Green Chemistry Principles",
                ]
            },
            "Inorganic Chemistry": {
                "depth": 0.22,
                "concepts": [
                    "Coordination Chemistry Werner", "Ligand Field Theory",
                    "Crystal Field Theory", "d-Orbital Splitting",
                    "Transition Metal Complexes", "Chelation Effect",
                    "Organometallic Catalysis", "Homogeneous Catalysis",
                    "Bioinorganic Chemistry", "Hemoglobin Iron",
                    "Lanthanides Actinides", "Rare Earth Elements",
                    "Solid State Chemistry", "Crystal Structures",
                    "Zeolites Molecular Sieves", "Metal-Organic Frameworks MOF",
                    "Cluster Chemistry", "Main Group Chemistry",
                ]
            },
            "Analytical Chemistry": {
                "depth": 0.24,
                "concepts": [
                    "Spectroscopy UV-Vis", "Infrared Spectroscopy IR",
                    "Nuclear Magnetic Resonance NMR", "Mass Spectrometry",
                    "X-Ray Crystallography", "Electron Microscopy Chemistry",
                    "Chromatography HPLC GC", "Electrophoresis",
                    "Titration Methods", "Gravimetric Analysis",
                    "Electroanalytical Methods", "Atomic Absorption Spectroscopy",
                    "Fluorescence Spectroscopy", "Raman Spectroscopy",
                    "Surface Analysis XPS", "Thermal Analysis DSC TGA",
                ]
            },
            "Biochemistry": {
                "depth": 0.26,
                "concepts": [
                    "Amino Acids Twenty", "Protein Structure Primary Secondary Tertiary",
                    "Enzyme Mechanisms Lock Key", "Allosteric Regulation",
                    "Nucleic Acids DNA RNA Structure", "Watson Crick Base Pairing",
                    "Lipids Membranes Bilayer", "Carbohydrates Sugars",
                    "Metabolic Pathways", "Citric Acid Cycle Krebs",
                    "Electron Transport Chain", "ATP Production Mechanism",
                    "Photosynthesis Light Dark Reactions", "Calvin Cycle",
                    "Hormone Signaling Biochemistry", "Signal Cascades",
                    "Cofactors Coenzymes Vitamins", "Iron-Sulfur Clusters",
                ]
            },
            "Nuclear & Radiochemistry": {
                "depth": 0.28,
                "concepts": [
                    "Radioactive Decay Alpha Beta Gamma", "Half-Life",
                    "Nuclear Fission", "Nuclear Fusion",
                    "Isotopes and Isotope Effects", "Radioisotope Dating Carbon-14",
                    "Nuclear Medicine PET Scan", "Radiation Therapy",
                    "Chernobyl Fukushima", "Nuclear Waste Management",
                    "Transuranium Elements", "Nuclear Chain Reaction",
                    "Manhattan Project Chemistry", "Curie Radioactivity",
                    "Neutron Activation Analysis", "Nuclear Reactor Chemistry",
                ]
            },
        }
    },

    # ─── GALAXY 2: METALLURGY & MATERIALS SCIENCE ──────────────────────
    "Metallurgy & Materials": {
        "depth": 0.07,
        "stars": {
            "Physical Metallurgy": {
                "depth": 0.17,
                "concepts": [
                    "Crystal Structure BCC FCC HCP", "Unit Cell Parameters",
                    "Dislocations Edge Screw", "Slip Systems",
                    "Grain Boundaries", "Recrystallization Annealing",
                    "Solidification Nucleation Growth", "Dendrite Formation",
                    "Phase Diagrams Binary", "Eutectic Systems",
                    "Peritectic Reaction", "Eutectoid Transformation",
                    "Diffusion in Metals Fick Laws", "Vacancy Mechanism",
                    "Work Hardening Strain Hardening", "Recovery Recrystallization",
                    "Precipitation Hardening", "Age Hardening",
                    "Mechanical Properties Testing", "Tensile Test Stress-Strain",
                    "Hardness Testing Vickers Rockwell", "Fatigue Failure",
                    "Creep Deformation", "Fracture Mechanics Griffith",
                ]
            },
            "Steel & Iron Alloys": {
                "depth": 0.20,
                "concepts": [
                    "Iron-Carbon Phase Diagram", "Austenite Ferrite Cementite",
                    "Pearlite Microstructure", "Martensite Transformation",
                    "Bainite Formation", "TTT Diagrams",
                    "CCT Diagrams", "Quenching and Tempering",
                    "Stainless Steel Chromium", "Austenitic Stainless 304 316",
                    "Martensitic Stainless", "Duplex Stainless Steel",
                    "High-Speed Steel Tool", "HSLA Steel",
                    "Weathering Steel Corten", "Cast Iron Types",
                    "Damascus Steel Wootz", "Bessemer Process",
                    "Electric Arc Furnace", "Continuous Casting",
                ]
            },
            "Non-Ferrous Alloys": {
                "depth": 0.22,
                "concepts": [
                    "Aluminum Alloys 2xxx 6xxx 7xxx", "Aluminum Precipitation Hardening",
                    "Copper Alloys Bronze Brass", "Beryllium Copper",
                    "Titanium Alloys Ti-6Al-4V", "Titanium Alpha Beta",
                    "Nickel Superalloys Inconel", "Turbine Blade Materials",
                    "Magnesium Alloys Lightweight", "Zinc Alloys Die Casting",
                    "Tungsten Carbide Hard Metals", "Cobalt Chrome Alloys",
                    "Shape Memory Alloys Nitinol", "Superelasticity",
                    "Refractory Metals W Mo Ta", "Precious Metals Pt Au Ag",
                    "Lead-Free Solders", "Tin Alloys",
                ]
            },
            "Advanced Materials": {
                "depth": 0.25,
                "concepts": [
                    "Carbon Fiber Composites", "Glass Fiber Reinforced",
                    "Kevlar Aramid Fibers", "Metal Matrix Composites",
                    "Ceramic Matrix Composites", "Silicon Carbide SiC",
                    "Graphene Materials", "Carbon Nanotubes",
                    "Aerogels", "Metamaterials Engineering",
                    "Biomaterials Implants", "Biodegradable Polymers",
                    "Smart Materials Piezoelectric", "Magnetostrictive Materials",
                    "Thermoelectric Materials", "Photovoltaic Materials",
                    "High Entropy Alloys", "Bulk Metallic Glasses",
                    "Additive Manufacturing Materials", "Powder Metallurgy",
                ]
            },
            "Corrosion & Surface Engineering": {
                "depth": 0.27,
                "concepts": [
                    "Corrosion Electrochemical", "Galvanic Corrosion",
                    "Pitting Corrosion", "Crevice Corrosion",
                    "Stress Corrosion Cracking", "Intergranular Corrosion",
                    "Cathodic Protection", "Anodizing Process",
                    "Electroplating", "Hot-Dip Galvanizing",
                    "PVD Physical Vapor Deposition", "CVD Chemical Vapor Deposition",
                    "Thermal Spray Coatings", "Plasma Nitriding",
                    "Surface Hardening Carburizing", "Shot Peening",
                ]
            },
        }
    },

    # ─── GALAXY 3: GENOMICS & GENETICS ─────────────────────────────────
    "Genomics & Genetics": {
        "depth": 0.08,
        "stars": {
            "Human Genome": {
                "depth": 0.18,
                "concepts": [
                    "Human Genome 3 Billion Base Pairs", "Chromosome 23 Pairs",
                    "Exome Coding Regions", "Introns Non-Coding",
                    "Gene Density Variation", "Telomeres Aging",
                    "Centromere Structure", "Sex Chromosomes X Y",
                    "Mitochondrial DNA Maternal", "Human Genome Project Completion",
                    "1000 Genomes Project", "Genome-Wide Association Studies GWAS",
                    "Single Nucleotide Polymorphism SNP", "Copy Number Variation",
                    "Structural Variants", "Mobile Elements Alu LINE",
                    "Pseudogenes", "Gene Families Clusters",
                    "HLA Complex Immune", "ABO Blood Group Genetics",
                ]
            },
            "Epigenetics & Gene Regulation": {
                "depth": 0.21,
                "concepts": [
                    "DNA Methylation CpG", "Histone Modifications",
                    "Histone Acetylation Deacetylation", "Chromatin Remodeling",
                    "Euchromatin Heterochromatin", "X-Inactivation Barr Body",
                    "Genomic Imprinting", "Transgenerational Epigenetics",
                    "Non-Coding RNA lncRNA", "MicroRNA miRNA Regulation",
                    "Enhancers Silencers Regulatory", "Promoter CpG Islands",
                    "Transcription Factors Binding", "Pioneer Factors",
                    "Epigenetic Clock Horvath", "Environmental Epigenetics",
                    "Cancer Epigenetics", "Epigenome Mapping ENCODE",
                ]
            },
            "CRISPR & Gene Editing": {
                "depth": 0.23,
                "concepts": [
                    "CRISPR-Cas9 Mechanism", "Guide RNA sgRNA",
                    "PAM Sequence Recognition", "Double-Strand Break Repair",
                    "Homology-Directed Repair HDR", "Non-Homologous End Joining NHEJ",
                    "Base Editing ABE CBE", "Prime Editing",
                    "CRISPR Screening Libraries", "Gene Drive Technology",
                    "CAR-T Cell Therapy", "In Vivo Gene Therapy",
                    "Doudna Charpentier Nobel", "CRISPR Ethics Debate",
                    "He Jiankui Controversy", "Germline Editing",
                    "CRISPR Diagnostics SHERLOCK", "Agricultural Gene Editing",
                ]
            },
            "Evolutionary Genomics": {
                "depth": 0.25,
                "concepts": [
                    "Comparative Genomics", "Synteny Conservation",
                    "Phylogenomics", "Molecular Phylogenetics",
                    "Horizontal Gene Transfer Bacteria", "Endosymbiotic Gene Transfer",
                    "Whole Genome Duplication", "Gene Duplication Divergence",
                    "Positive Selection Ka/Ks", "Neutral Evolution",
                    "Molecular Clock Calibration", "Ancient DNA Paleogenomics",
                    "Neanderthal Genome Paabo", "Denisovan Genome",
                    "Out of Africa Migration", "Population Bottleneck",
                    "Founder Effect Genetics", "Genetic Drift Populations",
                ]
            },
            "Genomic Medicine": {
                "depth": 0.27,
                "concepts": [
                    "Pharmacogenomics Drug Response", "Precision Medicine",
                    "Oncogenomics Cancer Genomics", "Tumor Mutational Burden",
                    "Liquid Biopsy ctDNA", "Companion Diagnostics",
                    "Rare Disease Genomics", "Mendelian Disorders",
                    "Polygenic Risk Scores", "Direct-to-Consumer Genetics",
                    "Newborn Screening Genetics", "Prenatal Genetic Testing",
                    "Gene Panels Sequencing", "Whole Exome Sequencing WES",
                    "Long-Read Sequencing Nanopore", "Single-Cell Genomics",
                    "Spatial Transcriptomics", "Multi-Omics Integration",
                ]
            },
        }
    },

    # ─── GALAXY 4: PHARMACOLOGY & MEDICINE ─────────────────────────────
    "Pharmacology & Medicine": {
        "depth": 0.09,
        "stars": {
            "Pharmacology Principles": {
                "depth": 0.19,
                "concepts": [
                    "Drug-Receptor Interactions", "Agonist Antagonist",
                    "Dose-Response Curve", "ED50 LD50 Therapeutic Index",
                    "Pharmacokinetics ADME", "Absorption Distribution",
                    "Metabolism Cytochrome P450", "Renal Excretion",
                    "Bioavailability Oral IV", "Half-Life Elimination",
                    "First-Pass Effect", "Blood-Brain Barrier Drug",
                    "Drug Interactions Synergism", "Tolerance Tachyphylaxis",
                    "Pharmacodynamics Mechanisms", "Signal Transduction Drug",
                    "Prodrugs Activation", "Drug Design Rational",
                ]
            },
            "Neuropharmacology": {
                "depth": 0.22,
                "concepts": [
                    "Antidepressants SSRIs SNRIs", "Monoamine Hypothesis",
                    "Anxiolytics Benzodiazepines", "GABA Receptor Modulation",
                    "Antipsychotics Dopamine", "Typical Atypical Antipsychotics",
                    "Opioid Pharmacology", "Mu Kappa Delta Receptors",
                    "Anesthetics Local General", "Sodium Channel Blockers",
                    "Psychedelics Serotonin 5HT2A", "Psilocybin Pharmacology",
                    "LSD Mechanism Action", "Ketamine NMDA Antagonist",
                    "Cannabis Endocannabinoid System", "THC CBD Pharmacology",
                    "Nootropics Cognitive Enhancement", "Stimulants Amphetamine Mechanism",
                ]
            },
            "Immunology & Vaccines": {
                "depth": 0.24,
                "concepts": [
                    "Innate Immunity Adaptive Immunity", "T-Cells B-Cells",
                    "Antibodies Immunoglobulins", "Antigen Presentation MHC",
                    "Cytokines Interleukins", "Complement System",
                    "Vaccine Mechanisms", "mRNA Vaccines Technology",
                    "Adjuvants Immune Response", "Monoclonal Antibodies Therapy",
                    "Checkpoint Inhibitors PD-1", "CAR-T Immunotherapy",
                    "Autoimmune Diseases Mechanisms", "Allergy Hypersensitivity",
                    "Transplant Immunology", "Herd Immunity Threshold",
                    "Vaccine Development Phases", "Jenner Pasteur Salk",
                ]
            },
            "Medical Diagnostics": {
                "depth": 0.26,
                "concepts": [
                    "MRI Magnetic Resonance Imaging", "CT Scan Computed Tomography",
                    "Ultrasound Imaging", "PET Scan Positron Emission",
                    "X-Ray Radiography", "Mammography Screening",
                    "ECG Electrocardiogram", "EEG Electroencephalogram",
                    "Blood Tests CBC CMP", "Urinalysis",
                    "Biopsy Histopathology", "Flow Cytometry",
                    "PCR Diagnostic Testing", "ELISA Immunoassay",
                    "Point-of-Care Testing", "AI Medical Imaging Diagnosis",
                ]
            },
            "Surgery & Procedures": {
                "depth": 0.28,
                "concepts": [
                    "Minimally Invasive Surgery", "Laparoscopic Techniques",
                    "Robotic Surgery Da Vinci", "Microsurgery",
                    "Organ Transplantation", "Heart Surgery Bypass",
                    "Neurosurgery", "Orthopedic Surgery Joint Replacement",
                    "Ophthalmology LASIK", "Cardiac Catheterization",
                    "Endoscopy Colonoscopy", "Interventional Radiology",
                    "Stem Cell Transplant", "Regenerative Medicine",
                    "3D Bioprinting Organs", "Telemedicine Surgery",
                ]
            },
        }
    },

    # ─── GALAXY 5: EARTH SCIENCES ──────────────────────────────────────
    "Earth Sciences": {
        "depth": 0.08,
        "stars": {
            "Geology": {
                "depth": 0.18,
                "concepts": [
                    "Plate Tectonics Wegener", "Continental Drift",
                    "Seafloor Spreading", "Subduction Zones",
                    "Mid-Ocean Ridges", "Transform Faults",
                    "Igneous Rocks Magma", "Sedimentary Rocks Layers",
                    "Metamorphic Rocks Pressure", "Rock Cycle",
                    "Minerals Crystallography", "Mohs Hardness Scale",
                    "Geological Time Scale Eons", "Stratigraphy",
                    "Radioactive Dating Geology", "Fossil Record",
                    "Earthquake Seismology", "Richter Moment Magnitude",
                    "Volcanic Eruptions Types", "Supervolcanoes Yellowstone",
                ]
            },
            "Oceanography": {
                "depth": 0.22,
                "concepts": [
                    "Ocean Currents Thermohaline", "Gulf Stream",
                    "El Nino La Nina ENSO", "Ocean Acidification",
                    "Coral Reef Bleaching", "Deep Sea Hydrothermal Vents",
                    "Abyssal Plains", "Continental Shelf Slope",
                    "Tides Gravitational", "Tsunami Mechanics",
                    "Marine Food Chain", "Phytoplankton Primary Production",
                    "Ocean Carbon Sink", "Sea Level Rise Projections",
                    "Marine Protected Areas", "Deep Sea Mining",
                    "Mariana Trench", "Mid-Atlantic Ridge",
                ]
            },
            "Atmospheric Science": {
                "depth": 0.24,
                "concepts": [
                    "Atmospheric Layers Troposphere", "Stratosphere Ozone Layer",
                    "Greenhouse Effect CO2", "Radiative Forcing",
                    "Climate Models GCM", "Paleoclimatology Ice Cores",
                    "Milankovitch Cycles Orbital", "Carbon Cycle Global",
                    "Hurricanes Typhoons Formation", "Jet Stream Patterns",
                    "Monsoon Systems", "Coriolis Effect Weather",
                    "Cloud Formation Condensation", "Lightning Physics",
                    "Air Pollution Particulates", "Ozone Depletion CFC",
                    "Climate Sensitivity ECS", "Tipping Points Climate",
                ]
            },
            "Planetary Science": {
                "depth": 0.26,
                "concepts": [
                    "Solar System Formation", "Planetary Differentiation",
                    "Mars Geology Olympus Mons", "Venus Atmosphere",
                    "Jupiter Gas Giant", "Saturn Rings Composition",
                    "Titan Methane Lakes", "Europa Subsurface Ocean",
                    "Asteroid Belt Composition", "Kuiper Belt Objects",
                    "Oort Cloud Comets", "Meteorites Classification",
                    "Moon Formation Giant Impact", "Lunar Geology",
                    "Exoplanet Detection Methods", "Habitable Zone Planets",
                    "Planetary Atmospheres Composition", "Terraforming Concepts",
                ]
            },
        }
    },

    # ─── GALAXY 6: ENGINEERING & TECHNOLOGY ────────────────────────────
    "Engineering": {
        "depth": 0.07,
        "stars": {
            "Mechanical Engineering": {
                "depth": 0.17,
                "concepts": [
                    "Stress Analysis Finite Element", "Beam Theory Euler-Bernoulli",
                    "Fluid Dynamics Navier-Stokes", "Reynolds Number Turbulence",
                    "Heat Transfer Conduction Convection", "Thermodynamic Cycles",
                    "Turbine Design", "Internal Combustion Engine",
                    "Gear Systems Mechanisms", "Bearing Design",
                    "Vibration Analysis Modal", "Control Systems PID",
                    "Robotics Kinematics", "Mechatronics",
                    "CAD CAM Manufacturing", "CNC Machining",
                    "Injection Molding", "Welding Metallurgy",
                ]
            },
            "Electrical Engineering": {
                "depth": 0.20,
                "concepts": [
                    "Circuit Theory Kirchhoff", "Ohm Law",
                    "AC DC Circuits", "Impedance Reactance",
                    "Semiconductor Devices Transistor", "MOSFET Operation",
                    "Operational Amplifiers", "Digital Logic Gates",
                    "Microprocessor Architecture", "FPGA Design",
                    "Power Electronics Converters", "Electric Motors AC DC",
                    "Transformer Principles", "Power Grid Systems",
                    "Antenna Theory Maxwell", "RF Engineering",
                    "Signal Processing DSP", "Control Theory Feedback",
                ]
            },
            "Civil & Structural Engineering": {
                "depth": 0.23,
                "concepts": [
                    "Structural Analysis", "Truss Bridge Design",
                    "Reinforced Concrete Design", "Prestressed Concrete",
                    "Steel Structure Design", "Foundation Engineering",
                    "Soil Mechanics", "Geotechnical Investigation",
                    "Earthquake Resistant Design", "Seismic Isolation",
                    "Dam Engineering", "Tunnel Engineering",
                    "Highway Design", "Traffic Engineering",
                    "Water Treatment Systems", "Wastewater Engineering",
                    "Surveying Geodesy", "Construction Management",
                ]
            },
            "Aerospace Engineering": {
                "depth": 0.25,
                "concepts": [
                    "Aerodynamics Lift Drag", "Bernoulli Equation Flight",
                    "Wing Design Airfoil", "Supersonic Hypersonic Flow",
                    "Rocket Propulsion Tsiolkovsky", "Specific Impulse",
                    "Orbital Mechanics Hohmann Transfer", "Satellite Design",
                    "Space Launch Systems", "Reusable Rockets SpaceX",
                    "Composite Materials Aerospace", "Thermal Protection Systems",
                    "Flight Control Systems", "Avionics",
                    "Space Station Engineering ISS", "Mars Mission Design",
                    "Ion Propulsion", "Space Elevator Concept",
                ]
            },
            "Chemical & Process Engineering": {
                "depth": 0.27,
                "concepts": [
                    "Mass Transfer Operations", "Distillation Column Design",
                    "Reactor Design CSTR PFR", "Reaction Engineering",
                    "Process Control Instrumentation", "Heat Exchanger Design",
                    "Separation Processes", "Membrane Technology",
                    "Fluidization", "Catalytic Reactor Design",
                    "Petrochemical Refining", "Polymer Processing",
                    "Pharmaceutical Manufacturing GMP", "Bioprocess Engineering",
                    "Fermentation Bioreactor", "Water Desalination",
                ]
            },
        }
    },

    # ─── GALAXY 7: ECONOMICS & GAME THEORY ─────────────────────────────
    "Economics & Game Theory": {
        "depth": 0.09,
        "stars": {
            "Microeconomics": {
                "depth": 0.19,
                "concepts": [
                    "Supply and Demand", "Marginal Utility Theory",
                    "Consumer Surplus Producer Surplus", "Price Elasticity",
                    "Perfect Competition", "Monopoly Pricing",
                    "Oligopoly Game Theory", "Monopolistic Competition",
                    "Market Failure Externalities", "Public Goods Free Rider",
                    "Asymmetric Information Akerlof", "Moral Hazard",
                    "Adverse Selection", "Principal-Agent Problem",
                    "Behavioral Economics Nudge", "Prospect Theory Economics",
                    "Endowment Effect", "Bounded Rationality Simon",
                ]
            },
            "Macroeconomics": {
                "depth": 0.22,
                "concepts": [
                    "GDP Measurement", "Inflation CPI",
                    "Unemployment Phillips Curve", "Business Cycle",
                    "Keynesian Economics", "Fiscal Policy Multiplier",
                    "Monetary Policy Central Banks", "Interest Rate Mechanism",
                    "Quantitative Easing", "Modern Monetary Theory MMT",
                    "Austrian Economics Mises Hayek", "Chicago School Friedman",
                    "International Trade Ricardo", "Comparative Advantage",
                    "Exchange Rate Systems", "Balance of Payments",
                    "Economic Growth Solow Model", "Endogenous Growth Theory",
                ]
            },
            "Game Theory": {
                "depth": 0.24,
                "concepts": [
                    "Nash Equilibrium", "Prisoners Dilemma",
                    "Zero-Sum Games", "Cooperative Games Coalition",
                    "Bayesian Games Incomplete Information", "Signaling Games",
                    "Evolutionary Game Theory", "Hawk-Dove Game",
                    "Repeated Games Folk Theorem", "Subgame Perfect Equilibrium",
                    "Mechanism Design Revelation", "Auction Theory Vickrey",
                    "Voting Theory Arrow Impossibility", "Fair Division",
                    "Matching Theory Gale-Shapley", "Market Design Roth",
                    "Stochastic Games", "Mean Field Games",
                ]
            },
            "Financial Mathematics": {
                "depth": 0.26,
                "concepts": [
                    "Black-Scholes Option Pricing", "Brownian Motion Finance",
                    "Stochastic Calculus Ito", "Risk-Neutral Pricing",
                    "Portfolio Theory Markowitz", "Capital Asset Pricing Model",
                    "Efficient Market Hypothesis", "Arbitrage Pricing Theory",
                    "Value at Risk VaR", "Monte Carlo Simulation Finance",
                    "Derivatives Futures Options Swaps", "Credit Risk Models",
                    "Yield Curve Term Structure", "Fixed Income Analytics",
                    "Algorithmic Trading", "High-Frequency Trading",
                    "Cryptocurrency Economics", "DeFi Automated Market Makers",
                ]
            },
        }
    },

    # ─── GALAXY 8: ENERGY & SUSTAINABILITY ─────────────────────────────
    "Energy & Sustainability": {
        "depth": 0.10,
        "stars": {
            "Renewable Energy": {
                "depth": 0.20,
                "concepts": [
                    "Solar Photovoltaic Technology", "Solar Cell Efficiency",
                    "Perovskite Solar Cells", "Concentrated Solar Power",
                    "Wind Turbine Design", "Offshore Wind Farms",
                    "Hydroelectric Power Dam", "Tidal Energy",
                    "Wave Energy Converter", "Geothermal Energy",
                    "Biomass Energy Biofuels", "Bioethanol Biodiesel",
                    "Hydrogen Fuel Cell", "Green Hydrogen Electrolysis",
                    "Energy Storage Battery", "Lithium-Ion Technology",
                    "Solid-State Batteries", "Flow Batteries Vanadium",
                    "Pumped Hydro Storage", "Grid-Scale Storage",
                ]
            },
            "Nuclear Energy": {
                "depth": 0.23,
                "concepts": [
                    "Nuclear Power Plant PWR BWR", "Uranium Enrichment",
                    "Nuclear Fuel Cycle", "Spent Fuel Reprocessing",
                    "Generation IV Reactors", "Small Modular Reactors SMR",
                    "Thorium Molten Salt Reactor", "Fast Breeder Reactor",
                    "Fusion Energy ITER", "Tokamak Design",
                    "Stellarator Design", "Inertial Confinement Fusion",
                    "Plasma Physics Fusion", "Tritium Breeding",
                    "Nuclear Safety Containment", "Nuclear Proliferation",
                ]
            },
            "Environmental Science": {
                "depth": 0.25,
                "concepts": [
                    "Carbon Footprint Calculation", "Life Cycle Assessment",
                    "Circular Economy", "Waste Management Hierarchy",
                    "Plastic Pollution Microplastics", "Ocean Cleanup",
                    "Deforestation Rates", "Reforestation Afforestation",
                    "Biodiversity Loss Sixth Extinction", "Species Conservation",
                    "Water Scarcity Global", "Sustainable Agriculture",
                    "Organic Farming Permaculture", "Agroforestry",
                    "Carbon Capture and Storage CCS", "Direct Air Capture",
                    "Geoengineering Solar", "Environmental Justice",
                ]
            },
        }
    },
}

# Cross-galaxy bridges
CROSS_EDGES = [
    # Chemistry ↔ Metallurgy
    ("Chemical Bonding Covalent Ionic", "Crystal Structure BCC FCC HCP", "Determines"),
    ("Electrochemistry Nernst", "Corrosion Electrochemical", "Governs"),
    ("Solid State Chemistry", "Crystal Structures", "Studies"),
    ("Organometallic Chemistry", "Organometallic Catalysis", "Applies"),
    ("Catalysis Homogeneous Heterogeneous", "Homogeneous Catalysis", "Type"),

    # Chemistry ↔ Genomics
    ("Nucleic Acids DNA RNA Structure", "Watson Crick Base Pairing", "Structure"),
    ("Protein Structure Primary Secondary Tertiary", "Amino Acids Twenty", "Builds"),
    ("Enzyme Mechanisms Lock Key", "Drug-Receptor Interactions", "Analogous"),

    # Chemistry ↔ Pharmacology
    ("Drug Design Rational", "Pharmacodynamics Mechanisms", "Applies"),
    ("Metabolism Cytochrome P450", "Organic Chemistry", "Requires"),
    ("Nuclear Medicine PET Scan", "PET Scan Positron Emission", "Same"),

    # Genomics ↔ Pharmacology
    ("Pharmacogenomics Drug Response", "Precision Medicine", "Enables"),
    ("CRISPR-Cas9 Mechanism", "CAR-T Cell Therapy", "Enables"),
    ("CAR-T Immunotherapy", "CAR-T Cell Therapy", "Related"),
    ("Oncogenomics Cancer Genomics", "Checkpoint Inhibitors PD-1", "Guides"),

    # Metallurgy ↔ Engineering
    ("Fatigue Failure", "Stress Analysis Finite Element", "Predicts"),
    ("Titanium Alloys Ti-6Al-4V", "Composite Materials Aerospace", "Competes"),
    ("Carbon Fiber Composites", "Wing Design Airfoil", "Constructs"),
    ("Additive Manufacturing Materials", "3D Bioprinting Organs", "Related"),
    ("Welding Metallurgy", "Steel Structure Design", "Joins"),
    ("Damascus Steel Wootz", "Iron-Carbon Phase Diagram", "Example"),

    # Engineering ↔ Earth Sciences
    ("Earthquake Resistant Design", "Earthquake Seismology", "Responds"),
    ("Dam Engineering", "Hydroelectric Power Dam", "Combines"),
    ("Tunnel Engineering", "Soil Mechanics", "Requires"),
    ("Rocket Propulsion Tsiolkovsky", "Space Launch Systems", "Powers"),
    ("Mars Mission Design", "Mars Geology Olympus Mons", "Targets"),

    # Earth Sciences ↔ Energy
    ("Geothermal Energy", "Volcanic Eruptions Types", "Exploits"),
    ("Ocean Carbon Sink", "Carbon Capture and Storage CCS", "Natural"),
    ("Climate Models GCM", "Radiative Forcing", "Uses"),
    ("Paleoclimatology Ice Cores", "Milankovitch Cycles Orbital", "Records"),

    # Economics ↔ Energy
    ("Cryptocurrency Economics", "DeFi Automated Market Makers", "Enables"),
    ("Carbon Footprint Calculation", "Market Failure Externalities", "Example"),
    ("Efficient Market Hypothesis", "Algorithmic Trading", "Tests"),

    # Chemistry ↔ Energy
    ("Lithium-Ion Technology", "Electrochemistry Nernst", "Applies"),
    ("Green Hydrogen Electrolysis", "Electrochemistry Nernst", "Uses"),
    ("Hydrogen Fuel Cell", "Oxidation Reduction Reactions", "Uses"),
    ("Perovskite Solar Cells", "Crystal Structures", "Related"),

    # Game Theory ↔ Evolution
    ("Evolutionary Game Theory", "Hawk-Dove Game", "Models"),
    ("Nash Equilibrium", "Prisoners Dilemma", "Solution"),
    ("Mechanism Design Revelation", "Auction Theory Vickrey", "Applies"),

    # Nuclear ↔ Earth
    ("Radioisotope Dating Carbon-14", "Radioactive Dating Geology", "Same"),
    ("Nuclear Fission", "Nuclear Power Plant PWR BWR", "Powers"),
    ("Nuclear Fusion", "Fusion Energy ITER", "Goal"),
    ("Plasma Physics Fusion", "Tokamak Design", "Contains"),
]


def make_poincare_embedding(galaxy_name, star_name, concept_name, depth, dim):
    galaxy_hash = hashlib.sha256(galaxy_name.encode()).digest()
    star_hash = hashlib.sha256(f"{galaxy_name}/{star_name}".encode()).digest()
    concept_hash = hashlib.sha256(f"{galaxy_name}/{star_name}/{concept_name}".encode()).digest()
    coords = []
    for i in range(dim):
        g_val = galaxy_hash[i % 32] / 255.0 - 0.5
        s_val = star_hash[i % 32] / 255.0 - 0.5
        c_val = concept_hash[i % 32] / 255.0 - 0.5
        val = 0.5 * g_val + 0.3 * s_val + 0.2 * c_val
        byte_idx = (i * 7 + 13) % 32
        val += 0.1 * (concept_hash[byte_idx] / 255.0 - 0.5)
        coords.append(val)
    norm = math.sqrt(sum(c*c for c in coords))
    if norm > 0:
        jitter = (concept_hash[0] / 255.0 - 0.5) * 0.1
        target_radius = min(depth + jitter, 0.95)
        target_radius = max(target_radius, 0.02)
        coords = [c / norm * target_radius for c in coords]
    return coords


def make_edge_request(pb2, **kwargs):
    from_val = kwargs.pop('from_node')
    req = pb2.InsertEdgeRequest(**kwargs)
    setattr(req, 'from', from_val)
    return req


def insert_galaxies(host, collection, metric="poincare", dim=128):
    pb2, pb2_grpc = ensure_proto_compiled()
    channel = grpc.insecure_channel(host, options=[
        ('grpc.max_send_message_length', 256 * 1024 * 1024),
        ('grpc.max_receive_message_length', 256 * 1024 * 1024),
    ])
    stub = pb2_grpc.NietzscheDBStub(channel)

    print(f"[*] Waiting for gRPC server at {host}...")
    try:
        grpc.channel_ready_future(channel).result(timeout=120)
        print(f"[+] gRPC server ready!")
    except grpc.FutureTimeoutError:
        print(f"[!] Timeout. Aborting.")
        return

    print(f"\n{'='*60}")
    print(f"  NietzscheDB Science & Engineering Ingestion")
    print(f"  Collection: {collection} | Dim: {dim}")
    print(f"{'='*60}\n")

    try:
        stub.CreateCollection(pb2.CreateCollectionRequest(collection=collection, dim=dim, metric=metric))
        print(f"[+] Collection '{collection}' created")
    except grpc.RpcError as e:
        print(f"[!] CreateCollection: {e.details() if hasattr(e, 'details') else e}")

    all_nodes = []
    node_ids = {}

    for galaxy_name, galaxy_data in GALAXIES.items():
        galaxy_depth = galaxy_data["depth"]
        galaxy_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"science-galaxy:{galaxy_name}"))
        node_ids[galaxy_name] = galaxy_id
        emb = make_poincare_embedding(galaxy_name, "__galaxy__", galaxy_name, galaxy_depth, dim)
        all_nodes.append({"id": galaxy_id,
            "content": json.dumps({"name": galaxy_name, "type": "galaxy", "domain": "science"}).encode('utf-8'),
            "node_type": "Concept", "energy": 1.0, "embedding": emb})

        for star_name, star_data in galaxy_data["stars"].items():
            star_depth = star_data["depth"]
            star_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"science-star:{galaxy_name}/{star_name}"))
            node_ids[star_name] = star_id
            emb = make_poincare_embedding(galaxy_name, star_name, star_name, star_depth, dim)
            all_nodes.append({"id": star_id,
                "content": json.dumps({"name": star_name, "type": "star", "galaxy": galaxy_name}).encode('utf-8'),
                "node_type": "Concept", "energy": 0.8, "embedding": emb})

            for concept_name in star_data["concepts"]:
                concept_depth = star_depth + 0.1 + (hash(concept_name) % 100) / 1000.0
                concept_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"science-concept:{galaxy_name}/{star_name}/{concept_name}"))
                node_ids[concept_name] = concept_id
                emb = make_poincare_embedding(galaxy_name, star_name, concept_name, concept_depth, dim)
                all_nodes.append({"id": concept_id,
                    "content": json.dumps({"name": concept_name, "type": "concept",
                        "star": star_name, "galaxy": galaxy_name}).encode('utf-8'),
                    "node_type": "Semantic", "energy": 0.6, "embedding": emb})

    total_nodes = len(all_nodes)
    print(f"[*] Inserting {total_nodes} nodes...")

    batch_size = 50
    inserted = 0
    for i in range(0, total_nodes, batch_size):
        batch = all_nodes[i:i+batch_size]
        requests = [pb2.InsertNodeRequest(id=n["id"],
            embedding=pb2.PoincareVector(coords=n["embedding"]),
            content=n["content"], node_type=n["node_type"],
            energy=n["energy"], collection=collection) for n in batch]
        try:
            stub.BatchInsertNodes(pb2.BatchInsertNodesRequest(nodes=requests, collection=collection))
            inserted += len(batch)
            pct = inserted / total_nodes * 100
            print(f"  [{pct:5.1f}%] {inserted}/{total_nodes}", end='\r')
        except grpc.RpcError as e:
            print(f"\n  [!] Batch {i//batch_size}: {e.details() if hasattr(e, 'details') else e}")

    print(f"\n[+] Nodes: {inserted}/{total_nodes}")

    print(f"\n[*] Creating edges...")
    edge_count = 0
    for galaxy_name, galaxy_data in GALAXIES.items():
        galaxy_id = node_ids[galaxy_name]
        for star_name, star_data in galaxy_data["stars"].items():
            star_id = node_ids.get(star_name)
            if not star_id: continue
            try:
                stub.InsertEdge(make_edge_request(pb2, from_node=galaxy_id, to=star_id,
                    edge_type="Hierarchical", weight=1.0, collection=collection))
                edge_count += 1
            except grpc.RpcError: pass

            for concept_name in star_data["concepts"]:
                concept_id = node_ids.get(concept_name)
                if not concept_id: continue
                try:
                    stub.InsertEdge(make_edge_request(pb2, from_node=star_id, to=concept_id,
                        edge_type="Hierarchical", weight=0.8, collection=collection))
                    edge_count += 1
                except grpc.RpcError: pass

            clist = [c for c in star_data["concepts"] if c in node_ids]
            for j in range(len(clist)):
                for k in range(j+1, min(j+4, len(clist))):
                    try:
                        stub.InsertEdge(make_edge_request(pb2, from_node=node_ids[clist[j]],
                            to=node_ids[clist[k]], edge_type="Association", weight=0.6, collection=collection))
                        edge_count += 1
                    except grpc.RpcError: pass

        print(f"  '{galaxy_name}': {edge_count} edges     ", end='\r')

    print(f"\n[+] Hierarchical edges: {edge_count}")

    print(f"[*] Cross-galaxy bridges...")
    bridge_count = 0
    for from_name, to_name, rel_type in CROSS_EDGES:
        from_id = node_ids.get(from_name)
        to_id = node_ids.get(to_name)
        if from_id and to_id:
            try:
                stub.InsertEdge(make_edge_request(pb2, from_node=from_id, to=to_id,
                    edge_type="Association", weight=0.7, collection=collection))
                bridge_count += 1
            except grpc.RpcError: pass

    total_edges = edge_count + bridge_count
    print(f"[+] Bridges: {bridge_count}")

    print(f"\n{'='*60}")
    print(f"  SCIENCE & ENGINEERING INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Nodes: {inserted} | Edges: {total_edges} | Bridges: {bridge_count}")
    print(f"{'='*60}")
    for gn, gd in GALAXIES.items():
        s = len(gd["stars"])
        c = sum(len(sd["concepts"]) for sd in gd["stars"].values())
        print(f"  {gn}: {s} stars, {c} concepts")
    print(f"\n  Dashboard: http://{host.split(':')[0]}:8080")
    channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost:50051")
    parser.add_argument("--collection", default="science_galaxies")
    parser.add_argument("--metric", default="poincare")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    insert_galaxies(args.host, args.collection, args.metric, args.dim)

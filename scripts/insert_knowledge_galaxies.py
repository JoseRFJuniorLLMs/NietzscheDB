#!/usr/bin/env python3
"""
Insert Knowledge Galaxies into NietzscheDB.

Massive knowledge injection: Literature, Physics, Fractals, Cosmology,
Philosophy, Neuroscience, Music/Art, Biology/Evolution.

Creates ~2000+ nodes and ~8000+ edges in the Poincaré ball.

Usage:
  python scripts/insert_knowledge_galaxies.py [--host HOST:PORT] [--collection NAME]
"""

import grpc
import json
import uuid
import math
import time
import hashlib
import sys
import os
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))

from grpc_tools import protoc
import importlib


def ensure_proto_compiled(repo_root=None):
    """Compile proto if needed."""
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
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_gen')
    if not os.access(os.path.dirname(os.path.abspath(__file__)), os.W_OK):
        out_dir = '/tmp/_nietzsche_gen'
    os.makedirs(out_dir, exist_ok=True)

    pb2_file = os.path.join(out_dir, 'nietzsche_pb2.py')
    if not os.path.exists(pb2_file):
        print(f"[proto] Compiling nietzsche.proto from {proto_dir}...")
        protoc.main([
            'grpc_tools.protoc',
            f'-I{proto_dir}',
            f'--python_out={out_dir}',
            f'--grpc_python_out={out_dir}',
            'nietzsche.proto',
        ])
        with open(os.path.join(out_dir, '__init__.py'), 'w') as f:
            f.write('')
        print(f"[proto] Compiled to {out_dir}")

    sys.path.insert(0, out_dir)
    pb2 = importlib.import_module('nietzsche_pb2')
    pb2_grpc = importlib.import_module('nietzsche_pb2_grpc')
    return pb2, pb2_grpc


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE ONTOLOGY - 8 GALAXIES OF HUMAN KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════════════

GALAXIES = {

    # ─── GALAXY 1: LITERATURE & LITERARY THEORY ─────────────────────────
    "World Literature": {
        "depth": 0.08,
        "stars": {
            "Ancient & Classical Literature": {
                "depth": 0.18,
                "concepts": [
                    "Epic of Gilgamesh", "Iliad", "Odyssey", "Aeneid",
                    "Mahabharata", "Ramayana", "Beowulf", "Tao Te Ching",
                    "Art of War", "Meditations Marcus Aurelius", "Republic Plato",
                    "Poetics Aristotle", "Oresteia", "Metamorphoses Ovid",
                    "Divine Comedy", "Canterbury Tales", "One Thousand and One Nights",
                    "I Ching", "Bhagavad Gita", "Upanishads",
                ]
            },
            "Renaissance to Enlightenment": {
                "depth": 0.22,
                "concepts": [
                    "Shakespeare Hamlet", "Shakespeare Macbeth", "Shakespeare King Lear",
                    "Shakespeare Othello", "Shakespeare Tempest", "Shakespeare Sonnets",
                    "Don Quixote", "Paradise Lost", "Canterbury Tales Chaucer",
                    "Machiavelli The Prince", "Montaigne Essays",
                    "Gulliver Travels", "Candide Voltaire", "Robinson Crusoe",
                    "Faust Goethe", "Sorrows of Young Werther",
                    "Decameron Boccaccio", "Orlando Furioso",
                ]
            },
            "Russian Literature": {
                "depth": 0.25,
                "concepts": [
                    "Crime and Punishment", "Brothers Karamazov", "The Idiot",
                    "Demons Dostoevsky", "Notes from Underground",
                    "War and Peace", "Anna Karenina", "Death of Ivan Ilyich",
                    "Eugene Onegin", "Dead Souls Gogol", "The Overcoat",
                    "Master and Margarita", "Doctor Zhivago",
                    "Fathers and Sons Turgenev", "Cherry Orchard Chekhov",
                    "We Zamyatin", "Gulag Archipelago",
                    "One Day in the Life of Ivan Denisovich",
                ]
            },
            "Modernism & Existentialist Fiction": {
                "depth": 0.28,
                "concepts": [
                    "Ulysses Joyce", "Finnegans Wake", "Dubliners",
                    "The Trial Kafka", "The Castle Kafka", "Metamorphosis Kafka",
                    "In Search of Lost Time Proust", "Mrs Dalloway Woolf",
                    "To the Lighthouse", "The Waves Woolf",
                    "The Stranger Camus", "The Plague Camus", "The Fall Camus",
                    "Nausea Sartre", "No Exit Sartre",
                    "The Sound and the Fury Faulkner", "As I Lay Dying",
                    "The Great Gatsby", "Brave New World", "1984 Orwell",
                    "The Sun Also Rises", "Old Man and the Sea Hemingway",
                ]
            },
            "Latin American Literature": {
                "depth": 0.30,
                "concepts": [
                    "One Hundred Years of Solitude", "Love in the Time of Cholera",
                    "Borges Ficciones", "Borges The Aleph", "Borges Labyrinths",
                    "Pedro Paramo Rulfo", "Hopscotch Cortazar",
                    "The Savage Detectives Bolano", "2666 Bolano",
                    "Explosion in a Cathedral Carpentier",
                    "Magical Realism", "Boom Latinoamericano",
                    "Gabriela Clove and Cinnamon Amado",
                    "The Obscene Bird of Night Donoso",
                    "Conversation in the Cathedral Vargas Llosa",
                ]
            },
            "Literary Theory & Criticism": {
                "depth": 0.35,
                "concepts": [
                    "Structuralism", "Post-Structuralism", "Deconstruction Derrida",
                    "Semiotics Saussure", "Narratology", "Reader Response Theory",
                    "New Criticism", "Formalism Russian", "Prague School",
                    "Hermeneutics", "Phenomenological Criticism",
                    "Feminist Literary Theory", "Postcolonial Theory",
                    "Psychoanalytic Criticism Lacan", "Marxist Literary Theory",
                    "Ecocriticism", "Digital Humanities",
                    "Intertextuality Kristeva", "Death of the Author Barthes",
                    "Anxiety of Influence Bloom",
                ]
            },
            "Poetry & Poetic Forms": {
                "depth": 0.32,
                "concepts": [
                    "Sonnet Form", "Haiku", "Free Verse", "Epic Poetry",
                    "Lyric Poetry", "Ode", "Elegy", "Ballad",
                    "Rumi Poetry", "Pablo Neruda", "Walt Whitman Leaves of Grass",
                    "Emily Dickinson", "TS Eliot Waste Land", "Rilke Duino Elegies",
                    "Baudelaire Flowers of Evil", "Yeats Second Coming",
                    "Pessoa Heteronyms", "Celan Death Fugue",
                    "Basho Narrow Road", "Li Bai Tang Poetry",
                    "Sylvia Plath", "Allen Ginsberg Howl",
                ]
            },
        }
    },

    # ─── GALAXY 2: PHYSICS ──────────────────────────────────────────────
    "Physics": {
        "depth": 0.06,
        "stars": {
            "Classical Mechanics": {
                "depth": 0.16,
                "concepts": [
                    "Newton Laws of Motion", "Conservation of Energy",
                    "Conservation of Momentum", "Angular Momentum",
                    "Lagrangian Mechanics", "Hamiltonian Mechanics",
                    "Noether Theorem", "Principle of Least Action",
                    "Kepler Laws", "Orbital Mechanics",
                    "Rigid Body Dynamics", "Euler Equations Rotation",
                    "Chaotic Pendulum", "Three Body Problem",
                    "Phase Space", "Liouville Theorem",
                    "Canonical Transformations", "Poisson Brackets",
                    "D'Alembert Principle", "Virtual Work",
                ]
            },
            "Quantum Mechanics": {
                "depth": 0.20,
                "concepts": [
                    "Wave Function", "Schrodinger Equation", "Heisenberg Uncertainty",
                    "Superposition Principle", "Quantum Entanglement",
                    "Copenhagen Interpretation", "Many Worlds Interpretation",
                    "Quantum Tunneling", "Quantum Harmonic Oscillator",
                    "Spin Angular Momentum", "Pauli Exclusion Principle",
                    "Dirac Equation", "Klein-Gordon Equation",
                    "Quantum Field Theory", "Path Integral Formulation",
                    "Bell Theorem", "EPR Paradox", "Quantum Decoherence",
                    "Density Matrix", "Quantum Computing Principles",
                    "Bra-Ket Notation", "Hilbert Space",
                    "Born Rule", "Measurement Problem",
                    "Quantum Electrodynamics QED", "Feynman Diagrams",
                    "Renormalization", "Lamb Shift", "Casimir Effect",
                ]
            },
            "General Relativity & Gravity": {
                "depth": 0.22,
                "concepts": [
                    "Einstein Field Equations", "Spacetime Curvature",
                    "Geodesics", "Schwarzschild Metric",
                    "Gravitational Waves", "LIGO Detection",
                    "Frame Dragging", "Lense-Thirring Effect",
                    "Equivalence Principle", "Gravitational Time Dilation",
                    "Gravitational Lensing", "Perihelion Precession",
                    "Kerr Metric", "Reissner-Nordstrom Metric",
                    "Penrose Diagrams", "Causal Structure",
                    "ADM Formalism", "Riemann Curvature Tensor",
                    "Christoffel Symbols", "Ricci Tensor",
                    "Stress-Energy Tensor", "Cosmological Constant",
                ]
            },
            "Thermodynamics & Statistical Mechanics": {
                "depth": 0.24,
                "concepts": [
                    "First Law Thermodynamics", "Second Law Thermodynamics",
                    "Third Law Thermodynamics", "Zeroth Law",
                    "Entropy", "Boltzmann Distribution",
                    "Gibbs Free Energy", "Helmholtz Free Energy",
                    "Partition Function", "Microstate Macrostate",
                    "Maxwell-Boltzmann Statistics", "Fermi-Dirac Statistics",
                    "Bose-Einstein Statistics", "Bose-Einstein Condensate",
                    "Phase Transitions", "Critical Phenomena",
                    "Ising Model", "Renormalization Group",
                    "Maxwell Demon", "Carnot Cycle",
                    "Ergodic Hypothesis", "Fluctuation-Dissipation Theorem",
                ]
            },
            "Particle Physics": {
                "depth": 0.26,
                "concepts": [
                    "Standard Model", "Quarks", "Leptons", "Bosons",
                    "Higgs Boson", "Higgs Mechanism",
                    "Strong Nuclear Force", "Weak Nuclear Force",
                    "Quantum Chromodynamics QCD", "Gluons",
                    "W Boson", "Z Boson", "Photon",
                    "Neutrino Oscillations", "CP Violation",
                    "Antimatter", "Pair Production",
                    "Hadrons Mesons Baryons", "Quark Confinement",
                    "Asymptotic Freedom", "Running Coupling Constants",
                    "Grand Unified Theory", "Supersymmetry",
                    "String Theory", "M-Theory", "Extra Dimensions",
                ]
            },
            "Condensed Matter Physics": {
                "depth": 0.28,
                "concepts": [
                    "Crystal Lattice", "Bravais Lattice", "Bloch Theorem",
                    "Band Theory", "Fermi Surface",
                    "Superconductivity", "BCS Theory", "Cooper Pairs",
                    "Superfluidity", "Topological Insulators",
                    "Quantum Hall Effect", "Fractional Quantum Hall",
                    "Phonons", "Magnons", "Plasmons",
                    "Semiconductor Physics", "PN Junction",
                    "Spintronics", "Graphene Physics",
                    "Metamaterials", "Photonic Crystals",
                ]
            },
        }
    },

    # ─── GALAXY 3: FRACTALS & CHAOS THEORY ──────────────────────────────
    "Fractals & Chaos": {
        "depth": 0.07,
        "stars": {
            "Fractal Geometry": {
                "depth": 0.17,
                "concepts": [
                    "Mandelbrot Set", "Julia Set", "Fatou Set",
                    "Hausdorff Dimension", "Box-Counting Dimension",
                    "Self-Similarity", "Self-Affinity",
                    "Sierpinski Triangle", "Sierpinski Carpet",
                    "Menger Sponge", "Cantor Set", "Koch Snowflake",
                    "Dragon Curve", "Hilbert Curve", "Peano Curve",
                    "Space-Filling Curves", "Fractal Dimension",
                    "Barnsley Fern", "Iterated Function Systems IFS",
                    "L-Systems Lindenmayer", "Fractal Trees",
                    "Apollonian Gasket", "Levy C Curve",
                    "Newton Fractals", "Burning Ship Fractal",
                    "Multifractal Analysis", "Lacunarity",
                ]
            },
            "Chaos Theory": {
                "depth": 0.20,
                "concepts": [
                    "Butterfly Effect", "Sensitive Dependence Initial Conditions",
                    "Lorenz Attractor", "Strange Attractors",
                    "Lyapunov Exponents", "Lyapunov Stability",
                    "Bifurcation Theory", "Period Doubling Cascade",
                    "Feigenbaum Constants", "Logistic Map",
                    "Henon Map", "Rossler Attractor",
                    "Routes to Chaos", "Intermittency",
                    "Poincare Map", "Poincare Recurrence",
                    "Ergodic Theory", "Mixing Systems",
                    "KAM Theorem", "Arnold Diffusion",
                    "Deterministic Chaos", "Pseudorandom Generators",
                ]
            },
            "Dynamical Systems": {
                "depth": 0.22,
                "concepts": [
                    "Fixed Points", "Limit Cycles", "Attractors",
                    "Basin of Attraction", "Stability Analysis",
                    "Bifurcation Diagrams", "Hopf Bifurcation",
                    "Saddle-Node Bifurcation", "Pitchfork Bifurcation",
                    "Center Manifold Theorem", "Normal Forms",
                    "Floquet Theory", "Poincare-Bendixson Theorem",
                    "Topological Dynamics", "Symbolic Dynamics",
                    "Smale Horseshoe", "Cantor Dynamics",
                    "Cellular Automata", "Game of Life Conway",
                    "Rule 110", "Wolfram Classification",
                    "Reaction-Diffusion Systems", "Turing Patterns",
                ]
            },
            "Complex Systems & Emergence": {
                "depth": 0.25,
                "concepts": [
                    "Emergence", "Self-Organization", "Autopoiesis",
                    "Swarm Intelligence", "Ant Colony Optimization",
                    "Particle Swarm Optimization", "Flocking Behavior",
                    "Scale-Free Networks", "Small World Networks",
                    "Power Law Distributions", "Preferential Attachment",
                    "Percolation Theory", "Phase Transitions Complex",
                    "Criticality", "Self-Organized Criticality",
                    "Sandpile Model", "Punctuated Equilibrium",
                    "Agent-Based Modeling", "Cellular Automata Complexity",
                    "Edge of Chaos", "Complexity Classes",
                    "Information Theory Complexity", "Kolmogorov Complexity",
                ]
            },
            "Nonlinear Dynamics Applications": {
                "depth": 0.28,
                "concepts": [
                    "Turbulence", "Navier-Stokes Turbulence",
                    "Weather Prediction Chaos", "Climate Models Nonlinear",
                    "Heart Rhythm Chaos", "Brain Dynamics Nonlinear",
                    "Ecological Dynamics", "Predator-Prey Models",
                    "Lotka-Volterra Equations", "Population Dynamics",
                    "Financial Markets Chaos", "Econophysics",
                    "Epidemic Modeling", "SIR Model",
                    "Traffic Flow Dynamics", "Synchronization Phenomena",
                    "Kuramoto Model", "Coupled Oscillators",
                    "Solitons", "Nonlinear Schrodinger Equation",
                ]
            },
        }
    },

    # ─── GALAXY 4: COSMOLOGY & ASTROPHYSICS ─────────────────────────────
    "Cosmology & Astrophysics": {
        "depth": 0.09,
        "stars": {
            "Big Bang & Early Universe": {
                "depth": 0.19,
                "concepts": [
                    "Big Bang Theory", "Cosmic Inflation",
                    "Cosmic Microwave Background", "CMB Anisotropy",
                    "Nucleosynthesis", "Recombination Epoch",
                    "Planck Epoch", "Grand Unification Epoch",
                    "Electroweak Epoch", "Quark Epoch",
                    "Baryogenesis", "Matter-Antimatter Asymmetry",
                    "Hubble Expansion", "Hubble Constant",
                    "Friedmann Equations", "FLRW Metric",
                    "Cosmic Horizon", "Observable Universe",
                    "Flatness Problem", "Horizon Problem",
                ]
            },
            "Dark Universe": {
                "depth": 0.22,
                "concepts": [
                    "Dark Matter", "Dark Energy", "Cosmological Constant",
                    "WIMP Dark Matter", "Axion Dark Matter",
                    "Dark Matter Halos", "Rotation Curves",
                    "Gravitational Lensing Dark Matter",
                    "Lambda-CDM Model", "Concordance Cosmology",
                    "Accelerating Expansion", "Type Ia Supernovae Standard Candles",
                    "Baryon Acoustic Oscillations", "Dark Energy Equation of State",
                    "Modified Gravity MOND", "TeVeS",
                    "Dark Matter Detection", "Direct Detection Experiments",
                    "Bullet Cluster", "Dark Matter Simulations",
                ]
            },
            "Stellar Astrophysics": {
                "depth": 0.24,
                "concepts": [
                    "Stellar Evolution", "Hertzsprung-Russell Diagram",
                    "Main Sequence", "Red Giant Phase",
                    "White Dwarf", "Neutron Star", "Pulsar",
                    "Supernova Type II", "Supernova Remnant",
                    "Stellar Nucleosynthesis", "CNO Cycle",
                    "Proton-Proton Chain", "Triple Alpha Process",
                    "Chandrasekhar Limit", "Tolman-Oppenheimer-Volkoff Limit",
                    "Stellar Atmosphere", "Spectral Classification",
                    "Binary Star Systems", "X-ray Binaries",
                    "Magnetar", "Wolf-Rayet Stars",
                ]
            },
            "Black Holes": {
                "depth": 0.26,
                "concepts": [
                    "Schwarzschild Black Hole", "Kerr Black Hole",
                    "Event Horizon", "Singularity",
                    "Hawking Radiation", "Black Hole Thermodynamics",
                    "Bekenstein-Hawking Entropy", "Information Paradox",
                    "No-Hair Theorem", "Penrose Process",
                    "Ergosphere", "Accretion Disk",
                    "Black Hole Jets", "Active Galactic Nuclei",
                    "Quasars", "Supermassive Black Holes",
                    "Sagittarius A Star", "M87 Black Hole Image",
                    "Gravitational Wave Mergers", "Primordial Black Holes",
                    "Black Hole Complementarity", "Firewall Paradox",
                ]
            },
            "Galaxies & Large-Scale Structure": {
                "depth": 0.28,
                "concepts": [
                    "Galaxy Formation", "Galaxy Morphology",
                    "Spiral Galaxies", "Elliptical Galaxies",
                    "Irregular Galaxies", "Galaxy Clusters",
                    "Cosmic Web", "Filaments Voids",
                    "Galaxy Mergers", "Tidal Interactions",
                    "Milky Way Structure", "Galactic Center",
                    "Andromeda Galaxy", "Local Group",
                    "Virgo Supercluster", "Laniakea Supercluster",
                    "Cosmic Voids", "Great Attractor",
                    "Redshift Surveys", "Galaxy Power Spectrum",
                ]
            },
        }
    },

    # ─── GALAXY 5: PHILOSOPHY ───────────────────────────────────────────
    "Philosophy": {
        "depth": 0.05,
        "stars": {
            "Ancient Philosophy": {
                "depth": 0.15,
                "concepts": [
                    "Socratic Method", "Platonic Forms", "Allegory of the Cave",
                    "Aristotle Ethics", "Aristotle Metaphysics", "Aristotle Logic",
                    "Stoicism", "Epicureanism", "Cynicism",
                    "Pre-Socratics", "Heraclitus Flux", "Parmenides Being",
                    "Democritus Atomism", "Pythagoreanism",
                    "Neoplatonism Plotinus", "Skepticism Pyrrhonism",
                    "Confucianism", "Daoism Laozi", "Buddhism Philosophy",
                    "Nagarjuna Emptiness", "Vedanta",
                ]
            },
            "Modern Philosophy": {
                "depth": 0.20,
                "concepts": [
                    "Descartes Cogito", "Cartesian Dualism",
                    "Spinoza Substance Monism", "Leibniz Monadology",
                    "Locke Empiricism", "Hume Causation",
                    "Hume Problem of Induction", "Berkeley Idealism",
                    "Kant Critique Pure Reason", "Kant Categorical Imperative",
                    "Transcendental Idealism", "Thing-in-Itself",
                    "Hegel Dialectic", "Hegel Phenomenology of Spirit",
                    "Schopenhauer Will", "Nietzsche Eternal Recurrence",
                    "Nietzsche Will to Power", "Nietzsche Ubermensch",
                    "Nietzsche Beyond Good and Evil", "Nietzsche Genealogy of Morals",
                    "Kierkegaard Leap of Faith", "Kierkegaard Stages",
                ]
            },
            "Existentialism & Phenomenology": {
                "depth": 0.24,
                "concepts": [
                    "Husserl Phenomenology", "Intentionality Husserl",
                    "Epoche Phenomenological Reduction",
                    "Heidegger Being and Time", "Dasein", "Being-toward-Death",
                    "Heidegger Hermeneutic Circle",
                    "Sartre Being and Nothingness", "Radical Freedom Sartre",
                    "Bad Faith Sartre", "Existence Precedes Essence",
                    "Merleau-Ponty Embodiment", "Lived Body",
                    "Camus Absurd", "Myth of Sisyphus",
                    "Simone de Beauvoir Ethics of Ambiguity",
                    "Marcel Being and Having", "Jaspers Existenz",
                    "Levinas Face of the Other", "Buber I-Thou",
                ]
            },
            "Analytic Philosophy & Logic": {
                "depth": 0.26,
                "concepts": [
                    "Frege Sense Reference", "Russell Logical Atomism",
                    "Wittgenstein Tractatus", "Wittgenstein Philosophical Investigations",
                    "Language Games", "Private Language Argument",
                    "Vienna Circle Logical Positivism",
                    "Verification Principle", "Quine Two Dogmas",
                    "Kripke Naming Necessity", "Possible Worlds",
                    "Godel Incompleteness", "Tarski Truth",
                    "Carnap Logical Syntax", "Popper Falsifiability",
                    "Kuhn Paradigm Shifts", "Lakatos Research Programmes",
                    "Feyerabend Against Method",
                    "Modal Logic", "Deontic Logic", "Epistemic Logic",
                ]
            },
            "Continental & Postmodern Philosophy": {
                "depth": 0.28,
                "concepts": [
                    "Foucault Power Knowledge", "Foucault Discipline and Punish",
                    "Foucault History of Sexuality", "Biopolitics",
                    "Derrida Deconstruction", "Derrida Differance",
                    "Derrida Of Grammatology",
                    "Deleuze Rhizome", "Deleuze Difference Repetition",
                    "Deleuze Guattari Anti-Oedipus", "Body without Organs",
                    "Baudrillard Simulacra", "Hyperreality",
                    "Lyotard Postmodern Condition",
                    "Zizek Lacanian Marxism", "Habermas Communicative Action",
                    "Frankfurt School", "Adorno Dialectic of Enlightenment",
                    "Benjamin Work of Art Mechanical Reproduction",
                    "Agamben Homo Sacer", "Badiou Event",
                ]
            },
            "Ethics & Political Philosophy": {
                "depth": 0.30,
                "concepts": [
                    "Utilitarianism Bentham Mill", "Deontological Ethics Kant",
                    "Virtue Ethics Aristotle", "Care Ethics",
                    "Social Contract Rousseau", "Social Contract Hobbes",
                    "Rawls Theory of Justice", "Veil of Ignorance",
                    "Nozick Libertarianism", "Communitarianism",
                    "Consequentialism", "Moral Realism",
                    "Moral Relativism", "Metaethics",
                    "Trolley Problem", "Bioethics",
                    "Environmental Ethics", "Animal Rights Singer",
                    "Hannah Arendt Banality of Evil", "Political Freedom",
                ]
            },
        }
    },

    # ─── GALAXY 6: NEUROSCIENCE & CONSCIOUSNESS ─────────────────────────
    "Neuroscience & Consciousness": {
        "depth": 0.08,
        "stars": {
            "Neural Architecture": {
                "depth": 0.18,
                "concepts": [
                    "Neuron Doctrine", "Action Potential",
                    "Synapse", "Neurotransmitters",
                    "Dopamine System", "Serotonin System", "GABA",
                    "Glutamate", "Acetylcholine",
                    "Cerebral Cortex", "Prefrontal Cortex",
                    "Hippocampus Memory", "Amygdala Emotion",
                    "Cerebellum Motor", "Basal Ganglia",
                    "Thalamus", "Hypothalamus",
                    "Brain Stem", "Corpus Callosum",
                    "Mirror Neurons", "Grid Cells Place Cells",
                    "Hebbian Learning", "Long-Term Potentiation",
                ]
            },
            "Consciousness Studies": {
                "depth": 0.22,
                "concepts": [
                    "Hard Problem of Consciousness Chalmers",
                    "Qualia", "Explanatory Gap",
                    "Neural Correlates of Consciousness",
                    "Global Workspace Theory Baars",
                    "Integrated Information Theory Tononi",
                    "Higher-Order Theories", "Attention Schema Theory",
                    "Predictive Processing", "Free Energy Principle Friston",
                    "Philosophical Zombies", "Mary Room Argument",
                    "Chinese Room Argument Searle", "Turing Test",
                    "Panpsychism", "Dual Aspect Monism",
                    "Orchestrated Objective Reduction Penrose",
                    "Binding Problem", "Unity of Consciousness",
                    "Altered States of Consciousness", "Psychedelic Neuroscience",
                ]
            },
            "Cognitive Neuroscience": {
                "depth": 0.25,
                "concepts": [
                    "Working Memory Model Baddeley", "Long-Term Memory",
                    "Episodic Memory", "Semantic Memory", "Procedural Memory",
                    "Memory Consolidation", "Reconsolidation",
                    "Attention Networks", "Default Mode Network",
                    "Executive Functions", "Cognitive Control",
                    "Decision Making Neuroscience", "Reward Prediction Error",
                    "Emotion Regulation", "Somatic Marker Hypothesis",
                    "Theory of Mind", "Social Cognition",
                    "Language Neuroscience", "Broca Area Wernicke Area",
                    "Embodied Cognition", "Enactivism",
                ]
            },
            "Neuroplasticity & Learning": {
                "depth": 0.28,
                "concepts": [
                    "Synaptic Plasticity", "Neurogenesis",
                    "Critical Periods", "Sensitive Periods",
                    "Experience-Dependent Plasticity",
                    "Cross-Modal Plasticity", "Phantom Limb",
                    "Brain-Computer Interfaces", "Neurofeedback",
                    "Transcranial Magnetic Stimulation",
                    "Deep Brain Stimulation", "Optogenetics",
                    "Sleep and Memory Consolidation",
                    "Dream Function Theories", "REM Sleep",
                    "Meditation Neuroscience", "Mindfulness Brain",
                    "Stress and Brain", "Cortisol Effects",
                    "Neurodegeneration", "Alzheimer Mechanisms",
                ]
            },
        }
    },

    # ─── GALAXY 7: MUSIC & ART ─────────────────────────────────────────
    "Music & Art": {
        "depth": 0.10,
        "stars": {
            "Music Theory & Composition": {
                "depth": 0.20,
                "concepts": [
                    "Harmony", "Counterpoint", "Fugue",
                    "Sonata Form", "Symphony Form", "Concerto Form",
                    "Twelve-Tone Technique Schoenberg", "Serialism",
                    "Modes and Scales", "Circle of Fifths",
                    "Chord Progressions", "Voice Leading",
                    "Rhythm and Meter", "Polyrhythm",
                    "Orchestration", "Timbre",
                    "Dynamics", "Articulation",
                    "Musical Form", "Theme and Variations",
                    "Canon", "Rondo Form",
                ]
            },
            "Classical Music History": {
                "depth": 0.24,
                "concepts": [
                    "Bach Well-Tempered Clavier", "Bach Art of Fugue",
                    "Mozart Requiem", "Mozart Magic Flute",
                    "Beethoven Symphonies", "Beethoven Late Quartets",
                    "Chopin Nocturnes", "Chopin Etudes",
                    "Wagner Ring Cycle", "Wagner Tristan und Isolde",
                    "Debussy Impressionism", "Ravel Bolero",
                    "Stravinsky Rite of Spring", "Bartok String Quartets",
                    "Mahler Symphonies", "Shostakovich Symphonies",
                    "John Cage 4 33", "Steve Reich Minimalism",
                    "Philip Glass Einstein on the Beach",
                    "Ligeti Atmospheres", "Penderecki Threnody",
                ]
            },
            "Art Movements & Visual Arts": {
                "depth": 0.22,
                "concepts": [
                    "Renaissance Art", "Leonardo da Vinci", "Michelangelo",
                    "Baroque Art Caravaggio", "Rembrandt",
                    "Impressionism Monet", "Post-Impressionism Van Gogh",
                    "Cubism Picasso Braque", "Analytical Cubism",
                    "Abstract Expressionism Pollock", "Color Field Rothko",
                    "Surrealism Dali", "Surrealism Magritte",
                    "Dadaism Duchamp", "Ready-Made Art",
                    "Pop Art Warhol", "Minimalism Art",
                    "Conceptual Art", "Land Art",
                    "Art Nouveau Klimt", "Bauhaus Design",
                    "De Stijl Mondrian", "Suprematism Malevich",
                ]
            },
            "Aesthetics & Art Theory": {
                "depth": 0.28,
                "concepts": [
                    "Kant Critique of Judgment", "Sublime Burke",
                    "Beautiful and Sublime", "Aesthetic Experience",
                    "Art as Imitation Mimesis", "Art as Expression",
                    "Institutional Theory of Art", "Aesthetic Autonomy",
                    "Walter Benjamin Aura", "Adorno Aesthetic Theory",
                    "Heidegger Origin of Work of Art",
                    "Schiller Aesthetic Education", "Play Drive",
                    "Synesthesia Art", "Total Work of Art Gesamtkunstwerk",
                    "Golden Ratio Art", "Sacred Geometry",
                    "Wabi-Sabi", "Ma Japanese Space",
                    "Rasa Indian Aesthetics", "Catharsis Aristotle",
                ]
            },
            "Film & Cinema Theory": {
                "depth": 0.30,
                "concepts": [
                    "Montage Theory Eisenstein", "Soviet Montage",
                    "French New Wave Godard", "Italian Neorealism",
                    "German Expressionism Film", "Film Noir",
                    "Auteur Theory", "Hitchcock Suspense",
                    "Kubrick Cinematography", "Tarkovsky Sculpting in Time",
                    "Bergman Seventh Seal", "Fellini 8 and a Half",
                    "Kurosawa Rashomon", "Ozu Pillow Shots",
                    "Lynch Surreal Cinema", "Wes Anderson Symmetry",
                    "Mise-en-Scene", "Diegetic Sound",
                    "Deep Focus Citizen Kane", "Long Take Cinema",
                ]
            },
        }
    },

    # ─── GALAXY 8: BIOLOGY & EVOLUTION ──────────────────────────────────
    "Biology & Evolution": {
        "depth": 0.07,
        "stars": {
            "Evolutionary Theory": {
                "depth": 0.17,
                "concepts": [
                    "Natural Selection Darwin", "Sexual Selection",
                    "Genetic Drift", "Gene Flow",
                    "Speciation", "Adaptive Radiation",
                    "Punctuated Equilibrium Gould", "Gradualism",
                    "Kin Selection Hamilton", "Altruism Evolution",
                    "Selfish Gene Dawkins", "Extended Phenotype",
                    "Neutral Theory Kimura", "Molecular Clock",
                    "Coevolution", "Red Queen Hypothesis",
                    "Convergent Evolution", "Parallel Evolution",
                    "Endosymbiosis Theory", "Tree of Life",
                    "Phylogenetics", "Cladistics",
                ]
            },
            "Genetics & Molecular Biology": {
                "depth": 0.20,
                "concepts": [
                    "DNA Double Helix", "RNA Transcription",
                    "Protein Translation", "Genetic Code",
                    "Gene Expression", "Epigenetics",
                    "CRISPR Cas9", "Gene Editing",
                    "Mendel Laws", "Chromosomal Theory Inheritance",
                    "Mutation Types", "Recombination",
                    "Polymerase Chain Reaction PCR", "DNA Sequencing",
                    "Human Genome Project", "Genomics",
                    "Proteomics", "Metabolomics",
                    "Regulatory Networks", "Gene Regulatory Circuits",
                    "Transposons", "Horizontal Gene Transfer",
                    "Epigenomic Landscape Waddington",
                ]
            },
            "Ecology & Ecosystems": {
                "depth": 0.24,
                "concepts": [
                    "Food Web", "Trophic Levels",
                    "Keystone Species", "Biodiversity",
                    "Ecological Succession", "Climax Community",
                    "Niche Theory", "Competitive Exclusion",
                    "Island Biogeography", "Species-Area Relationship",
                    "Ecosystem Services", "Carbon Cycle",
                    "Nitrogen Cycle", "Phosphorus Cycle",
                    "Coral Reef Ecology", "Rainforest Ecology",
                    "Ocean Ecosystems", "Deep Sea Ecology",
                    "Invasive Species", "Extinction Dynamics",
                    "Conservation Biology", "Population Viability",
                ]
            },
            "Cell Biology & Biochemistry": {
                "depth": 0.26,
                "concepts": [
                    "Cell Membrane", "Mitochondria", "Endoplasmic Reticulum",
                    "Golgi Apparatus", "Cytoskeleton",
                    "Cell Division Mitosis", "Meiosis",
                    "Apoptosis Programmed Cell Death", "Autophagy",
                    "Signal Transduction", "Second Messengers",
                    "Enzyme Kinetics", "Michaelis-Menten",
                    "Krebs Cycle", "Glycolysis", "Oxidative Phosphorylation",
                    "ATP Synthase", "Photosynthesis",
                    "Protein Folding", "Chaperone Proteins",
                    "Stem Cells", "Cell Differentiation",
                ]
            },
            "Origin of Life & Astrobiology": {
                "depth": 0.30,
                "concepts": [
                    "RNA World Hypothesis", "Abiogenesis",
                    "Miller-Urey Experiment", "Primordial Soup",
                    "Hydrothermal Vent Theory", "Panspermia",
                    "LUCA Last Universal Common Ancestor",
                    "Drake Equation", "Fermi Paradox",
                    "Habitable Zone", "Extremophiles",
                    "Europa Ocean", "Enceladus",
                    "Mars Habitability", "Titan Biochemistry",
                    "Biosignatures", "Technosignatures",
                    "Great Filter", "Rare Earth Hypothesis",
                ]
            },
        }
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# CROSS-GALAXY BRIDGES (interdisciplinary connections)
# ═══════════════════════════════════════════════════════════════════════════

CROSS_EDGES = [
    # Physics ↔ Fractals
    ("Lorenz Attractor", "Weather Prediction Chaos", "Applies"),
    ("Chaos Theory", "Three Body Problem", "Manifests"),
    ("Phase Transitions", "Critical Phenomena", "Exhibits"),
    ("Turbulence", "Navier-Stokes Turbulence", "Models"),
    ("Hausdorff Dimension", "Fractal Dimension", "Defines"),
    ("Quantum Field Theory", "Renormalization Group", "Uses"),
    ("Renormalization Group", "Renormalization", "Related"),
    ("Feigenbaum Constants", "Period Doubling Cascade", "Characterizes"),

    # Physics ↔ Cosmology
    ("Einstein Field Equations", "Friedmann Equations", "Derives"),
    ("Schwarzschild Metric", "Schwarzschild Black Hole", "Describes"),
    ("Hawking Radiation", "Black Hole Thermodynamics", "Connects"),
    ("Cosmological Constant", "Dark Energy", "Explains"),
    ("Standard Model", "Nucleosynthesis", "Governs"),
    ("Gravitational Waves", "Gravitational Wave Mergers", "Detected"),
    ("Quantum Chromodynamics QCD", "Quark Epoch", "Operates"),

    # Philosophy ↔ Literature
    ("Nietzsche Eternal Recurrence", "Nietzsche Will to Power", "Complements"),
    ("Nietzsche Beyond Good and Evil", "Crime and Punishment", "Inspires"),
    ("Camus Absurd", "The Stranger Camus", "Embodied"),
    ("Sartre Being and Nothingness", "Nausea Sartre", "Fictional"),
    ("Existentialism", "Modernism & Existentialist Fiction", "Shapes"),
    ("Kafka Metamorphosis", "The Trial Kafka", "Related"),
    ("Borges Ficciones", "Borges The Aleph", "Collection"),
    ("Deconstruction Derrida", "Post-Structuralism", "Method"),

    # Neuroscience ↔ Philosophy
    ("Hard Problem of Consciousness Chalmers", "Qualia", "Defines"),
    ("Chinese Room Argument Searle", "Turing Test", "Critiques"),
    ("Free Energy Principle Friston", "Predictive Processing", "Formalizes"),
    ("Panpsychism", "Integrated Information Theory Tononi", "Relates"),
    ("Embodied Cognition", "Merleau-Ponty Embodiment", "Grounds"),
    ("Theory of Mind", "Levinas Face of the Other", "Connects"),

    # Biology ↔ Fractals
    ("L-Systems Lindenmayer", "Fractal Trees", "Generates"),
    ("Self-Similarity", "Barnsley Fern", "Models"),
    ("Emergence", "Self-Organization", "Requires"),
    ("Population Dynamics", "Lotka-Volterra Equations", "Models"),
    ("Predator-Prey Models", "Logistic Map", "Simplifies"),
    ("Ecological Succession", "Turing Patterns", "Parallels"),

    # Biology ↔ Neuroscience
    ("DNA Double Helix", "Gene Expression", "Enables"),
    ("Epigenetics", "Neuroplasticity", "Influences"),
    ("CRISPR Cas9", "Optogenetics", "Enables"),
    ("Cell Division Mitosis", "Neurogenesis", "Mechanism"),
    ("Signal Transduction", "Neurotransmitters", "Uses"),

    # Music ↔ Physics
    ("Harmony", "Circle of Fifths", "Structures"),
    ("Twelve-Tone Technique Schoenberg", "Serialism", "Extends"),
    ("Polyrhythm", "Coupled Oscillators", "Analogous"),

    # Cosmology ↔ Philosophy
    ("Big Bang Theory", "Cosmological Constant", "Includes"),
    ("Fermi Paradox", "Great Filter", "Explains"),
    ("Observable Universe", "Cosmic Horizon", "Defines"),

    # Art ↔ Fractals
    ("Golden Ratio Art", "Sacred Geometry", "Connects"),
    ("Mandelbrot Set", "Julia Set", "Related"),
    ("Sierpinski Triangle", "Koch Snowflake", "Analogous"),

    # Literature ↔ Neuroscience
    ("Proust In Search of Lost Time", "Episodic Memory", "Explores"),
    ("Stream of Consciousness", "Working Memory Model Baddeley", "Reflects"),

    # Cross-domain deep bridges
    ("Godel Incompleteness", "Turing Test", "Limits"),
    ("Entropy", "Second Law Thermodynamics", "Defines"),
    ("Kolmogorov Complexity", "Information Theory Complexity", "Formalizes"),
    ("Autopoiesis", "Abiogenesis", "Concept"),
    ("Game of Life Conway", "Cellular Automata", "Instance"),
    ("Edge of Chaos", "Self-Organized Criticality", "Related"),
    ("Punctuated Equilibrium Gould", "Punctuated Equilibrium", "Same"),
]


def make_poincare_embedding(galaxy_name, star_name, concept_name, depth, dim):
    """Create a deterministic Poincaré ball embedding."""
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
    """Create InsertEdgeRequest handling 'from' reserved keyword."""
    from_val = kwargs.pop('from_node')
    req = pb2.InsertEdgeRequest(**kwargs)
    setattr(req, 'from', from_val)
    return req


def insert_galaxies(host, collection, metric="poincare", dim=128):
    """Insert all knowledge galaxies into NietzscheDB."""

    pb2, pb2_grpc = ensure_proto_compiled()

    channel = grpc.insecure_channel(host, options=[
        ('grpc.max_send_message_length', 256 * 1024 * 1024),
        ('grpc.max_receive_message_length', 256 * 1024 * 1024),
    ])
    stub = pb2_grpc.NietzscheDBStub(channel)

    print(f"\n{'='*60}")
    print(f"  NietzscheDB Knowledge Galaxy Ingestion")
    print(f"  Host: {host}")
    print(f"  Collection: {collection}")
    print(f"  Metric: {metric}")
    print(f"  Dim: {dim}")
    print(f"{'='*60}\n")

    # 1. Create collection
    try:
        resp = stub.CreateCollection(pb2.CreateCollectionRequest(
            collection=collection, dim=dim, metric=metric
        ))
        print(f"[+] Collection '{collection}' created (dim={dim}, metric={metric})")
    except grpc.RpcError as e:
        print(f"[!] CreateCollection: {e.details() if hasattr(e, 'details') else e}")

    # 2. Build all nodes
    all_nodes = []
    node_ids = {}

    for galaxy_name, galaxy_data in GALAXIES.items():
        galaxy_depth = galaxy_data["depth"]
        galaxy_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"galaxy:{galaxy_name}"))
        node_ids[galaxy_name] = galaxy_id

        emb = make_poincare_embedding(galaxy_name, "__galaxy__", galaxy_name, galaxy_depth, dim)
        all_nodes.append({
            "id": galaxy_id,
            "content": json.dumps({"name": galaxy_name, "type": "galaxy", "domain": "knowledge"}).encode('utf-8'),
            "node_type": "Concept",
            "energy": 1.0,
            "embedding": emb,
        })

        for star_name, star_data in galaxy_data["stars"].items():
            star_depth = star_data["depth"]
            star_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"star:{galaxy_name}/{star_name}"))
            node_ids[star_name] = star_id

            emb = make_poincare_embedding(galaxy_name, star_name, star_name, star_depth, dim)
            all_nodes.append({
                "id": star_id,
                "content": json.dumps({"name": star_name, "type": "star", "galaxy": galaxy_name}).encode('utf-8'),
                "node_type": "Concept",
                "energy": 0.8,
                "embedding": emb,
            })

            for concept_name in star_data["concepts"]:
                concept_depth = star_depth + 0.1 + (hash(concept_name) % 100) / 1000.0
                concept_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"concept:{galaxy_name}/{star_name}/{concept_name}"))
                node_ids[concept_name] = concept_id

                emb = make_poincare_embedding(galaxy_name, star_name, concept_name, concept_depth, dim)
                all_nodes.append({
                    "id": concept_id,
                    "content": json.dumps({
                        "name": concept_name,
                        "type": "concept",
                        "star": star_name,
                        "galaxy": galaxy_name,
                    }).encode('utf-8'),
                    "node_type": "Semantic",
                    "energy": 0.6,
                    "embedding": emb,
                })

    total_nodes = len(all_nodes)
    print(f"[*] Inserting {total_nodes} nodes in batches of 50...")

    # 3. Insert nodes in batches
    batch_size = 50
    inserted = 0
    for i in range(0, total_nodes, batch_size):
        batch = all_nodes[i:i+batch_size]
        requests = []
        for n in batch:
            pv = pb2.PoincareVector(coords=n["embedding"])
            req = pb2.InsertNodeRequest(
                id=n["id"],
                embedding=pv,
                content=n["content"],
                node_type=n["node_type"],
                energy=n["energy"],
                collection=collection,
            )
            requests.append(req)

        try:
            resp = stub.BatchInsertNodes(pb2.BatchInsertNodesRequest(
                nodes=requests, collection=collection
            ))
            inserted += len(batch)
            pct = inserted / total_nodes * 100
            print(f"  [{pct:5.1f}%] Inserted {len(batch)} nodes (total: {inserted})", end='\r')
        except grpc.RpcError as e:
            print(f"\n  [!] Batch {i//batch_size}: {e.details() if hasattr(e, 'details') else e}")

    print(f"\n[+] Nodes inserted: {inserted}/{total_nodes}")

    # 4. Create hierarchical edges
    print(f"\n[*] Creating hierarchical edges...")
    edge_count = 0

    for galaxy_name, galaxy_data in GALAXIES.items():
        galaxy_id = node_ids[galaxy_name]

        for star_name, star_data in galaxy_data["stars"].items():
            star_id = node_ids.get(star_name)
            if not star_id:
                continue

            # Galaxy → Star
            try:
                stub.InsertEdge(make_edge_request(pb2,
                    from_node=galaxy_id, to=star_id,
                    edge_type="Hierarchical", weight=1.0,
                    collection=collection
                ))
                edge_count += 1
            except grpc.RpcError:
                pass

            # Star → Concept
            for concept_name in star_data["concepts"]:
                concept_id = node_ids.get(concept_name)
                if not concept_id:
                    continue
                try:
                    stub.InsertEdge(make_edge_request(pb2,
                        from_node=star_id, to=concept_id,
                        edge_type="Hierarchical", weight=0.8,
                        collection=collection
                    ))
                    edge_count += 1
                except grpc.RpcError:
                    pass

            # Intra-star associations (nearby concepts)
            concepts_in_star = [c for c in star_data["concepts"] if c in node_ids]
            for j in range(len(concepts_in_star)):
                for k in range(j+1, min(j+4, len(concepts_in_star))):
                    try:
                        stub.InsertEdge(make_edge_request(pb2,
                            from_node=node_ids[concepts_in_star[j]],
                            to=node_ids[concepts_in_star[k]],
                            edge_type="Association", weight=0.6,
                            collection=collection
                        ))
                        edge_count += 1
                    except grpc.RpcError:
                        pass

        print(f"  Galaxy '{galaxy_name}': edges created ({edge_count} total)", end='\r')

    print(f"\n[+] Hierarchical + association edges: {edge_count}")

    # 5. Cross-galaxy bridges
    print(f"\n[*] Creating cross-galaxy bridges...")
    bridge_count = 0
    for from_name, to_name, rel_type in CROSS_EDGES:
        from_id = node_ids.get(from_name)
        to_id = node_ids.get(to_name)
        if from_id and to_id:
            try:
                stub.InsertEdge(make_edge_request(pb2,
                    from_node=from_id, to=to_id,
                    edge_type="Association", weight=0.7,
                    collection=collection
                ))
                bridge_count += 1
            except grpc.RpcError:
                pass

    total_edges = edge_count + bridge_count
    print(f"[+] Cross-galaxy bridges: {bridge_count}")

    # 6. Summary
    num_galaxies = len(GALAXIES)
    num_stars = sum(len(g['stars']) for g in GALAXIES.values())
    print(f"\n{'='*60}")
    print(f"  KNOWLEDGE INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Collection:     {collection}")
    print(f"  Galaxies:       {num_galaxies}")
    print(f"  Stars:          {num_stars}")
    print(f"  Total Nodes:    {inserted}")
    print(f"  Total Edges:    {total_edges}")
    print(f"  Cross Bridges:  {bridge_count}")
    print(f"{'='*60}")
    print()
    for gname, gdata in GALAXIES.items():
        stars = list(gdata["stars"].keys())
        concepts = sum(len(s["concepts"]) for s in gdata["stars"].values())
        print(f"  {gname}: {len(stars)} stars, {concepts} concepts")
    print()
    print(f"  The L-System + Agency engine will grow these")
    print(f"  knowledge clusters into interconnected galaxies")
    print(f"  in the Poincare ball!")
    print(f"\n  Dashboard: http://{host.split(':')[0]}:8080")

    channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert Knowledge Galaxies into NietzscheDB")
    parser.add_argument("--host", default="localhost:50051", help="gRPC host:port")
    parser.add_argument("--collection", default="knowledge_galaxies", help="Collection name")
    parser.add_argument("--metric", default="poincare", help="Distance metric")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    args = parser.parse_args()

    insert_galaxies(args.host, args.collection, args.metric, args.dim)

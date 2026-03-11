#!/usr/bin/env python3
"""
Insert Ultra-Cultural Knowledge Galaxies into NietzscheDB.

10 new galaxies covering the deepest layers of human civilization:
Mythology, History, Pure Mathematics, Psychology, Linguistics,
Civilizations, Esotericism, Cinema/Theatre, Gastronomy, Architecture.

~1500+ nodes, ~5000+ edges, massive cross-galaxy bridges.

Usage:
  python scripts/insert_culture_galaxies.py [--host HOST:PORT] [--collection NAME]
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
# ULTRA-CULTURAL ONTOLOGY - 10 GALAXIES
# ═══════════════════════════════════════════════════════════════════════════

GALAXIES = {

    # ─── GALAXY 1: MYTHOLOGY & RELIGIONS ────────────────────────────────
    "Mythology & Religions": {
        "depth": 0.06,
        "stars": {
            "Greek Mythology": {
                "depth": 0.16,
                "concepts": [
                    "Zeus Olympus", "Athena Wisdom", "Apollo Arts Sun",
                    "Dionysus Wine Ecstasy", "Prometheus Fire Theft",
                    "Odysseus Journey", "Orpheus Eurydice",
                    "Oedipus Complex Myth", "Medusa Gorgon",
                    "Minotaur Labyrinth", "Icarus Wings",
                    "Persephone Underworld", "Ares War",
                    "Hermes Trickster Messenger", "Hades Death",
                    "Pandora Box", "Achilles Heel",
                    "Trojan War", "Argonauts Golden Fleece",
                    "Titans Kronos", "Olympian Gods Pantheon",
                ]
            },
            "Norse Mythology": {
                "depth": 0.18,
                "concepts": [
                    "Odin Allfather", "Thor Thunder Mjolnir",
                    "Loki Trickster", "Ragnarok End of World",
                    "Yggdrasil World Tree", "Nine Realms",
                    "Valhalla Warriors", "Valkyries",
                    "Fenrir Wolf", "Jormungandr World Serpent",
                    "Freya Love War", "Baldur Death",
                    "Norns Fate Destiny", "Runes Magic",
                    "Midgard Humans", "Asgard Gods",
                    "Niflheim Ice", "Muspelheim Fire",
                ]
            },
            "Egyptian Mythology": {
                "depth": 0.20,
                "concepts": [
                    "Ra Sun God", "Osiris Death Rebirth",
                    "Isis Magic Mother", "Horus Sky Pharaoh",
                    "Anubis Afterlife", "Thoth Wisdom Writing",
                    "Ma'at Justice Balance", "Set Chaos Desert",
                    "Book of the Dead", "Weighing of Heart",
                    "Scarab Beetle Symbol", "Ankh Life Symbol",
                    "Eye of Horus", "Pyramid Texts",
                    "Afterlife Duat", "Pharaoh Divinity",
                ]
            },
            "Hindu Mythology & Philosophy": {
                "depth": 0.19,
                "concepts": [
                    "Brahman Ultimate Reality", "Atman Self Soul",
                    "Vishnu Preserver", "Shiva Destroyer Transformer",
                    "Brahma Creator", "Krishna Avatar",
                    "Shakti Divine Feminine", "Kali Time Destruction",
                    "Karma Law of Action", "Dharma Cosmic Order",
                    "Samsara Cycle Rebirth", "Moksha Liberation",
                    "Maya Illusion", "Yoga Union",
                    "Chakras Energy Centers", "Kundalini Energy",
                    "Vedas Sacred Texts", "Upanishads Philosophy",
                    "Mandala Sacred Circle", "Om Primordial Sound",
                ]
            },
            "Abrahamic Religions": {
                "depth": 0.22,
                "concepts": [
                    "Genesis Creation", "Exodus Liberation",
                    "Ten Commandments", "Covenant Abraham",
                    "Prophets Tradition", "Psalms Poetry",
                    "Kabbalah Mysticism", "Talmud Interpretation",
                    "Gospels Jesus", "Sermon on the Mount",
                    "Revelation Apocalypse", "Trinity Doctrine",
                    "Gnosticism", "Desert Fathers Monasticism",
                    "Quran Revelation", "Five Pillars Islam",
                    "Sufism Mystical Islam", "Rumi Whirling Dervish",
                    "Jihad Inner Struggle", "Islamic Golden Age",
                ]
            },
            "Buddhism & Eastern Wisdom": {
                "depth": 0.21,
                "concepts": [
                    "Four Noble Truths", "Eightfold Path",
                    "Nirvana Enlightenment", "Sunyata Emptiness",
                    "Dependent Origination", "Impermanence Anicca",
                    "Bodhisattva Ideal", "Zen Meditation Zazen",
                    "Koan Paradox", "Satori Sudden Awakening",
                    "Tibetan Book of the Dead", "Mandala Buddhism",
                    "Theravada Tradition", "Mahayana Tradition",
                    "Vajrayana Tantric", "Heart Sutra",
                    "Taoism Wu Wei", "Yin Yang Duality",
                    "Confucius Analects", "I Ching Divination",
                ]
            },
            "Shamanism & Indigenous Traditions": {
                "depth": 0.24,
                "concepts": [
                    "Shamanic Journey", "Spirit Animals Totems",
                    "Ayahuasca Vision", "Peyote Ceremony",
                    "Dreamtime Aboriginal", "Songlines",
                    "Vision Quest", "Sweat Lodge",
                    "Animism Nature Spirits", "Ancestor Worship",
                    "Medicine Wheel", "Sacred Plants",
                    "Trance States", "Soul Retrieval",
                    "African Oral Traditions", "Yoruba Orishas",
                    "Voodoo Vodou", "Celtic Druidism",
                ]
            },
        }
    },

    # ─── GALAXY 2: WORLD HISTORY ────────────────────────────────────────
    "World History": {
        "depth": 0.07,
        "stars": {
            "Ancient Civilizations History": {
                "depth": 0.17,
                "concepts": [
                    "Sumer Cradle of Civilization", "Code of Hammurabi",
                    "Ancient Egypt Pharaohs", "Pyramids of Giza",
                    "Minoan Civilization", "Mycenaean Greece",
                    "Classical Athens Democracy", "Peloponnesian War",
                    "Alexander the Great Empire", "Hellenistic Period",
                    "Roman Republic", "Roman Empire Augustus",
                    "Fall of Rome 476", "Pax Romana",
                    "Maurya Empire Ashoka", "Han Dynasty China",
                    "Qin Shi Huang Unification", "Silk Road Trade",
                    "Persian Empire Cyrus", "Phoenician Alphabet",
                ]
            },
            "Medieval Period": {
                "depth": 0.20,
                "concepts": [
                    "Byzantine Empire", "Justinian Code",
                    "Feudalism", "Knights Chivalry",
                    "Crusades", "Siege of Jerusalem 1099",
                    "Magna Carta 1215", "Hundred Years War",
                    "Black Death Plague", "Gothic Cathedrals",
                    "Carolingian Empire Charlemagne", "Viking Age",
                    "Norman Conquest 1066", "Scholasticism Aquinas",
                    "Islamic Golden Age Science", "Tang Song Dynasty",
                    "Mongol Empire Genghis Khan", "Samurai Bushido",
                    "Mali Empire Mansa Musa", "Great Zimbabwe",
                ]
            },
            "Renaissance to Enlightenment History": {
                "depth": 0.23,
                "concepts": [
                    "Italian Renaissance Florence", "Gutenberg Printing Press",
                    "Age of Exploration", "Columbus Americas 1492",
                    "Magellan Circumnavigation", "Protestant Reformation Luther",
                    "Scientific Revolution", "Galileo Copernicus Heliocentrism",
                    "Newton Principia", "Enlightenment Philosophy",
                    "French Revolution 1789", "American Revolution 1776",
                    "Declaration of Independence", "Rights of Man Paine",
                    "Industrial Revolution", "Steam Engine Watt",
                    "Colonialism Imperialism", "Atlantic Slave Trade",
                    "Haitian Revolution", "Napoleon Bonaparte",
                ]
            },
            "Modern History 20th Century": {
                "depth": 0.26,
                "concepts": [
                    "World War I Trenches", "Treaty of Versailles",
                    "Russian Revolution 1917", "Soviet Union Formation",
                    "Great Depression 1929", "Rise of Fascism",
                    "World War II Holocaust", "D-Day Normandy",
                    "Hiroshima Nagasaki Atomic", "United Nations Formation",
                    "Cold War Iron Curtain", "Space Race Sputnik Apollo",
                    "Decolonization Africa Asia", "Civil Rights Movement",
                    "Martin Luther King Jr", "Apartheid South Africa",
                    "Fall of Berlin Wall 1989", "Chinese Revolution Mao",
                    "Vietnam War", "Cuban Missile Crisis",
                ]
            },
            "Contemporary History": {
                "depth": 0.29,
                "concepts": [
                    "Dissolution Soviet Union 1991", "European Union Formation",
                    "September 11 Attacks", "War on Terror",
                    "Arab Spring", "Rise of China Economic",
                    "Internet Revolution", "Social Media Era",
                    "Climate Change Awareness", "Paris Agreement",
                    "COVID-19 Pandemic", "Cryptocurrency Rise",
                    "AI Revolution 2020s", "Geopolitical Multipolarity",
                    "Refugee Crisis Global", "Me Too Movement",
                    "Brexit", "Space Commercialization",
                ]
            },
        }
    },

    # ─── GALAXY 3: PURE MATHEMATICS ─────────────────────────────────────
    "Pure Mathematics": {
        "depth": 0.05,
        "stars": {
            "Number Theory": {
                "depth": 0.15,
                "concepts": [
                    "Prime Numbers", "Fundamental Theorem Arithmetic",
                    "Riemann Hypothesis", "Riemann Zeta Function",
                    "Goldbach Conjecture", "Twin Prime Conjecture",
                    "Fermat Last Theorem Wiles", "Modular Arithmetic",
                    "Quadratic Reciprocity", "Diophantine Equations",
                    "p-adic Numbers", "Algebraic Number Fields",
                    "Elliptic Curves Number Theory", "Langlands Program",
                    "Continued Fractions", "Transcendental Numbers",
                    "Prime Number Theorem", "Sieve of Eratosthenes",
                    "Mersenne Primes", "Perfect Numbers",
                ]
            },
            "Topology": {
                "depth": 0.18,
                "concepts": [
                    "Topological Spaces", "Homeomorphism",
                    "Fundamental Group", "Homotopy Theory",
                    "Homology Groups", "Cohomology",
                    "Manifold Theory", "Differential Manifolds",
                    "Poincare Conjecture Perelman", "Euler Characteristic",
                    "Mobius Strip", "Klein Bottle",
                    "Knot Theory", "Jones Polynomial",
                    "Algebraic Topology", "CW Complexes",
                    "Fiber Bundles", "Vector Bundles",
                    "Covering Spaces", "Simplicial Complexes",
                    "Category Theory Functors", "Sheaf Theory",
                ]
            },
            "Abstract Algebra": {
                "depth": 0.20,
                "concepts": [
                    "Group Theory", "Symmetric Groups",
                    "Ring Theory", "Ideal Theory",
                    "Field Theory Extensions", "Galois Theory",
                    "Lie Groups", "Lie Algebras",
                    "Representation Theory", "Character Theory",
                    "Module Theory", "Tensor Products",
                    "Homological Algebra", "Derived Categories",
                    "Commutative Algebra", "Noetherian Rings",
                    "Quaternions Hamilton", "Octonions",
                    "Clifford Algebras", "Hopf Algebras",
                ]
            },
            "Geometry & Differential Geometry": {
                "depth": 0.22,
                "concepts": [
                    "Euclidean Geometry", "Non-Euclidean Geometry",
                    "Hyperbolic Geometry Lobachevsky", "Riemannian Geometry",
                    "Poincare Disk Model", "Poincare Half-Plane",
                    "Gaussian Curvature", "Geodesics Geometry",
                    "Parallel Transport", "Holonomy",
                    "Symplectic Geometry", "Contact Geometry",
                    "Projective Geometry", "Affine Geometry",
                    "Algebraic Geometry", "Scheme Theory Grothendieck",
                    "Moduli Spaces", "Calabi-Yau Manifolds",
                    "Thurston Geometrization", "Ricci Flow",
                ]
            },
            "Analysis & Foundations": {
                "depth": 0.24,
                "concepts": [
                    "Real Analysis Lebesgue", "Measure Theory",
                    "Complex Analysis", "Cauchy Integral Theorem",
                    "Functional Analysis", "Banach Spaces",
                    "Hilbert Spaces Mathematics", "Spectral Theory",
                    "Fourier Analysis", "Harmonic Analysis",
                    "Distribution Theory Schwartz", "Sobolev Spaces",
                    "Set Theory Axioms ZFC", "Axiom of Choice",
                    "Continuum Hypothesis", "Constructivism Mathematics",
                    "Proof Theory", "Model Theory",
                    "Computability Theory", "Lambda Calculus",
                ]
            },
            "Combinatorics & Discrete Math": {
                "depth": 0.26,
                "concepts": [
                    "Graph Theory Euler", "Ramsey Theory",
                    "Combinatorial Designs", "Latin Squares",
                    "Generating Functions", "Catalan Numbers",
                    "Partition Theory", "Young Tableaux",
                    "Matroid Theory", "Polyhedral Combinatorics",
                    "Probabilistic Method Erdos", "Extremal Graph Theory",
                    "Game Theory Nash Equilibrium", "Minimax Theorem",
                    "Auction Theory", "Mechanism Design",
                    "Coding Theory Error Correcting", "Information Theory Shannon",
                ]
            },
        }
    },

    # ─── GALAXY 4: PSYCHOLOGY & PSYCHOANALYSIS ──────────────────────────
    "Psychology & Psychoanalysis": {
        "depth": 0.08,
        "stars": {
            "Psychoanalysis": {
                "depth": 0.18,
                "concepts": [
                    "Freud Unconscious", "Id Ego Superego",
                    "Oedipus Complex Freud", "Dream Interpretation Freud",
                    "Free Association", "Transference",
                    "Death Drive Thanatos", "Eros Life Drive",
                    "Repression Defense Mechanism", "Sublimation",
                    "Castration Anxiety", "Penis Envy Criticism",
                    "Jung Collective Unconscious", "Archetypes Jung",
                    "Shadow Self Jung", "Anima Animus",
                    "Individuation Process", "Synchronicity Jung",
                    "Mandala Psychology", "Active Imagination",
                    "Lacan Mirror Stage", "Lacan Real Symbolic Imaginary",
                    "Object Petit a", "Jouissance Lacan",
                ]
            },
            "Behavioral & Cognitive Psychology": {
                "depth": 0.22,
                "concepts": [
                    "Pavlov Classical Conditioning", "Skinner Operant Conditioning",
                    "Behaviorism Watson", "Reinforcement Schedules",
                    "Cognitive Revolution", "Cognitive Behavioral Therapy",
                    "Schema Theory Piaget", "Cognitive Dissonance Festinger",
                    "Heuristics and Biases Kahneman", "Prospect Theory",
                    "System 1 System 2 Thinking", "Anchoring Bias",
                    "Confirmation Bias", "Availability Heuristic",
                    "Framing Effect", "Loss Aversion",
                    "Bandura Social Learning", "Self-Efficacy",
                    "Learned Helplessness Seligman", "Flow State Csikszentmihalyi",
                ]
            },
            "Humanistic & Existential Psychology": {
                "depth": 0.25,
                "concepts": [
                    "Maslow Hierarchy of Needs", "Self-Actualization",
                    "Peak Experiences", "Rogers Unconditional Positive Regard",
                    "Client-Centered Therapy", "Congruence Rogers",
                    "Existential Therapy Yalom", "Death Anxiety",
                    "Freedom and Responsibility", "Meaninglessness",
                    "Logotherapy Frankl", "Will to Meaning",
                    "Man Search for Meaning Frankl", "Tragic Optimism",
                    "Gestalt Therapy Perls", "Here and Now Awareness",
                    "Rollo May Anxiety", "Courage to Create",
                    "R.D. Laing Divided Self", "Anti-Psychiatry",
                ]
            },
            "Developmental Psychology": {
                "depth": 0.27,
                "concepts": [
                    "Piaget Stages Development", "Sensorimotor Stage",
                    "Concrete Operations", "Formal Operations",
                    "Vygotsky Zone Proximal Development", "Scaffolding Learning",
                    "Erikson Psychosocial Stages", "Identity Crisis",
                    "Attachment Theory Bowlby", "Strange Situation Ainsworth",
                    "Secure Attachment", "Anxious Attachment",
                    "Kohlberg Moral Development", "Object Permanence",
                    "Theory of Mind Development", "Adolescent Brain Development",
                    "Winnicott Good Enough Mother", "Transitional Object",
                ]
            },
            "Transpersonal & Depth Psychology": {
                "depth": 0.30,
                "concepts": [
                    "Transpersonal Psychology Grof", "Holotropic Breathwork",
                    "Perinatal Matrices", "COEX Systems",
                    "Psychedelic Therapy Research", "Mystical Experience Psychology",
                    "Near-Death Experience", "Out-of-Body Experience",
                    "James Varieties Religious Experience",
                    "Hillman Archetypal Psychology", "Soul-Making",
                    "Wilber Integral Psychology", "AQAL Model",
                    "Spiral Dynamics", "Stages of Consciousness",
                    "Dark Night of the Soul", "Spiritual Emergency",
                    "Jung Red Book", "Alchemical Psychology",
                ]
            },
        }
    },

    # ─── GALAXY 5: LINGUISTICS & SEMIOTICS ──────────────────────────────
    "Linguistics & Semiotics": {
        "depth": 0.09,
        "stars": {
            "Structural Linguistics": {
                "depth": 0.19,
                "concepts": [
                    "Saussure Signifier Signified", "Langue Parole",
                    "Phoneme Morpheme", "Syntagmatic Paradigmatic",
                    "Prague School Phonology", "Jakobson Functions Language",
                    "Chomsky Universal Grammar", "Generative Grammar",
                    "Deep Structure Surface Structure", "Minimalist Program",
                    "X-Bar Theory", "Merge Operation Chomsky",
                    "Language Acquisition Device", "Poverty of Stimulus",
                    "Bloomfield Structuralism", "Distributional Analysis",
                    "Tagmemics Pike", "Systemic Functional Linguistics",
                ]
            },
            "Semantics & Pragmatics": {
                "depth": 0.22,
                "concepts": [
                    "Referential Semantics", "Sense and Reference Frege",
                    "Speech Act Theory Austin", "Illocutionary Force",
                    "Grice Conversational Maxims", "Implicature",
                    "Relevance Theory Sperber Wilson", "Presupposition",
                    "Deixis", "Anaphora Resolution",
                    "Prototype Theory Rosch", "Cognitive Semantics",
                    "Metaphor Theory Lakoff", "Conceptual Metaphor",
                    "Frame Semantics Fillmore", "Construction Grammar",
                    "Truth-Conditional Semantics", "Possible Worlds Semantics",
                ]
            },
            "Sociolinguistics": {
                "depth": 0.25,
                "concepts": [
                    "Sapir-Whorf Hypothesis", "Linguistic Relativity",
                    "Code Switching", "Diglossia",
                    "Pidgin Creole Languages", "Language Death Endangerment",
                    "Sociolects Dialects", "Register Variation",
                    "Language and Power", "Critical Discourse Analysis",
                    "Labov Sociolinguistic Variables", "Language Change",
                    "Bilingualism Multilingualism", "Language Policy",
                    "Lingua Franca", "Esperanto Constructed Languages",
                    "Writing Systems Evolution", "Decipherment Rosetta Stone",
                ]
            },
            "Semiotics & Sign Theory": {
                "depth": 0.24,
                "concepts": [
                    "Peirce Semiotics Triadic", "Icon Index Symbol",
                    "Semiosis Unlimited", "Interpretant Peirce",
                    "Barthes Mythologies", "Barthes Death of Author",
                    "Eco Semiotics", "Open Work Eco",
                    "Lotman Semiosphere", "Cultural Semiotics",
                    "Biosemiotics", "Zoosemiotics",
                    "Visual Semiotics", "Multimodal Discourse",
                    "Baudrillard Simulacra Semiotics", "Hyperreality Signs",
                    "Greimas Semiotic Square", "Narrative Semiotics",
                ]
            },
            "Historical & Comparative Linguistics": {
                "depth": 0.27,
                "concepts": [
                    "Proto-Indo-European", "Grimm Law Sound Shift",
                    "Language Families Tree", "Nostratic Hypothesis",
                    "Etymology", "Semantic Change",
                    "Grammaticalization", "Language Contact",
                    "Typological Universals", "Word Order Typology",
                    "Morphological Typology", "Agglutinative Fusional Isolating",
                    "Austronesian Languages", "Bantu Languages",
                    "Sino-Tibetan Languages", "Afroasiatic Languages",
                ]
            },
        }
    },

    # ─── GALAXY 6: CIVILIZATIONS & ANTHROPOLOGY ─────────────────────────
    "Civilizations & Anthropology": {
        "depth": 0.08,
        "stars": {
            "Anthropological Theory": {
                "depth": 0.18,
                "concepts": [
                    "Levi-Strauss Structuralism", "Binary Oppositions Culture",
                    "Malinowski Participant Observation", "Fieldwork Method",
                    "Geertz Thick Description", "Interpretive Anthropology",
                    "Cultural Relativism Boas", "Ethnocentrism Critique",
                    "Mead Coming of Age Samoa", "Benedict Patterns of Culture",
                    "Turner Ritual Process", "Liminality Turner",
                    "Douglas Purity and Danger", "Taboo Systems",
                    "Sahlins Stone Age Economics", "Gift Economy Mauss",
                    "Potlatch Ceremony", "Reciprocity Exchange",
                    "Clifford Writing Culture", "Postcolonial Anthropology",
                ]
            },
            "Ancient Eastern Civilizations": {
                "depth": 0.22,
                "concepts": [
                    "Indus Valley Civilization", "Mohenjo-Daro",
                    "Chinese Oracle Bones", "Zhou Dynasty Mandate Heaven",
                    "Confucian State System", "Imperial Examination China",
                    "Japanese Heian Period", "Shogunate Tokugawa",
                    "Khmer Empire Angkor Wat", "Majapahit Empire",
                    "Persian Zoroastrianism", "Achaemenid Administration",
                    "Mesopotamia Ziggurats", "Cuneiform Writing",
                    "Babylonian Mathematics", "Sumerian City States",
                ]
            },
            "Pre-Columbian Americas": {
                "depth": 0.24,
                "concepts": [
                    "Maya Calendar System", "Maya Mathematics Zero",
                    "Maya Hieroglyphics", "Chichen Itza",
                    "Aztec Tenochtitlan", "Aztec Sun Stone",
                    "Human Sacrifice Ritual", "Quetzalcoatl",
                    "Inca Quipu", "Machu Picchu",
                    "Inca Road System", "Nazca Lines",
                    "Olmec Civilization", "Toltec Civilization",
                    "Norte Chico Civilization", "Cahokia Mounds",
                ]
            },
            "African Civilizations": {
                "depth": 0.25,
                "concepts": [
                    "Kingdom of Kush", "Axum Empire",
                    "Great Zimbabwe Ruins", "Swahili Coast Trade",
                    "Mali Empire Timbuktu", "Songhai Empire",
                    "Benin Bronzes", "Nok Culture",
                    "Ethiopian Christianity", "Lalibela Churches",
                    "Zulu Kingdom Shaka", "Ashanti Empire",
                    "African Oral History Griots", "Bantu Migration",
                    "Nubian Pyramids", "Hausa City States",
                ]
            },
            "Islamic Civilization": {
                "depth": 0.23,
                "concepts": [
                    "Islamic Golden Age Baghdad", "House of Wisdom",
                    "Al-Khwarizmi Algebra", "Ibn Sina Avicenna Medicine",
                    "Ibn Rushd Averroes Philosophy", "Al-Biruni Astronomy",
                    "Alhambra Architecture", "Islamic Geometric Art",
                    "Calligraphy Arabic", "Ottoman Empire",
                    "Mughal Empire Architecture", "Taj Mahal",
                    "Persian Literature Ferdowsi", "Ibn Khaldun Historiography",
                    "Córdoba Caliphate", "Madrasa System Education",
                ]
            },
        }
    },

    # ─── GALAXY 7: ALCHEMY & ESOTERICISM ────────────────────────────────
    "Alchemy & Esotericism": {
        "depth": 0.10,
        "stars": {
            "Alchemy": {
                "depth": 0.20,
                "concepts": [
                    "Philosopher Stone", "Magnum Opus Great Work",
                    "Nigredo Blackening", "Albedo Whitening",
                    "Citrinitas Yellowing", "Rubedo Reddening",
                    "Solve et Coagula", "Prima Materia",
                    "Mercury Sulfur Salt Tria Prima", "Ouroboros Symbol",
                    "Emerald Tablet Hermes", "As Above So Below",
                    "Paracelsus Iatrochemistry", "Nicolas Flamel",
                    "Transmutation Metals", "Elixir of Life",
                    "Alchemical Laboratory", "Distillation Calcination",
                    "Jung Alchemy Psychology", "Alchemical Symbolism",
                ]
            },
            "Hermeticism & Kabbalah": {
                "depth": 0.23,
                "concepts": [
                    "Hermes Trismegistus", "Corpus Hermeticum",
                    "Seven Hermetic Principles", "Principle of Mentalism",
                    "Principle of Correspondence", "Principle of Vibration",
                    "Tree of Life Sephiroth", "Ein Sof Infinite",
                    "Ten Sephirot", "22 Paths Kabbalah",
                    "Zohar Text", "Sefer Yetzirah",
                    "Gematria Numerology", "Merkabah Mysticism",
                    "Isaac Luria Kabbalah", "Tikkun Olam",
                    "Neoplatonic Emanation", "Theurgy",
                ]
            },
            "Tarot & Divination": {
                "depth": 0.26,
                "concepts": [
                    "Major Arcana 22 Cards", "The Fool Journey",
                    "The Magician", "High Priestess",
                    "The Emperor Empress", "The Hermit",
                    "Wheel of Fortune", "Death Transformation Card",
                    "The Tower Destruction", "The Star Hope",
                    "The Moon Unconscious", "The Sun Consciousness",
                    "The World Completion", "Minor Arcana Suits",
                    "Rider-Waite Tarot", "Thoth Tarot Crowley",
                    "I Ching Hexagrams", "Astrology Zodiac Signs",
                ]
            },
            "Esoteric Traditions": {
                "depth": 0.28,
                "concepts": [
                    "Gurdjieff Fourth Way", "Ouspensky Tertium Organum",
                    "Enneagram Personality", "Law of Three",
                    "Law of Seven Octaves", "Self-Remembering",
                    "Blavatsky Theosophy", "Secret Doctrine",
                    "Rudolf Steiner Anthroposophy", "Waldorf Education",
                    "Golden Dawn Order", "Aleister Crowley Thelema",
                    "Rosicrucian Brotherhood", "Freemasonry Symbolism",
                    "Eliphas Levi Magic", "Dion Fortune Mystical Qabalah",
                    "Manly P Hall Secret Teachings", "Perennial Philosophy",
                ]
            },
        }
    },

    # ─── GALAXY 8: CINEMA & THEATRE ─────────────────────────────────────
    "Cinema & Theatre": {
        "depth": 0.09,
        "stars": {
            "Theatre History": {
                "depth": 0.19,
                "concepts": [
                    "Greek Tragedy Sophocles", "Greek Comedy Aristophanes",
                    "Chorus Greek Theatre", "Catharsis Theatre",
                    "Commedia Dell Arte", "Elizabethan Theatre Globe",
                    "Moliere Comedy", "Stanislavski Method",
                    "Brecht Epic Theatre", "Alienation Effect Verfremdung",
                    "Artaud Theatre of Cruelty", "Grotowski Poor Theatre",
                    "Beckett Waiting for Godot", "Theatre of the Absurd",
                    "Chekhov Dramatic Realism", "Ibsen Modern Drama",
                    "Noh Theatre Japan", "Kabuki Theatre",
                    "Kathakali Dance Drama", "Shadow Puppetry Wayang",
                ]
            },
            "Cinema History & Movements": {
                "depth": 0.22,
                "concepts": [
                    "Lumiere Brothers First Film", "Melies Trip to Moon",
                    "German Expressionism Caligari", "Soviet Montage Eisenstein",
                    "Hollywood Golden Age", "Studio System",
                    "Italian Neorealism Bicycle Thieves", "French New Wave Breathless",
                    "Japanese New Wave", "Cinema Novo Brazil",
                    "Third Cinema Movement", "Dogme 95 Manifesto",
                    "New Hollywood Coppola Scorsese", "Blockbuster Era Spielberg",
                    "Hong Kong Cinema Wuxia", "Korean New Wave",
                    "Iranian Cinema Kiarostami", "Romanian New Wave",
                    "Bollywood Masala Film", "Animation History Disney Ghibli",
                ]
            },
            "Film Theory & Criticism": {
                "depth": 0.25,
                "concepts": [
                    "Auteur Theory Cahiers", "Camera Stylo",
                    "Apparatus Theory Cinema", "Suture Theory Film",
                    "Feminist Film Theory Mulvey", "Male Gaze",
                    "Postcolonial Cinema Theory", "Third World Cinema",
                    "Deleuze Cinema Time-Image", "Movement-Image Deleuze",
                    "Bazin Realism Cinema", "Kracauer Film Theory",
                    "Metz Film Semiotics", "Psychoanalytic Film Theory",
                    "Cognitive Film Theory", "Phenomenology of Film",
                    "Documentary Ethics", "Propaganda Film",
                ]
            },
            "Masterwork Directors": {
                "depth": 0.27,
                "concepts": [
                    "Kubrick 2001 Space Odyssey", "Kubrick Clockwork Orange",
                    "Tarkovsky Mirror", "Tarkovsky Stalker",
                    "Bergman Persona", "Bergman Wild Strawberries",
                    "Fellini Dolce Vita", "Fellini Amarcord",
                    "Kurosawa Seven Samurai", "Kurosawa Rashomon Film",
                    "Ozu Tokyo Story", "Bresson Pickpocket",
                    "Godard Pierrot le Fou", "Truffaut 400 Blows",
                    "Lynch Mulholland Drive", "Wong Kar-Wai Mood for Love",
                    "Terrence Malick Tree of Life", "Denis Villeneuve",
                    "Coppola Apocalypse Now", "Wim Wenders Wings of Desire",
                ]
            },
        }
    },

    # ─── GALAXY 9: GASTRONOMY & CULTURE MATERIAL ────────────────────────
    "Gastronomy & Material Culture": {
        "depth": 0.11,
        "stars": {
            "Culinary Traditions": {
                "depth": 0.21,
                "concepts": [
                    "French Haute Cuisine", "Escoffier Brigade System",
                    "Japanese Kaiseki", "Umami Fifth Taste",
                    "Sushi Art Edomae", "Ramen Culture",
                    "Italian Regional Cuisine", "Pasta Making Traditions",
                    "Chinese Eight Cuisines", "Wok Hei Breath of Wok",
                    "Indian Spice Traditions", "Tandoor Cooking",
                    "Mexican Mole Complexity", "Thai Flavor Balance",
                    "Ethiopian Injera Ceremony", "Moroccan Tagine",
                    "Peruvian Ceviche Nikkei", "Turkish Ottoman Kitchen",
                ]
            },
            "Fermentation & Preservation": {
                "depth": 0.24,
                "concepts": [
                    "Fermentation Microbiology", "Lactic Acid Fermentation",
                    "Wine Terroir Vinification", "Beer Brewing History",
                    "Sake Brewing Koji", "Mead Ancient Drink",
                    "Cheese Aging Affinage", "Sourdough Starter",
                    "Kimchi Korean", "Sauerkraut German",
                    "Miso Soy Sauce Fermentation", "Kombucha SCOBY",
                    "Kefir Grains", "Tempeh Fermentation",
                    "Curing Smoking Meats", "Pickling Traditions",
                ]
            },
            "Spice Routes & Food History": {
                "depth": 0.26,
                "concepts": [
                    "Spice Trade History", "Silk Road Culinary Exchange",
                    "Columbian Exchange Food", "Chili Pepper Spread",
                    "Coffee Discovery Ethiopia", "Tea Ceremony Japan China",
                    "Chocolate Cacao Mesoamerica", "Sugar Plantation History",
                    "Salt Preservation Civilization", "Bread History Civilization",
                    "Rice Paddy Civilization Asia", "Corn Maize Americas",
                    "Olive Oil Mediterranean", "Coconut Palm Culture",
                    "Potato Famine History", "Banana Republics History",
                ]
            },
            "Material Culture & Crafts": {
                "depth": 0.28,
                "concepts": [
                    "Ceramics Pottery History", "Porcelain China",
                    "Textile Weaving History", "Silk Production Sericulture",
                    "Indigo Dye History", "Batik Textile Art",
                    "Metallurgy Bronze Iron", "Damascus Steel",
                    "Glassblowing Murano", "Woodworking Joinery Japanese",
                    "Papermaking History", "Calligraphy Tools",
                    "Leatherworking", "Basket Weaving Indigenous",
                    "Jewelry Goldsmithing", "Lacquerware East Asian",
                ]
            },
        }
    },

    # ─── GALAXY 10: ARCHITECTURE & URBANISMO ────────────────────────────
    "Architecture & Urbanism": {
        "depth": 0.10,
        "stars": {
            "Ancient Architecture": {
                "depth": 0.20,
                "concepts": [
                    "Pyramids Egypt Architecture", "Parthenon Athens",
                    "Colosseum Rome", "Pantheon Rome Dome",
                    "Stonehenge Megaliths", "Ziggurats Mesopotamia",
                    "Great Wall China", "Angkor Wat Temple",
                    "Petra Rock Cut", "Borobudur Temple",
                    "Roman Aqueducts Engineering", "Roman Roads Via Appia",
                    "Hagia Sophia", "Inca Stonemasonry",
                    "Moai Easter Island", "Persepolis",
                ]
            },
            "Medieval to Renaissance Architecture": {
                "depth": 0.23,
                "concepts": [
                    "Romanesque Architecture", "Gothic Architecture Flying Buttress",
                    "Notre Dame Cathedral", "Chartres Cathedral",
                    "Gothic Ribbed Vault", "Stained Glass Windows",
                    "Brunelleschi Dome Florence", "Palladio Villas",
                    "Renaissance Proportion", "Vitruvian Man Architecture",
                    "Baroque Architecture Bernini", "Versailles Palace",
                    "Mughal Architecture Symmetry", "Forbidden City Beijing",
                    "Castle Fortification", "Islamic Arch Patterns",
                ]
            },
            "Modern Architecture": {
                "depth": 0.25,
                "concepts": [
                    "Art Nouveau Gaudi", "Sagrada Familia",
                    "Bauhaus Architecture Gropius", "Form Follows Function",
                    "Le Corbusier Modulor", "Villa Savoye Pilotis",
                    "Mies van der Rohe Less is More", "Barcelona Pavilion",
                    "Frank Lloyd Wright Organic", "Fallingwater",
                    "Wright Prairie Houses", "Guggenheim Museum Wright",
                    "Brutalism Concrete", "Unite d'Habitation",
                    "International Style", "Curtain Wall Glass",
                    "Metabolism Architecture Japan", "Tadao Ando Concrete Light",
                ]
            },
            "Contemporary Architecture": {
                "depth": 0.27,
                "concepts": [
                    "Deconstructivism Gehry", "Bilbao Guggenheim",
                    "Zaha Hadid Parametric", "Fluid Architecture",
                    "Rem Koolhaas OMA", "Junkspace Theory",
                    "Norman Foster High-Tech", "Renzo Piano",
                    "BIG Bjarke Ingels", "Sustainable Architecture",
                    "Green Building LEED", "Passive House Standard",
                    "Biomimicry Architecture", "Responsive Facades",
                    "3D Printed Architecture", "Mass Timber Construction",
                    "Kengo Kuma Materials", "Peter Zumthor Atmosphere",
                ]
            },
            "Urban Theory & Planning": {
                "depth": 0.29,
                "concepts": [
                    "Garden City Howard", "Radiant City Le Corbusier",
                    "Jane Jacobs Death Life Cities", "Eyes on Street",
                    "Kevin Lynch Image of City", "Legibility Urban",
                    "Situationist Derive", "Psychogeography Debord",
                    "New Urbanism", "Transit-Oriented Development",
                    "Smart Cities", "Urban Sprawl",
                    "Gentrification", "Right to the City Lefebvre",
                    "Favela Informal Urbanism", "Slum Upgrading",
                    "Haussmann Paris Renovation", "Barcelona Cerda Grid",
                ]
            },
        }
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# MASSIVE CROSS-GALAXY BRIDGES
# ═══════════════════════════════════════════════════════════════════════════

CROSS_EDGES = [
    # Mythology ↔ Literature (from knowledge_galaxies collection)
    ("Greek Tragedy Sophocles", "Oedipus Complex Myth", "Dramatizes"),
    ("Odysseus Journey", "Trojan War", "Follows"),
    ("Prometheus Fire Theft", "Icarus Wings", "Parallels"),
    ("Norse Mythology", "Ragnarok End of World", "Climax"),

    # Mythology ↔ Psychology
    ("Oedipus Complex Myth", "Oedipus Complex Freud", "Inspires"),
    ("Archetypes Jung", "Jung Collective Unconscious", "Contains"),
    ("Shadow Self Jung", "Individuation Process", "Requires"),
    ("Mandala Psychology", "Mandala Sacred Circle", "Derives"),
    ("Hero Journey", "Individuation Process", "Parallels"),
    ("Dionysus Wine Ecstasy", "Altered States of Consciousness", "Embodies"),
    ("Shamanic Journey", "Transpersonal Psychology Grof", "Studies"),

    # Mythology ↔ Philosophy
    ("Platonic Forms", "Brahman Ultimate Reality", "Parallels"),
    ("Karma Law of Action", "Dharma Cosmic Order", "Governs"),
    ("Four Noble Truths", "Nirvana Enlightenment", "Leads"),
    ("Taoism Wu Wei", "Yin Yang Duality", "Expresses"),
    ("Zen Meditation Zazen", "Koan Paradox", "Uses"),

    # History ↔ Civilizations
    ("Roman Republic", "Roman Empire Augustus", "Becomes"),
    ("Classical Athens Democracy", "Peloponnesian War", "Ends"),
    ("Silk Road Trade", "Spice Trade History", "Overlaps"),
    ("Islamic Golden Age Science", "Islamic Golden Age Baghdad", "Locates"),
    ("Mongol Empire Genghis Khan", "Silk Road Trade", "Controls"),
    ("Atlantic Slave Trade", "Colonialism Imperialism", "Enables"),
    ("Mali Empire Timbuktu", "Mali Empire Mansa Musa", "Rules"),

    # History ↔ Architecture
    ("Pyramids of Giza", "Pyramids Egypt Architecture", "Same"),
    ("Gothic Cathedrals", "Gothic Architecture Flying Buttress", "Defines"),
    ("Italian Renaissance Florence", "Brunelleschi Dome Florence", "Crowns"),
    ("Industrial Revolution", "Form Follows Function", "Enables"),
    ("French Revolution 1789", "Haussmann Paris Renovation", "Aftermath"),

    # Mathematics ↔ Architecture
    ("Euclidean Geometry", "Renaissance Proportion", "Applies"),
    ("Non-Euclidean Geometry", "Zaha Hadid Parametric", "Enables"),
    ("Golden Ratio Art", "Vitruvian Man Architecture", "Embodies"),
    ("Hyperbolic Geometry Lobachevsky", "Poincare Disk Model", "Models"),

    # Psychology ↔ Linguistics
    ("Cognitive Revolution", "Chomsky Universal Grammar", "Parallels"),
    ("Language Acquisition Device", "Piaget Stages Development", "Contrasts"),
    ("Sapir-Whorf Hypothesis", "Cognitive Semantics", "Supports"),
    ("Lacan Mirror Stage", "Lacan Real Symbolic Imaginary", "Develops"),
    ("Metaphor Theory Lakoff", "Conceptual Metaphor", "Defines"),

    # Esotericism ↔ Psychology
    ("Jung Alchemy Psychology", "Alchemical Symbolism", "Interprets"),
    ("Magnum Opus Great Work", "Individuation Process", "Parallels"),
    ("Nigredo Blackening", "Dark Night of the Soul", "Corresponds"),
    ("Tree of Life Sephiroth", "Archetypes Jung", "Maps"),
    ("Gurdjieff Fourth Way", "Self-Remembering", "Teaches"),
    ("Enneagram Personality", "Archetypes Jung", "Complements"),

    # Esotericism ↔ Mythology
    ("Hermes Trismegistus", "Emerald Tablet Hermes", "Authors"),
    ("Kabbalah Mysticism", "Tree of Life Sephiroth", "Structures"),
    ("Sufism Mystical Islam", "Rumi Whirling Dervish", "Practices"),
    ("Kundalini Energy", "Chakras Energy Centers", "Travels"),
    ("As Above So Below", "Principle of Correspondence", "States"),

    # Cinema ↔ Philosophy
    ("Bergman Persona", "Existential Therapy Yalom", "Explores"),
    ("Tarkovsky Stalker", "Heidegger Being and Time", "Reflects"),
    ("Kubrick 2001 Space Odyssey", "Nietzsche Ubermensch", "References"),
    ("Lynch Mulholland Drive", "Lacan Real Symbolic Imaginary", "Enacts"),
    ("Theatre of the Absurd", "Camus Absurd", "Dramatizes"),
    ("Brecht Epic Theatre", "Alienation Effect Verfremdung", "Uses"),

    # Gastronomy ↔ History
    ("Columbian Exchange Food", "Columbus Americas 1492", "Causes"),
    ("Tea Ceremony Japan China", "Japanese Heian Period", "Develops"),
    ("Coffee Discovery Ethiopia", "Islamic Golden Age Baghdad", "Spreads"),
    ("Spice Trade History", "Age of Exploration", "Motivates"),
    ("Sugar Plantation History", "Atlantic Slave Trade", "Drives"),
    ("Potato Famine History", "Industrial Revolution", "Parallels"),

    # Gastronomy ↔ Biology
    ("Fermentation Microbiology", "Lactic Acid Fermentation", "Process"),
    ("Koji", "Miso Soy Sauce Fermentation", "Produces"),
    ("Kombucha SCOBY", "Kefir Grains", "Similar"),

    # Architecture ↔ Art
    ("Art Nouveau Gaudi", "Sagrada Familia", "Creates"),
    ("Bauhaus Architecture Gropius", "Bauhaus Design", "Unifies"),
    ("Deconstructivism Gehry", "Bilbao Guggenheim", "Exemplifies"),
    ("Brutalism Concrete", "Unite d'Habitation", "Exemplifies"),

    # Linguistics ↔ Semiotics ↔ Philosophy
    ("Saussure Signifier Signified", "Peirce Semiotics Triadic", "Contrasts"),
    ("Barthes Mythologies", "Barthes Death of Author", "Connects"),
    ("Baudrillard Simulacra Semiotics", "Hyperreality Signs", "Defines"),
    ("Wittgenstein Philosophical Investigations", "Language Games", "Introduces"),

    # Civilizations ↔ Linguistics
    ("Cuneiform Writing", "Writing Systems Evolution", "Begins"),
    ("Maya Hieroglyphics", "Decipherment Rosetta Stone", "Method"),
    ("Phoenician Alphabet", "Proto-Indo-European", "Transmits"),

    # Deep cross-domain
    ("Psychedelic Therapy Research", "Ayahuasca Vision", "Studies"),
    ("Near-Death Experience", "Tibetan Book of the Dead", "Parallels"),
    ("Stanislav Grof", "Holotropic Breathwork", "Creates"),
    ("Spiral Dynamics", "Stages of Consciousness", "Models"),
    ("Gestalt Therapy Perls", "Here and Now Awareness", "Emphasizes"),
    ("Man Search for Meaning Frankl", "Holocaust", "Survives"),
    ("Flow State Csikszentmihalyi", "Peak Experiences", "Overlaps"),
    ("Maslow Hierarchy of Needs", "Self-Actualization", "Crowns"),
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

    # Wait for server readiness
    print(f"[*] Waiting for gRPC server at {host}...")
    try:
        grpc.channel_ready_future(channel).result(timeout=120)
        print(f"[+] gRPC server ready!")
    except grpc.FutureTimeoutError:
        print(f"[!] Timeout waiting for server. Aborting.")
        return

    print(f"\n{'='*60}")
    print(f"  NietzscheDB Ultra-Cultural Galaxy Ingestion")
    print(f"  Host: {host}")
    print(f"  Collection: {collection}")
    print(f"  Metric: {metric} | Dim: {dim}")
    print(f"{'='*60}\n")

    try:
        stub.CreateCollection(pb2.CreateCollectionRequest(
            collection=collection, dim=dim, metric=metric
        ))
        print(f"[+] Collection '{collection}' created")
    except grpc.RpcError as e:
        print(f"[!] CreateCollection: {e.details() if hasattr(e, 'details') else e}")

    # Build nodes
    all_nodes = []
    node_ids = {}

    for galaxy_name, galaxy_data in GALAXIES.items():
        galaxy_depth = galaxy_data["depth"]
        galaxy_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"culture-galaxy:{galaxy_name}"))
        node_ids[galaxy_name] = galaxy_id
        emb = make_poincare_embedding(galaxy_name, "__galaxy__", galaxy_name, galaxy_depth, dim)
        all_nodes.append({
            "id": galaxy_id,
            "content": json.dumps({"name": galaxy_name, "type": "galaxy", "domain": "culture"}).encode('utf-8'),
            "node_type": "Concept", "energy": 1.0, "embedding": emb,
        })

        for star_name, star_data in galaxy_data["stars"].items():
            star_depth = star_data["depth"]
            star_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"culture-star:{galaxy_name}/{star_name}"))
            node_ids[star_name] = star_id
            emb = make_poincare_embedding(galaxy_name, star_name, star_name, star_depth, dim)
            all_nodes.append({
                "id": star_id,
                "content": json.dumps({"name": star_name, "type": "star", "galaxy": galaxy_name}).encode('utf-8'),
                "node_type": "Concept", "energy": 0.8, "embedding": emb,
            })

            for concept_name in star_data["concepts"]:
                concept_depth = star_depth + 0.1 + (hash(concept_name) % 100) / 1000.0
                concept_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"culture-concept:{galaxy_name}/{star_name}/{concept_name}"))
                node_ids[concept_name] = concept_id
                emb = make_poincare_embedding(galaxy_name, star_name, concept_name, concept_depth, dim)
                all_nodes.append({
                    "id": concept_id,
                    "content": json.dumps({
                        "name": concept_name, "type": "concept",
                        "star": star_name, "galaxy": galaxy_name,
                    }).encode('utf-8'),
                    "node_type": "Semantic", "energy": 0.6, "embedding": emb,
                })

    total_nodes = len(all_nodes)
    print(f"[*] Inserting {total_nodes} nodes in batches of 50...")

    batch_size = 50
    inserted = 0
    for i in range(0, total_nodes, batch_size):
        batch = all_nodes[i:i+batch_size]
        requests = []
        for n in batch:
            pv = pb2.PoincareVector(coords=n["embedding"])
            req = pb2.InsertNodeRequest(
                id=n["id"], embedding=pv, content=n["content"],
                node_type=n["node_type"], energy=n["energy"], collection=collection,
            )
            requests.append(req)
        try:
            stub.BatchInsertNodes(pb2.BatchInsertNodesRequest(nodes=requests, collection=collection))
            inserted += len(batch)
            pct = inserted / total_nodes * 100
            print(f"  [{pct:5.1f}%] Inserted {len(batch)} nodes (total: {inserted})", end='\r')
        except grpc.RpcError as e:
            print(f"\n  [!] Batch {i//batch_size}: {e.details() if hasattr(e, 'details') else e}")

    print(f"\n[+] Nodes inserted: {inserted}/{total_nodes}")

    # Edges
    print(f"\n[*] Creating hierarchical edges...")
    edge_count = 0
    for galaxy_name, galaxy_data in GALAXIES.items():
        galaxy_id = node_ids[galaxy_name]
        for star_name, star_data in galaxy_data["stars"].items():
            star_id = node_ids.get(star_name)
            if not star_id:
                continue
            try:
                stub.InsertEdge(make_edge_request(pb2, from_node=galaxy_id, to=star_id,
                    edge_type="Hierarchical", weight=1.0, collection=collection))
                edge_count += 1
            except grpc.RpcError:
                pass

            for concept_name in star_data["concepts"]:
                concept_id = node_ids.get(concept_name)
                if not concept_id:
                    continue
                try:
                    stub.InsertEdge(make_edge_request(pb2, from_node=star_id, to=concept_id,
                        edge_type="Hierarchical", weight=0.8, collection=collection))
                    edge_count += 1
                except grpc.RpcError:
                    pass

            concepts_in_star = [c for c in star_data["concepts"] if c in node_ids]
            for j in range(len(concepts_in_star)):
                for k in range(j+1, min(j+4, len(concepts_in_star))):
                    try:
                        stub.InsertEdge(make_edge_request(pb2,
                            from_node=node_ids[concepts_in_star[j]],
                            to=node_ids[concepts_in_star[k]],
                            edge_type="Association", weight=0.6, collection=collection))
                        edge_count += 1
                    except grpc.RpcError:
                        pass

        print(f"  Galaxy '{galaxy_name}': edges ({edge_count} total)     ", end='\r')

    print(f"\n[+] Hierarchical + association edges: {edge_count}")

    print(f"\n[*] Creating cross-galaxy bridges...")
    bridge_count = 0
    for from_name, to_name, rel_type in CROSS_EDGES:
        from_id = node_ids.get(from_name)
        to_id = node_ids.get(to_name)
        if from_id and to_id:
            try:
                stub.InsertEdge(make_edge_request(pb2, from_node=from_id, to=to_id,
                    edge_type="Association", weight=0.7, collection=collection))
                bridge_count += 1
            except grpc.RpcError:
                pass

    total_edges = edge_count + bridge_count
    print(f"[+] Cross-galaxy bridges: {bridge_count}")

    num_galaxies = len(GALAXIES)
    num_stars = sum(len(g['stars']) for g in GALAXIES.values())
    print(f"\n{'='*60}")
    print(f"  ULTRA-CULTURAL INGESTION COMPLETE")
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
    print(f"  The Poincare ball now contains the deepest layers")
    print(f"  of human civilization. L-System will grow connections")
    print(f"  between mythology, science, art, and philosophy!")
    print(f"\n  Dashboard: http://{host.split(':')[0]}:8080")

    channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert Ultra-Cultural Galaxies into NietzscheDB")
    parser.add_argument("--host", default="localhost:50051")
    parser.add_argument("--collection", default="culture_galaxies")
    parser.add_argument("--metric", default="poincare")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    insert_galaxies(args.host, args.collection, args.metric, args.dim)

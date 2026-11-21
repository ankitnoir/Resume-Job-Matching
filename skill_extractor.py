"""
skill_extractor.py
==================
Rule-based skill extraction from resumes and job descriptions using a skills dictionary.

How it works
------------
* A skills dictionary maps each CANONICAL skill name (e.g. "Machine Learning") to a category
  and a list of aliases (e.g. "ml"). Aliases are normalised so "scikit-learn", "Scikit Learn"
  and "sklearn" all resolve to the same canonical skill.
* Matching is done with one compiled regular expression using custom word boundaries, so
  "java" does not match inside "javascript" and "sql" does not match inside "mysql".
* Very short, ambiguous skills ("R", "C") are matched case-sensitively.
* This is deliberately RULE-BASED (not machine learning): it only finds skills that exist in
  the dictionary. Extend the dictionary with your own CSV (see `skills_csv`).

Limitations (be honest about these in your report)
--------------------------------------------------
* Words such as "react", "spring" or "swift" can occasionally be false positives in non-technical
  context; "C" can match "Vitamin C".
* Skills missing from the dictionary are not detected.

Public API
----------
SkillExtractor(skills_csv=None, include_soft_skills=False, merge_with_default=True)
    .extract(text)               -> list[str]            canonical skills, most frequent first
    .extract_with_counts(text)   -> dict[str, int]       skill -> number of mentions
    .extract_by_category(text)   -> dict[str, list[str]]
    .get_category(skill)         -> str
get_default_extractor()          -> shared SkillExtractor instance
extract_skills(text, extractor=None)             -> list[str]
extract_skill_counts(text, extractor=None)       -> dict[str, int]
extract_skills_by_category(text, extractor=None) -> dict[str, list[str]]
get_skill_category(skill, extractor=None)        -> str

Custom dictionary CSV format (UTF-8, header required)
-----------------------------------------------------
skill,category,aliases
Machine Learning,Data Science & ML,ml|machine-learning
Kubernetes,Cloud & DevOps,k8s
(aliases are optional and separated by "|")
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd

from text_preprocessing import clean_text

logger = logging.getLogger(__name__)

SOFT_SKILLS_CATEGORY = "Soft Skills"

# --------------------------------------------------------------------------- #
# Default skills dictionary.
# Format of each entry:  "Canonical Name|alias 1|alias 2"
# (the canonical name itself is always an alias, matching is case-insensitive)
# --------------------------------------------------------------------------- #
DEFAULT_SKILLS: Dict[str, List[str]] = {
    "Programming Languages": [
        "Python", "Java", "JavaScript|ecmascript", "TypeScript", "C++|cpp|c plus plus",
        "C#|csharp|c sharp", "Golang", "Rust", "PHP", "Ruby", "Swift", "Kotlin", "Scala",
        "MATLAB", "Perl", "Dart", "Bash|shell scripting|shell script", "HTML|html5",
        "CSS|css3", "VBA", "Objective-C",
    ],
    "Web Development": [
        "React|reactjs|react.js|react js", "Angular|angularjs|angular.js",
        "Vue.js|vue|vuejs|vue js", "Node.js|nodejs|node js", "Express.js|expressjs|express js",
        "Next.js|nextjs", "Django", "Flask", "FastAPI", "Spring Boot|springboot",
        "Spring Framework", "ASP.NET|asp.net core|aspnet", ".NET|dotnet|.net core|.net framework",
        "jQuery", "Bootstrap", "Tailwind CSS|tailwind",
        "REST API|rest apis|restful api|restful apis|restful services",
        "GraphQL", "WordPress", "Laravel", "Responsive Design|responsive web design",
    ],
    "Data Science & ML": [
        "Machine Learning|ml", "Deep Learning", "Natural Language Processing|nlp",
        "Computer Vision", "Data Analysis|data analytics", "Data Visualization|data visualisation",
        "Statistics|statistical analysis|statistical modeling|statistical modelling",
        "Data Mining", "Feature Engineering",
        "Predictive Modeling|predictive modelling|predictive analytics",
        "Time Series|time series analysis|time series forecasting",
        "A/B Testing|ab testing|a b testing", "Reinforcement Learning",
        "Generative AI|genai|gen ai", "Large Language Models|llm|llms",
        "Neural Networks|neural network", "Scikit-learn|sklearn|scikit learn",
        "TensorFlow|tensor flow", "Keras", "PyTorch", "Pandas", "NumPy", "SciPy",
        "Matplotlib", "Seaborn", "Plotly", "OpenCV", "NLTK", "spaCy",
        "Hugging Face|huggingface", "XGBoost", "LightGBM", "Hadoop",
        "Apache Spark|pyspark|spark", "Apache Kafka|kafka", "Apache Airflow|airflow",
        "Apache Hive|hive", "ETL|etl pipelines|etl pipeline|extract transform load",
        "Data Warehousing|data warehouse", "Big Data",
        "Jupyter|jupyter notebook|jupyter notebooks", "Power BI|powerbi", "Tableau",
        "Excel|microsoft excel|ms excel|advanced excel",
    ],
    "Databases": [
        "SQL|structured query language", "MySQL", "PostgreSQL|postgres", "SQLite",
        "Oracle Database|oracle db|oracle",
        "SQL Server|mssql|ms sql|ms sql server|microsoft sql server",
        "PL/SQL|plsql|pl sql", "T-SQL|tsql", "MongoDB|mongo db|mongo", "Redis", "Cassandra",
        "Elasticsearch|elastic search", "DynamoDB", "NoSQL|no sql", "Firebase", "Snowflake",
        "Database Design|database management|dbms|rdbms",
    ],
    "Cloud & DevOps": [
        "AWS|amazon web services", "Azure|microsoft azure",
        "Google Cloud Platform|gcp|google cloud", "Docker", "Kubernetes|k8s", "Terraform",
        "Ansible", "Jenkins",
        "CI/CD|ci cd|cicd|continuous integration|continuous deployment|continuous delivery",
        "Git", "GitHub", "GitLab", "Bitbucket", "Linux|unix", "DevOps",
        "Microservices|microservice", "Serverless|aws lambda", "Prometheus", "Grafana", "Nginx",
        "Cloud Computing|cloud infrastructure",
    ],
    "Security & Networking": [
        "Cybersecurity|cyber security|information security|infosec", "Network Security",
        "Penetration Testing|pen testing|pentesting", "Firewalls|firewall", "Ethical Hacking",
        "SIEM", "Vulnerability Assessment", "Computer Networking|networking|tcp/ip",
    ],
    "Software Engineering & Testing": [
        "Object-Oriented Programming|oop|oops|object oriented programming",
        "Data Structures|data structures and algorithms|dsa", "Algorithms", "System Design",
        "Design Patterns", "Unit Testing|unit tests",
        "Test Automation|automation testing|automated testing", "Manual Testing", "Selenium",
        "JUnit", "Pytest", "Postman", "JIRA", "Agile|agile methodology", "Scrum", "Kanban",
        "Android Development|android", "iOS Development|ios", "Flutter", "React Native", "SAP",
        "Blockchain", "Solidity", "Smart Contracts|smart contract", "Ethereum", "Version Control",
    ],
    "Business & Management": [
        "Project Management|project planning", "Product Management",
        "Business Analysis|requirements gathering|requirement analysis",
        "Stakeholder Management", "Strategic Planning", "Budgeting|budget management",
        "Operations Management", "Supply Chain|supply chain management",
        "Six Sigma|lean six sigma", "Risk Management", "Change Management", "PMO",
        "MS Project|microsoft project", "Vendor Management",
    ],
    "Finance & Accounting": [
        "Financial Analysis", "Financial Reporting", "Financial Modeling|financial modelling",
        "Accounting", "Bookkeeping", "Accounts Payable", "Accounts Receivable",
        "Taxation|tax preparation|gst", "Auditing|internal audit", "Tally", "QuickBooks",
        "Payroll", "GAAP", "IFRS", "Reconciliation|bank reconciliation",
        "Forecasting|financial forecasting", "Investment Banking", "Credit Analysis", "KYC",
    ],
    "Marketing & Sales": [
        "Digital Marketing", "SEO|search engine optimization|search engine optimisation",
        "SEM|google ads|google adwords", "Social Media Marketing|social media management",
        "Content Writing|content creation|copywriting", "Email Marketing", "Google Analytics",
        "Market Research", "Lead Generation", "CRM|customer relationship management",
        "Salesforce", "HubSpot", "Negotiation",
        "Customer Service|customer support|customer success", "Business Development",
        "Public Relations", "Brand Management|branding", "Sales",
    ],
    "Human Resources": [
        "Recruitment|talent acquisition|recruiting", "Employee Relations", "Onboarding",
        "Performance Management", "HRIS|human resource information system",
        "Training and Development|learning and development",
        "Compensation and Benefits|compensation & benefits", "Employee Engagement",
        "Labor Law|labour law|employment law", "Talent Management",
    ],
    "Healthcare & Fitness": [
        "Patient Care", "Clinical Research",
        "Electronic Health Records|ehr|electronic medical records|emr", "Nursing",
        "First Aid|cpr", "Personal Training|fitness training", "Nutrition|diet planning",
        "HIPAA", "Pharmacology",
    ],
    "Design & Creative": [
        "Adobe Photoshop|photoshop", "Adobe Illustrator|illustrator", "Adobe InDesign|indesign",
        "Adobe XD", "Figma", "Canva", "UI/UX Design|ui ux|ux design|ui design|user experience",
        "Wireframing|wireframes", "Prototyping", "Video Editing|premiere pro|final cut pro",
        "Graphic Design", "Typography", "3D Modeling|3d modelling|blender",
        "Animation|after effects",
    ],
    "Engineering & Manufacturing": [
        "AutoCAD", "SolidWorks", "CAD|computer aided design", "PLC|plc programming", "SCADA",
        "Embedded Systems|embedded c", "Arduino", "Raspberry Pi", "Circuit Design",
        "Quality Control|quality assurance|qa", "Lean Manufacturing", "STAAD Pro|staad", "Revit",
        "Primavera", "HVAC", "ANSYS", "CATIA", "Mechanical Design",
    ],
    "Other Professional": [
        "Lesson Planning|curriculum development|classroom management",
        "Legal Research", "Litigation", "Contract Drafting|drafting contracts",
        "Menu Planning", "Food Safety|haccp", "Inventory Management|inventory control",
    ],
    SOFT_SKILLS_CATEGORY: [
        "Communication|communication skills|verbal communication|written communication",
        "Leadership|team leadership", "Teamwork|team player|collaboration",
        "Problem Solving|problem-solving", "Critical Thinking", "Time Management",
        "Adaptability|flexibility", "Presentation Skills|public speaking",
        "Analytical Skills|analytical thinking", "Creativity|creative thinking",
        "Attention to Detail|detail oriented|detail-oriented", "Decision Making",
        "Mentoring|coaching", "Conflict Resolution",
    ],
}

# Short ambiguous skills matched CASE-SENSITIVELY (exact spelling required).
CASE_SENSITIVE_SKILLS: Dict[str, str] = {
    "R": "Programming Languages",
    "C": "Programming Languages",
}


def _normalise_alias(text: str) -> str:
    """Lower-case and collapse whitespace/hyphens, so 'Scikit-Learn' == 'scikit learn'."""
    return re.sub(r"[\s\-]+", " ", text.strip().lower())


class SkillExtractor:
    """Dictionary-based skill extractor. Build once, reuse for many documents."""

    def __init__(
        self,
        skills_csv: Optional[Union[str, Path]] = None,
        include_soft_skills: bool = False,
        merge_with_default: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        skills_csv : optional path to a custom dictionary CSV (columns: skill, category, aliases).
        include_soft_skills : soft skills ("Communication", ...) are noisy for gap analysis, so
            they are excluded by default. Set True to include them.
        merge_with_default : if a CSV is given, True adds it to the built-in dictionary,
            False uses only the CSV.
        """
        self.include_soft_skills = include_soft_skills
        self._skill_category: Dict[str, str] = {}
        self._alias_map: Dict[str, str] = {}      # normalised alias -> canonical skill
        self._cs_alias_map: Dict[str, str] = {}   # case-sensitive alias -> canonical skill

        if skills_csv is None or merge_with_default:
            self._load_default()
        if skills_csv is not None:
            self._load_csv(skills_csv)
        self._compile_patterns()

    # ----------------------------- dictionary building ----------------------------- #
    def _register(self, canonical: str, category: str, aliases: Iterable[str],
                  case_sensitive: bool = False) -> None:
        canonical = canonical.strip()
        if not canonical:
            return
        if category == SOFT_SKILLS_CATEGORY and not self.include_soft_skills:
            return
        self._skill_category[canonical] = category
        for alias in [canonical, *aliases]:
            if case_sensitive:
                self._cs_alias_map[alias.strip()] = canonical
            else:
                norm = _normalise_alias(alias)
                if norm:
                    self._alias_map[norm] = canonical

    def _load_default(self) -> None:
        for category, entries in DEFAULT_SKILLS.items():
            for entry in entries:
                parts = [p.strip() for p in entry.split("|") if p.strip()]
                self._register(parts[0], category, parts[1:])
        for skill, category in CASE_SENSITIVE_SKILLS.items():
            self._register(skill, category, [], case_sensitive=True)

    def _load_csv(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Skills dictionary not found: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Could not read skills dictionary '{path}': {exc}") from exc
        if "skill" not in df.columns:
            raise ValueError("Skills CSV must contain a 'skill' column "
                             "(optional columns: category, aliases).")
        for _, row in df.iterrows():
            if pd.isna(row["skill"]):
                continue
            category = row["category"] if "category" in df.columns and pd.notna(row["category"]) else "Other"
            aliases = []
            if "aliases" in df.columns and pd.notna(row["aliases"]):
                aliases = [a for a in str(row["aliases"]).split("|") if a.strip()]
            self._register(str(row["skill"]), str(category), aliases)

    def _compile_patterns(self) -> None:
        """Build one regex for case-insensitive aliases and one for case-sensitive ones."""
        # Longest aliases first so "machine learning" wins over any shorter overlapping alias.
        aliases = sorted(self._alias_map, key=len, reverse=True)
        parts = [
            # Words inside an alias may be separated by spaces or hyphens in the text.
            r"[\s\-]+".join(re.escape(token) for token in alias.split(" "))
            for alias in aliases
        ]
        # Custom boundaries (instead of \b) so skills like "c++", "c#" and ".net" work.
        self._pattern = re.compile(
            r"(?<![A-Za-z0-9+#])(?:" + "|".join(parts) + r")(?![A-Za-z0-9+#])",
            re.IGNORECASE,
        ) if parts else None

        cs_parts = [re.escape(a) for a in sorted(self._cs_alias_map, key=len, reverse=True)]
        self._cs_pattern = re.compile(
            r"(?<![\w+#&./\-])(?:" + "|".join(cs_parts) + r")(?![\w+#&\-])"
        ) if cs_parts else None

    # --------------------------------- extraction --------------------------------- #
    def extract_with_counts(self, text: Optional[str]) -> Dict[str, int]:
        """Return {canonical skill: number of mentions}, sorted by frequency then name."""
        cleaned = clean_text(text)
        if not cleaned:
            return {}

        counts: Counter = Counter()
        if self._pattern is not None:
            for match in self._pattern.finditer(cleaned):
                canonical = self._alias_map.get(_normalise_alias(match.group(0)))
                if canonical:
                    counts[canonical] += 1
        if self._cs_pattern is not None:
            for match in self._cs_pattern.finditer(cleaned):
                canonical = self._cs_alias_map.get(match.group(0))
                if canonical:
                    counts[canonical] += 1

        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower())))

    def extract(self, text: Optional[str]) -> List[str]:
        """Return the list of canonical skills found in `text` (most frequent first)."""
        return list(self.extract_with_counts(text))

    def extract_by_category(self, text: Optional[str]) -> Dict[str, List[str]]:
        """Return {category: [skills]} for the skills found in `text`."""
        grouped: Dict[str, List[str]] = defaultdict(list)
        for skill in self.extract(text):
            grouped[self.get_category(skill)].append(skill)
        return dict(grouped)

    def get_category(self, skill: str) -> str:
        return self._skill_category.get(skill, "Other")

    @property
    def all_skills(self) -> List[str]:
        return sorted(self._skill_category)


# --------------------------------------------------------------------------- #
# Module-level convenience functions (what the Streamlit app will normally call)
# --------------------------------------------------------------------------- #
_DEFAULT_EXTRACTOR: Optional[SkillExtractor] = None


def get_default_extractor() -> SkillExtractor:
    """Shared extractor built from the default dictionary (created on first use)."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = SkillExtractor()
    return _DEFAULT_EXTRACTOR


def extract_skills(text: Optional[str], extractor: Optional[SkillExtractor] = None) -> List[str]:
    return (extractor or get_default_extractor()).extract(text)


def extract_skill_counts(text: Optional[str], extractor: Optional[SkillExtractor] = None) -> Dict[str, int]:
    return (extractor or get_default_extractor()).extract_with_counts(text)


def extract_skills_by_category(
    text: Optional[str], extractor: Optional[SkillExtractor] = None
) -> Dict[str, List[str]]:
    return (extractor or get_default_extractor()).extract_by_category(text)


def get_skill_category(skill: str, extractor: Optional[SkillExtractor] = None) -> str:
    return (extractor or get_default_extractor()).get_category(skill)

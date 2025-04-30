Here’s a minimal setup.py you can drop into your project root to turn cost-model into an installable package. You’ll still need to move your code into a proper package layout (e.g. src/cost_model/...), but this file shows the key fields:

# setup.py
import setuptools

setuptools.setup(
    name="cost_model",                          # your package name
    version="0.1.0",                            # start with 0.1.0 (change as you bump releases)
    author="Nicholas Amaral",                   # replace with your name
    description="Census projection & plan‐rules tools",
    long_description=open("README.md").read(),  # if you have a README
    long_description_content_type="text/markdown",

    # point setuptools to your source directory
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),

    # runtime dependencies
    install_requires=[
        "pandas>=1.4",
        "pyyaml",
        "numpy",
        # add any others you use, e.g. "pyarrow", "scikit-learn", ...
    ],

    entry_points={
        "console_scripts": [
            # this creates a `run-hr-snapshots` CLI that calls cost_model.scripts.run_hr_snapshots:main
            "run-hr-snapshots=cost_model.scripts.run_hr_snapshots:main",
            "run-plan-rules=cost_model.scripts.run_plan_rules:main",
        ],
    },

    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)



⸻

Next steps
	1.	Restructure your code

cost-model/
├── setup.py
├── README.md
├── src/
│   └── cost_model/
│       ├── __init__.py
│       ├── utils/           ← your utils/ here
│       └── scripts/
│           ├── __init__.py
│           ├── run_hr_snapshots.py
│           └── run_plan_rules.py
└── configs/…


	2.	Install in editable mode
From the project root:

pip install -e .

This lets you run:

run-hr-snapshots --config configs/config.yaml \
                 --census data/census_data.csv \
                 --output output/hr_snapshots \
                 --seed 42

run-plan-rules   --config configs/config.yaml \
                 --snapshots-dir output/hr_snapshots \
                 --output-dir output/plan_outputs


	3.	Version bumping & publishing
Whenever you make breaking changes, increment the version field. Down the road you can even publish to PyPI or host your own internal index.

This setup gives you:
	•	Zero sys.path fiddling
	•	Standard import cost_model.utils… everywhere
	•	Convenient CLI entry points
	•	Well-behaved packaging for CI, testing, and future distribution.
# IDAES publications
Materials related to or referenced in IDAES publications

## Contributing to this repository
If you have a paper or presentation using IDAES-PSE (or funded by IDAES),
it would be nice to contribute the code to this repository so that others
may reproduce your results and use them in their own studies.
This does not necessitate testing, readability, code review, future
maintenance, or any other standards for contributing code to the IDAES-PSE
or Examples-PSE repositories.
The requirements for a contribution to this repository are:
1. Code is contained in a subdirectory named to uniquely
correspond to the paper whose results it produces. The format
`name_venueYEAR/` is suggested.
2. The version of all (Python package) dependencies is included.
This may be in a `requirements.txt` file or a README file.
Non-Python dependencies (for instance, your paper compares standard
power network optimization solves between two solvers, one in Julia,
one in Python) will be handled on a case-by-case basis.
3. The code runs. We will generally take your word on this, although
we reserve the right to postpone merging pull requests if we find
the code to not work on anybody else's local machine.
Optionally, you may structure your code its own Python package with
a small `setup.py` to make it easier for others to import and experiment
with your code.

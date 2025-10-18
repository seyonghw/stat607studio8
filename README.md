STATS607 Studio7

Names : Seyong Hwang, Ethan Schubert, Dongsun Yoon

Running `main.ipnyb` outputs the plot.
The plot is saved as `output.png`, and the whole pipeline runs by `run_all()` in `analyze.py`.


### Installation
1. Clone the repository:
```bash
git clone https://github.com/seyonghw/stat607studio8
cd stat607studio8
```
2. Create environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage
```bash
python run_simulation.py --out results.pkl    
```
This code allows to run full simulation study and generates the full results file results.pkl

```bash
python analyze_results.py --in results.pkl --out figure.pdf        
```
By running the code above you will be able to recreate figures 

### Testing

This project includes basic tests to ensure pipeline correctness.

To run all tests:
```bash
pytest tests/
```
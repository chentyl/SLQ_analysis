import subprocess

files = [
    'fig1',
    'fig2',
    'fig3',
    'fig4',
    'fig5',
    'fig6',
    'fig7',
]

for file in files:
    subprocess.run(f'jupyter nbconvert --execute --to notebook --inplace --allow-errors --ExecutePreprocessor.timeout=-1 {file}.ipynb',shell=True)

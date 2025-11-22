import argparse
from jinja2 import Template
import json
import os

def zero_pad(s, n):
    s_str = str(s)
    pad = n - len(s_str)
    zero_padding = '0' * pad
    return zero_padding + s_str

def main():
    parser = argparse.ArgumentParser(description="Generate job scripts from a template.")
    parser.add_argument("--template", required=True, help="Path to the template .sh file")
    parser.add_argument("--params", required=True, help="Path to the JSON parameters file")
    parser.add_argument("--outdir", default=".", help="Directory to write output scripts into")

    args = parser.parse_args()

    # Load template
    with open(args.template) as f:
        template = Template(f.read())

    # Load parameter sets
    with open(args.params) as f:
        param_list = json.load(f)

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    # Generate output scripts
    for i, params in enumerate(param_list):
        output_filename = os.path.join(args.outdir, os.path.basename(args.template)[:-6] + '_' + zero_pad(i, 3) + '.slurm')
        with open(output_filename, "w") as out:
            out.write(template.render(**params))
        print(f'sbatch {output_filename}')
    
if __name__ == "__main__":
    main()
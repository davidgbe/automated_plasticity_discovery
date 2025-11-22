import argparse
from jinja2 import Template
import json
import os

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
    for params in param_list:
        output_filename = os.path.join(args.outdir, f"job_{params['job_name']}.sh")
        with open(output_filename, "w") as out:
            out.write(template.render(**params))
        print("Wrote", output_filename)


if __name__ == "__main__":
    main()
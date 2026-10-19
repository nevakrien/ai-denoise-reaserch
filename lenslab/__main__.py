import argparse
import json

from lenslab import ROOT
from lenslab.experiments import cpu_threads, create_plan, prompts, run_worker
from lenslab.inventory import CATALOG, atomic_json, discover, load_catalog


def main():
    parser = argparse.ArgumentParser(description="GGUF logit-lens experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("discover")
    scan.add_argument("roots", nargs="*")
    worker = commands.add_parser("worker")
    worker.add_argument("directory")
    run = commands.add_parser("run")
    run.add_argument("--models", nargs="+", required=True, help="Catalog IDs")
    run.add_argument("--prompts", nargs="+", help="Prompt IDs; default: all")
    run.add_argument("--tokens", type=int, default=64)
    run.add_argument("--threads", type=int, default=cpu_threads())
    run.add_argument("--gpu-budget-gib", type=float, default=0)
    run.add_argument("--binary", default=str(ROOT / "build/gguf-lens"))
    run.add_argument("--save-hidden", action="store_true")
    run.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.command == "discover":
        catalog = discover(args.roots)
        atomic_json(CATALOG, catalog)
        print(json.dumps(catalog, indent=2))
        return 0
    if args.command == "worker":
        return run_worker(args.directory)
    catalog = load_catalog()
    by_id = {m["id"]: m for m in catalog["models"]}
    models = [by_id[k] for k in dict.fromkeys(args.models) if k in by_id]
    if len(models) != len(set(args.models)):
        parser.error("Unknown model ID; run discover first")
    suite = [p for p in prompts() if not args.prompts or p["id"] in args.prompts]
    if args.prompts and len(suite) != len(set(args.prompts)):
        parser.error("Unknown prompt ID")
    directory = create_plan(models, suite, binary=args.binary, tokens=args.tokens, threads=args.threads,
                            gpu_budget_gib=args.gpu_budget_gib, save_hidden=args.save_hidden, timeout=args.timeout)
    print(directory, flush=True)
    return run_worker(directory)


if __name__ == "__main__":
    raise SystemExit(main())

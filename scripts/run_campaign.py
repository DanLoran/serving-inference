"""Run an ordered, config-driven collection of inference sweeps."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generate_prompts
import run_experiment
from experiment_manifest import collect_git, sanitize, utc_now, write_json


SCHEMA_VERSION = "1.0"
CAMPAIGN_FIELDS = {
    "$schema",
    "schema_version",
    "name",
    "output_root",
    "defaults",
    "workloads",
    "sweeps",
}
RESERVED_DEFAULT_FIELDS = {"name", "prompts", "output_dir"}
RESERVED_SWEEP_FIELDS = {"prompts", "output_dir"}


def path_safe_name(value, field):
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "\\" in value
    ):
        raise ValueError("%s must be a non-empty path-safe value" % field)


def validate_campaign(config):
    required = {"schema_version", "name", "defaults", "workloads", "sweeps"}
    missing = required.difference(config)
    if missing:
        raise ValueError(
            "missing campaign fields: %s" % ", ".join(sorted(missing))
        )
    unknown = set(config).difference(CAMPAIGN_FIELDS)
    if unknown:
        raise ValueError(
            "unknown campaign fields: %s" % ", ".join(sorted(unknown))
        )
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported campaign schema_version")
    path_safe_name(config["name"], "campaign name")

    defaults = config["defaults"]
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object")
    reserved = RESERVED_DEFAULT_FIELDS.intersection(defaults)
    if reserved:
        raise ValueError(
            "defaults contains campaign-managed fields: %s"
            % ", ".join(sorted(reserved))
        )

    workloads = config["workloads"]
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("workloads must be a non-empty list")
    workload_names = []
    for index, workload in enumerate(workloads):
        if not isinstance(workload, dict):
            raise ValueError("workloads[%d] must be an object" % index)
        if set(workload) != {"name", "config"}:
            raise ValueError(
                "workloads[%d] must contain only name and config" % index
            )
        path_safe_name(workload["name"], "workloads[%d].name" % index)
        if not isinstance(workload["config"], str) or not workload["config"]:
            raise ValueError("workloads[%d].config must be a path" % index)
        workload_names.append(workload["name"])
    if len(workload_names) != len(set(workload_names)):
        raise ValueError("workload names must be unique")

    sweeps = config["sweeps"]
    if not isinstance(sweeps, list) or not sweeps:
        raise ValueError("sweeps must be a non-empty list")
    sweep_names = []
    for index, sweep in enumerate(sweeps):
        if not isinstance(sweep, dict):
            raise ValueError("sweeps[%d] must be an object" % index)
        missing_sweep = {"name", "workload", "concurrency"}.difference(sweep)
        if missing_sweep:
            raise ValueError(
                "sweeps[%d] missing fields: %s"
                % (index, ", ".join(sorted(missing_sweep)))
            )
        path_safe_name(sweep["name"], "sweeps[%d].name" % index)
        if sweep["workload"] not in workload_names:
            raise ValueError(
                "sweeps[%d] references unknown workload: %s"
                % (index, sweep["workload"])
            )
        reserved = RESERVED_SWEEP_FIELDS.intersection(sweep)
        if reserved:
            raise ValueError(
                "sweeps[%d] contains campaign-managed fields: %s"
                % (index, ", ".join(sorted(reserved)))
            )
        sweep_names.append(sweep["name"])
    if len(sweep_names) != len(set(sweep_names)):
        raise ValueError("sweep names must be unique")
    return config


def load_campaign(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return validate_campaign(config)


def resolve_input_path(value, campaign_path):
    path = Path(value)
    if not path.is_absolute():
        path = campaign_path.parent / path
    return path.resolve()


def resolved_output_root(value):
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def build_plan(
    config,
    campaign_path,
    output_root=None,
    concurrency=None,
    num_requests=None,
    warmups=None,
    repeats=None,
    seed=None,
):
    campaign_path = Path(campaign_path).resolve()
    root = resolved_output_root(
        output_root or config.get("output_root", "results/campaigns")
    ) / config["name"]

    workload_plans = []
    workloads = {}
    for item in config["workloads"]:
        source = resolve_input_path(item["config"], campaign_path)
        workload_config = generate_prompts.load_config(source)
        workload_dir = root / "workloads" / item["name"]
        workload_plan = {
            "name": item["name"],
            "source_config": str(source),
            "resolved_config": workload_config,
            "generated_config": str(
                root / "configs" / "workloads" / (item["name"] + ".json")
            ),
            "prompts": str(workload_dir / (workload_config["name"] + ".jsonl")),
            "metadata": str(
                workload_dir / (workload_config["name"] + ".metadata.json")
            ),
        }
        workload_plans.append(workload_plan)
        workloads[item["name"]] = workload_plan

    sweep_plans = []
    for index, sweep in enumerate(config["sweeps"], start=1):
        workload = workloads[sweep["workload"]]
        experiment = {
            **config["defaults"],
            **{
                key: value
                for key, value in sweep.items()
                if key not in {"name", "workload"}
            },
            "name": sweep["name"],
            "prompts": workload["prompts"],
            "output_dir": str(root / "runs"),
            "campaign": config["name"],
            "campaign_workload": sweep["workload"],
        }
        if concurrency is not None:
            experiment["concurrency"] = concurrency
        if num_requests is not None:
            experiment["num_requests"] = num_requests
        if warmups is not None:
            experiment["warmups"] = warmups
        if repeats is not None:
            experiment["repeats"] = repeats
        if seed is not None:
            experiment["seed"] = seed
        run_experiment.validate_config(experiment)
        available = workload["resolved_config"]["request_count"]
        if experiment["num_requests"] > available:
            raise ValueError(
                "sweep %s requests %d rows but workload %s contains %d"
                % (
                    sweep["name"],
                    experiment["num_requests"],
                    sweep["workload"],
                    available,
                )
            )
        sweep_plans.append(
            {
                "order": index,
                "name": sweep["name"],
                "workload": sweep["workload"],
                "output": str(root / "runs" / sweep["name"]),
                "experiment": experiment,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "name": config["name"],
        "campaign_config": str(campaign_path),
        "campaign_root": str(root),
        "workloads": workload_plans,
        "sweeps": sweep_plans,
    }


def select_sweeps(plan, sweep_names=None, workload_names=None):
    known_sweeps = {item["name"] for item in plan["sweeps"]}
    known_workloads = {item["name"] for item in plan["workloads"]}
    requested_sweeps = set(sweep_names or known_sweeps)
    requested_workloads = set(workload_names or known_workloads)
    unknown_sweeps = requested_sweeps.difference(known_sweeps)
    unknown_workloads = requested_workloads.difference(known_workloads)
    if unknown_sweeps:
        raise ValueError("unknown sweeps: %s" % ", ".join(sorted(unknown_sweeps)))
    if unknown_workloads:
        raise ValueError(
            "unknown workloads: %s" % ", ".join(sorted(unknown_workloads))
        )
    selected = [
        item
        for item in plan["sweeps"]
        if item["name"] in requested_sweeps
        and item["workload"] in requested_workloads
    ]
    if not selected:
        raise ValueError("sweep and workload filters select no conditions")
    return selected


def write_once(path, value, description):
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != value:
            raise RuntimeError(
                "%s differs from the existing campaign; use a new campaign name "
                "or output root" % description
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def prepare_workload(workload, tokenizer=None):
    generated_config = Path(workload["generated_config"])
    output_dir = Path(workload["prompts"]).parent
    write_once(
        generated_config,
        workload["resolved_config"],
        "generated workload config",
    )
    prompts = Path(workload["prompts"])
    metadata = Path(workload["metadata"])
    if prompts.exists() != metadata.exists():
        raise RuntimeError(
            "refusing to replace a partial generated workload: %s" % output_dir
        )
    if prompts.exists():
        digest = generate_prompts.verify_workload(generated_config, output_dir)
    else:
        _, _, digest = generate_prompts.write_workload(
            generated_config, output_dir, tokenizer=tokenizer
        )
    return digest


def initial_manifest(plan, git_state):
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": {
            "name": plan["name"],
            "started_at_utc": utc_now(),
            "completed_at_utc": None,
            "status": "running",
        },
        "source": {"git": git_state},
        "sweeps": [
            {
                "name": item["name"],
                "workload": item["workload"],
                "order": item["order"],
                "output": item["output"],
                "status": "not_run",
            }
            for item in plan["sweeps"]
        ],
    }


def ensure_clean(git_state):
    if not git_state.get("available"):
        raise RuntimeError("cannot verify Git state for an official campaign")
    if git_state.get("dirty"):
        raise RuntimeError(
            "refusing to run an official campaign from a dirty checkout; "
            "commit the campaign definition or pass --allow-dirty"
        )


def run_campaign(
    config,
    campaign_path,
    output_root=None,
    concurrency=None,
    num_requests=None,
    warmups=None,
    repeats=None,
    seed=None,
    sweep_names=None,
    workload_names=None,
    dry_run=False,
    allow_dirty=False,
    experiment_runner=run_experiment.run_experiment,
    tokenizer=None,
    repo_root=REPO_ROOT,
):
    validate_campaign(config)
    plan = build_plan(
        config,
        campaign_path,
        output_root=output_root,
        concurrency=concurrency,
        num_requests=num_requests,
        warmups=warmups,
        repeats=repeats,
        seed=seed,
    )
    selected = select_sweeps(plan, sweep_names, workload_names)
    if dry_run:
        return {**plan, "selected_sweeps": [item["name"] for item in selected]}

    git_state = collect_git(repo_root)
    if not allow_dirty:
        ensure_clean(git_state)

    root = Path(plan["campaign_root"])
    root.mkdir(parents=True, exist_ok=True)
    write_once(root / "campaign.original.json", sanitize(config), "campaign config")
    write_once(root / "campaign.resolved.json", sanitize(plan), "campaign plan")

    manifest_path = root / "campaign-manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["campaign"].update(
            {"status": "running", "completed_at_utc": None}
        )
    else:
        manifest = initial_manifest(plan, git_state)
    write_json(manifest_path, manifest)

    workloads = {item["name"]: item for item in plan["workloads"]}
    prepared = {}
    status_by_name = {item["name"]: item for item in manifest["sweeps"]}
    active_status = None
    try:
        for sweep in selected:
            workload_name = sweep["workload"]
            if workload_name not in prepared:
                prepared[workload_name] = prepare_workload(
                    workloads[workload_name], tokenizer=tokenizer
                )
            status = status_by_name[sweep["name"]]
            active_status = status
            status["status"] = "running"
            status["workload_sha256"] = prepared[workload_name]
            write_json(manifest_path, manifest)
            experiment_runner(
                sweep["experiment"],
                root / "runs",
                original_config=sweep["experiment"],
                repo_root=repo_root,
            )
            status["status"] = "completed"
            active_status = None
            write_json(manifest_path, manifest)
    except Exception:
        if active_status is not None:
            active_status["status"] = "failed"
        manifest["campaign"]["status"] = "failed"
        write_json(manifest_path, manifest)
        raise

    all_completed = all(
        item["status"] == "completed" for item in manifest["sweeps"]
    )
    manifest["campaign"].update(
        {
            "status": "completed" if all_completed else "partial",
            "completed_at_utc": utc_now() if all_completed else None,
        }
    )
    write_json(manifest_path, manifest)
    return {"plan": plan, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="campaign JSON definition")
    parser.add_argument("--output-root")
    parser.add_argument("--sweep", action="append", dest="sweep_names")
    parser.add_argument("--workload", action="append", dest="workload_names")
    parser.add_argument("--concurrency", nargs="+", type=int)
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    try:
        config = load_campaign(args.config)
        result = run_campaign(
            config,
            args.config,
            output_root=args.output_root,
            concurrency=args.concurrency,
            num_requests=args.num_requests,
            warmups=args.warmups,
            repeats=args.repeats,
            seed=args.seed,
            sweep_names=args.sweep_names,
            workload_names=args.workload_names,
            dry_run=args.dry_run,
            allow_dirty=args.allow_dirty,
        )
        if args.dry_run:
            print(json.dumps(result, indent=2, sort_keys=True))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, "error: %s\n" % error)


if __name__ == "__main__":
    main()

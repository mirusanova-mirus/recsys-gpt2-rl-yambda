"""Small submission archive with an independently verifiable SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        assert len(names)==len(set(names)),"Duplicate members"
        assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names)
        manifest=json.loads(z.read("task2_rl/MANIFEST.sha256.json"))
        assert set(names)==set(manifest["sha256"])|{"task2_rl/MANIFEST.sha256.json"}
        for name,value in manifest["sha256"].items():
            assert digest(z.read(name))==value,(name,"checksum mismatch")
        assert z.testzip() is None
    return {"archive":str(Path(path).resolve()),"verified_payload_files":len(manifest["sha256"]),"bytes":Path(path).stat().st_size,
        "sha256":digest(Path(path).read_bytes())}


def build(config):
    out=Path(config["output_dir"])
    assert (out/"validation_final.json").exists(),"Run final audit first"
    assert Path("output/pdf/yambda_gpt2_rl_research.pdf").exists(),"Create the PDF first"
    files=[Path(p) for p in ["README.md","REPORT.md","EXAMPLES.md","EXPERIMENT_PLAN.md","DIAGNOSTIC_PLAN.md","requirements.txt","Makefile",".gitignore","data/SOURCE_MANIFEST.json"]]
    for base in [Path("seqrl"),Path("tests"),Path("configs"),Path("output")]:
        files.extend(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for p in out.rglob("*"):
        if p.is_file() and not any(part in ["cache","models"] for part in p.relative_to(out).parts):
            files.append(p)
    # Retain only the factual design audit of the rejected small-vocabulary pilot.
    pilot=Path("artifacts/design_2048/audit.json")
    if pilot.exists():
        files.append(pilot)
    engineering=Path("artifacts/engineering_sampling_v1")
    if engineering.exists():
        files.extend(p for p in engineering.iterdir() if p.is_file() and p.suffix in [".md",".json",".log"])
    files=sorted(set(files))
    assert all(p.suffix not in [".pt",".cbm",".joblib",".parquet"] for p in files)
    payload={f"task2_rl/{p.as_posix()}":p.read_bytes() for p in files}
    payload[f"task2_rl/{out.as_posix()}/ARCHIVE_RESULTS_ONLY.json"]=json.dumps({
        "purpose":"These are reference measurements; fitted weights and tensor caches are not included.",
        "reproduce":"seqrl.build preserves this folder as artifacts/reference_main_N before constructing a fresh main run."},indent=2).encode()
    manifest={"sha256":{name:digest(data) for name,data in payload.items()},"omitted":"raw/processed data, tensor caches and fitted weights; regenerate with make reproduce"}
    target=Path("submission/recsys_rl.zip");target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in payload.items():
            z.writestr(name,data)
        z.writestr("task2_rl/MANIFEST.sha256.json",json.dumps(manifest,indent=2,ensure_ascii=False))
    result=verify(target)
    (target.parent/"verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/main.json");p.add_argument("--verify")
    args=p.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify),indent=2))
    else:
        build(json.loads(Path(args.config).read_text()))

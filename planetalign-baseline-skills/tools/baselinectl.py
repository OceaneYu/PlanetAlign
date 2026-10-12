#!/usr/bin/env python3
# 文件作用：管理外部基准复现的隔离目录、状态清单、源码变动检查和正式集成检查。
# 数据来源：main() 从命令行解析子命令；各 cmd_*() 读取 .reference/<name>/manifest.json 与 git 状态。
# 结果去向：初始化、阶段、补丁与完成命令更新隔离工件；guard 和 verify-integration 报告检查结果。
# 阅读建议：先读 build_parser() → main() → 对应 cmd_*()；流程说明见 planetalign-baseline-skills/README.md。

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


# 功能：执行指定外部命令，按参数捕获输出；check=True 时失败抛出 RuntimeError。
def run(cmd, cwd=None, check=True, text=True):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    if check and p.returncode != 0:
        raise RuntimeError(
            f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n"
            f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p


def git_root() -> Path:
    p = run(["git", "rev-parse", "--show-toplevel"])
    return Path(p.stdout.strip()).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def status_paths(root: Path):
    p = run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root)
    items = []
    for line in p.stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        raw_path = line[3:]
        # For renames/copies, compare the destination path.
        path = raw_path.split(" -> ", 1)[-1]
        path = path.strip('"')
        if path == ".reference" or path.startswith(".reference/"):
            continue
        items.append((status, path))
    return items


# 功能：记录隔离目录之外的宿主源码状态与哈希，供 cmd_guard() 比较复现前后变化。
def capture_host_state(root: Path):
    state = {}
    for status, rel in status_paths(root):
        p = root / rel
        record = {"status": status}
        if p.exists() and p.is_file():
            record["sha256"] = sha256_file(p)
            record["size"] = p.stat().st_size
        elif p.exists() and p.is_dir():
            record["kind"] = "dir"
        else:
            record["kind"] = "missing"
        idx = run(["git", "ls-files", "-s", "--", rel], cwd=root, check=False)
        record["index"] = idx.stdout.strip()
        state[rel] = record
    return state


def repo_slug(url: str) -> str:
    path = urllib.parse.urlparse(url).path.rstrip("/")
    slug = Path(path).name or "baseline"
    if slug.endswith(".git"):
        slug = slug[:-4]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-._")
    return slug or "baseline"


def baseline_dir(root: Path, name: str) -> Path:
    return root / ".reference" / name


def manifest_path(root: Path, name: str) -> Path:
    return baseline_dir(root, name) / "baseline_manifest.json"


# 功能：读取指定外部基准的 JSON 清单，尚未初始化时提示先执行 init。
def load_manifest(root: Path, name: str):
    path = manifest_path(root, name)
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# 功能：更新时间并写回基准清单，供各状态变更子命令使用。
def save_manifest(root: Path, name: str, data):
    data["updated_at"] = now()
    path = manifest_path(root, name)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# 功能：在 .git/info/exclude 中加入 .reference/，只影响本地忽略规则。
def ensure_local_exclude(root: Path):
    exclude = root / ".git" / "info" / "exclude"
    if not exclude.exists():
        return
    content = exclude.read_text(encoding="utf-8", errors="ignore")
    marker = "/.reference/"
    if marker not in content.splitlines():
        with exclude.open("a", encoding="utf-8") as f:
            if content and not content.endswith("\n"):
                f.write("\n")
            f.write(marker + "\n")


# 功能：复制本地论文或按 URL 下载至隔离论文目录，返回保存路径。
def copy_or_download_paper(paper: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(paper)
    if parsed.scheme in ("http", "https"):
        name = Path(parsed.path).name or "paper.pdf"
        if not Path(name).suffix:
            name += ".pdf"
        dst = out_dir / name
        if not dst.exists():
            urllib.request.urlretrieve(paper, dst)
        return dst
    src = Path(paper).expanduser().resolve()
    if not src.exists():
        raise RuntimeError(f"paper not found: {src}")
    dst = out_dir / src.name
    if src != dst:
        shutil.copy2(src, dst)
    return dst


# 核心思路：创建 .reference/<name> 目录结构、克隆外部源码并保存论文，记录参考提交与宿主初始状态。
# 输出结果：写入 manifest.json；已有同一仓库的清单时返回恢复信息。
def cmd_init(args):
    root = git_root()
    ensure_local_exclude(root)
    name = args.name or repo_slug(args.repo)
    bdir = baseline_dir(root, name)
    mpath = manifest_path(root, name)

    if mpath.exists():
        m = load_manifest(root, name)
        if m.get("repo_url") != args.repo:
            raise RuntimeError(
                f"{bdir} already belongs to {m.get('repo_url')}, not {args.repo}"
            )
        print(json.dumps({"name": name, "baseline_dir": str(bdir), "resumed": True}, indent=2))
        return

    bdir.mkdir(parents=True, exist_ok=False)
    for rel in [
        "paper",
        "patches",
        "reproduction/configs",
        "reproduction/logs",
        "reproduction/results",
        "planetalign_benchmark/adapter",
        "planetalign_benchmark/configs",
        "planetalign_benchmark/logs",
        "planetalign_benchmark/results",
        "integration",
    ]:
        (bdir / rel).mkdir(parents=True, exist_ok=True)

    source = bdir / "source"
    run(["git", "clone", args.repo, str(source)], cwd=root)
    upstream = run(["git", "rev-parse", "HEAD"], cwd=source).stdout.strip()
    paper_dst = copy_or_download_paper(args.paper, bdir / "paper")
    host_head = run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()

    manifest = {
        "schema_version": 1,
        "baseline_name": name,
        "repo_url": args.repo,
        "paper_input": args.paper,
        "paper_file": str(paper_dst.relative_to(root)),
        "upstream_commit": upstream,
        "host_root": str(root),
        "host_head_at_start": host_head,
        "host_state_at_start": capture_host_state(root),
        "started_at": now(),
        "updated_at": now(),
        "phases": {},
        "reference_status": "IN_PROGRESS",
        "integration_status": "NOT_STARTED",
        "guard_violations": [],
    }
    save_manifest(root, name, manifest)
    print(json.dumps({"name": name, "baseline_dir": str(bdir), "resumed": False}, indent=2))


# 输出结果：返回两个宿主状态快照中不一致的路径记录，供 cmd_guard() 使用。
def host_state_delta(initial, current):
    keys = sorted(set(initial) | set(current))
    changes = []
    for k in keys:
        if initial.get(k) != current.get(k):
            changes.append(
                {"path": k, "initial": initial.get(k), "current": current.get(k)}
            )
    return changes


# 功能：与初始宿主状态比较，发现隔离目录外的新改动则记录违规并以退出码 2 报告。
def cmd_guard(args):
    root = git_root()
    m = load_manifest(root, args.name)
    current = capture_host_state(root)
    changes = host_state_delta(m.get("host_state_at_start", {}), current)
    if changes:
        entry = {"at": now(), "changes": changes}
        m.setdefault("guard_violations", []).append(entry)
        save_manifest(root, args.name, m)
        print(json.dumps({"ok": False, "violations": changes}, indent=2, ensure_ascii=False))
        sys.exit(2)
    print(json.dumps({"ok": True, "message": "no new host-source changes detected"}, indent=2))


# 功能：将一个复现阶段的状态、摘要与时间写入清单。
def cmd_phase(args):
    root = git_root()
    m = load_manifest(root, args.name)
    m.setdefault("phases", {})[args.phase] = {
        "status": args.status,
        "summary": args.summary,
        "updated_at": now(),
    }
    save_manifest(root, args.name, m)
    print(json.dumps(m["phases"][args.phase], indent=2, ensure_ascii=False))


# 功能：将外部源码目录的未跟踪文件以二进制兼容 diff 追加到补丁文件。
def append_untracked_patch(source: Path, patch_path: Path):
    p = run(["git", "ls-files", "--others", "--exclude-standard"], cwd=source)
    files = [x for x in p.stdout.splitlines() if x]
    if not files:
        return
    with patch_path.open("ab") as out:
        for rel in files:
            fp = source / rel
            if not fp.is_file():
                continue
            diff = subprocess.run(
                ["git", "diff", "--no-index", "--binary", "--", "/dev/null", str(fp)],
                cwd=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # git diff --no-index returns 1 when files differ.
            if diff.returncode in (0, 1):
                out.write(diff.stdout)


# 功能：从参考提交提取外部源码差异，追加未跟踪文件补丁并保存源码状态。
def cmd_save_patch(args):
    root = git_root()
    m = load_manifest(root, args.name)
    bdir = baseline_dir(root, args.name)
    source = bdir / "source"
    patch = bdir / "patches" / "codex.patch"
    upstream = m["upstream_commit"]

    p = run(["git", "diff", "--binary", upstream, "--"], cwd=source, check=False, text=False)
    patch.write_bytes(p.stdout)
    append_untracked_patch(source, patch)
    status = run(["git", "status", "--short"], cwd=source).stdout
    (bdir / "patches" / "source_status.txt").write_text(status, encoding="utf-8")

    print(json.dumps({"patch": str(patch), "source_dirty": bool(status.strip())}, indent=2))


# 功能：要求复现 REPORT.md 已存在，再记录 reference_status 与结束时间。
def cmd_finalize_reference(args):
    root = git_root()
    bdir = baseline_dir(root, args.name)
    report = bdir / "REPORT.md"
    if not report.exists():
        raise RuntimeError(f"REPORT.md is required before finalization: {report}")
    m = load_manifest(root, args.name)
    m["reference_status"] = args.status
    m["reference_finalized_at"] = now()
    save_manifest(root, args.name, m)
    print(json.dumps({"reference_status": args.status, "report": str(report)}, indent=2))


# 功能：读取并打印指定基准清单，不改变状态。
def cmd_show(args):
    root = git_root()
    m = load_manifest(root, args.name)
    print(json.dumps(m, indent=2, ensure_ascii=False))


# 功能：检查复现状态为 PASS/PARTIAL 且报告存在，供正式集成流程开始前使用。
def cmd_integration_preflight(args):
    root = git_root()
    m = load_manifest(root, args.name)
    status = m.get("reference_status")
    report = baseline_dir(root, args.name) / "REPORT.md"
    if status not in {"PASS", "PARTIAL"}:
        raise RuntimeError(f"reference status must be PASS or PARTIAL, got {status}")
    if not report.exists():
        raise RuntimeError(f"missing reference report: {report}")
    print(json.dumps({
        "ok": True,
        "baseline": args.name,
        "reference_status": status,
        "report": str(report),
    }, indent=2))


def iter_text_files(path: Path):
    exts = {".py", ".pyi", ".toml", ".yaml", ".yml", ".json", ".md", ".txt", ".sh"}
    for p in path.rglob("*"):
        if p.is_symlink():
            yield p
        elif p.is_file() and p.suffix.lower() in exts:
            yield p


# 功能：扫描正式算法目录，拒绝仍指向 .reference 的源码文本或符号链接依赖。
def cmd_verify_integration(args):
    root = git_root()
    alg = (root / args.algorithm_path).resolve()
    if not alg.exists() or not alg.is_dir():
        raise RuntimeError(f"algorithm path not found: {alg}")

    violations = []
    for p in iter_text_files(alg):
        if p.is_symlink():
            target = p.resolve()
            if ".reference" in target.parts:
                violations.append({"file": str(p.relative_to(root)), "reason": "symlink resolves into .reference"})
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if ".reference" in text or "../.reference" in text:
            violations.append({"file": str(p.relative_to(root)), "reason": "contains .reference runtime/path reference"})

    if violations:
        print(json.dumps({"ok": False, "violations": violations}, indent=2, ensure_ascii=False))
        sys.exit(2)
    print(json.dumps({"ok": True, "algorithm_path": str(alg)}, indent=2))


# 功能：要求 integration/REPORT.md 已存在，再记录集成状态与结束时间。
def cmd_finalize_integration(args):
    root = git_root()
    bdir = baseline_dir(root, args.name)
    report = bdir / "integration" / "REPORT.md"
    if not report.exists():
        raise RuntimeError(f"integration/REPORT.md is required: {report}")
    m = load_manifest(root, args.name)
    m["integration_status"] = args.status
    m["integration_finalized_at"] = now()
    save_manifest(root, args.name, m)
    print(json.dumps({"integration_status": args.status, "report": str(report)}, indent=2))


# 功能：定义各子命令与参数，并将 args.func 绑定到对应 cmd_*() 处理函数。
def build_parser():
    ap = argparse.ArgumentParser(description="PlanetAlign external baseline workflow guard/manifest utility")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--repo", required=True)
    p.add_argument("--paper", required=True)
    p.add_argument("--name")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("guard")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_guard)

    p = sub.add_parser("phase")
    p.add_argument("--name", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--status", required=True, choices=["PASS", "PARTIAL", "FAIL", "IN_PROGRESS", "SKIPPED"])
    p.add_argument("--summary", required=True)
    p.set_defaults(func=cmd_phase)

    p = sub.add_parser("save-patch")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_save_patch)

    p = sub.add_parser("finalize-reference")
    p.add_argument("--name", required=True)
    p.add_argument("--status", required=True, choices=["PASS", "PARTIAL", "FAIL"])
    p.set_defaults(func=cmd_finalize_reference)

    p = sub.add_parser("show")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("integration-preflight")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_integration_preflight)

    p = sub.add_parser("verify-integration")
    p.add_argument("--name", required=True)
    p.add_argument("--algorithm-path", required=True)
    p.set_defaults(func=cmd_verify_integration)

    p = sub.add_parser("finalize-integration")
    p.add_argument("--name", required=True)
    p.add_argument("--status", required=True, choices=["PASS", "PARTIAL", "FAIL"])
    p.set_defaults(func=cmd_finalize_integration)

    return ap


# 核心思路：解析子命令后调用 args.func(args)，错误统一打印并返回失败退出码。
def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(f"baselinectl: ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""把 docs/ 里的代码清单拼装成 scratch_agents 包。

用法：
    python docs/assemble.py --check        # 与仓库 ./scratch_agents 逐字节比对（默认）
    python docs/assemble.py --out DIR      # 拼装到 DIR/scratch_agents
    python docs/assemble.py --list         # 只列出每个文件取自哪一章

规则：按文件名排序读取 docs/ch*.md，遇到 `<!-- FILE: <路径> -->` 标记的 python 代码块
就记录下来；同一文件出现多次时，排后面的章节覆盖前面的（即「本章状态」之后的状态胜出）。
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent
ROOT = DOC_DIR.parent

FILE_RE = re.compile(r"<!--\s*FILE:\s*(\S+?)\s*-->")


def extract():
    """返回 {相对路径: (代码内容, 来源章节)}，后者覆盖前者。"""
    files = {}
    for md in sorted(DOC_DIR.glob("ch*.md")):
        text = md.read_text(encoding="utf-8")
        for m in FILE_RE.finditer(text):
            rel = m.group(1)
            fence = text.find("````python", m.end())
            if fence == -1:
                continue
            start = text.find("\n", fence) + 1
            end = text.find("````", start)
            files[rel] = (text[start:end].rstrip("\n") + "\n", md.name)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="拼装输出目录（默认比对 ./scratch_agents）")
    parser.add_argument("--check", action="store_true", help="与 ./scratch_agents 逐字节比对（默认行为）")
    parser.add_argument("--list", action="store_true", help="列出文件来源")
    args = parser.parse_args()

    files = extract()
    if not files:
        sys.exit("docs/ch*.md 中没有找到任何 FILE 清单")

    if args.list:
        for rel, (_, src) in sorted(files.items()):
            print(f"{rel:<45} <- {src}")
        return

    if args.out:
        out_root = Path(args.out)
        for rel, (content, _) in files.items():
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        print(f"已拼装 {len(files)} 个文件到 {out_root}")
        return

    # --check：与仓库现有 scratch_agents 比对
    target = ROOT / "scratch_agents"
    failed = False
    disk = {f"scratch_agents/{p.relative_to(target)}": p
            for p in target.rglob("*") if p.is_file()}
    for rel, (content, src) in sorted(files.items()):
        path = ROOT / rel
        if not path.exists():
            print(f"MISSING ON DISK: {rel} (docs 中由 {src} 提供)")
            failed = True
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != content:
            failed = True
            print(f"DIFF: {rel} (docs 中来自 {src})")
            diff = difflib.unified_diff(
                actual.splitlines(), content.splitlines(),
                fromfile=f"repo/{rel}", tofile=f"docs/{rel}", lineterm="")
            for line in list(diff)[:20]:
                print("   ", line)
    for rel in sorted(set(disk) - set(files)):
        print(f"NOT IN DOCS: {rel}")
        failed = True
    if failed:
        sys.exit(1)
    print(f"OK：docs 拼装结果与 {target} 完全一致（{len(files)} 个文件）")


if __name__ == "__main__":
    main()

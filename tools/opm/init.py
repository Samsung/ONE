#!/usr/bin/env python3
import os
import shutil
import subprocess
import venv
from config import VENV_DIR, O2O_DIR, BUILD_DIR
from util import run_cmd

# PR-related constants (used only in init.py)
PR_WORKTREE = "_pr_16233"
PR_BRANCH = "pr-16233"
PR_REF = "refs/pull/16233/head"


def main():
    # 1. Create virtual environment
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment at {VENV_DIR}...")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(f"Virtual environment already exists at {VENV_DIR}.")

    # 2. Install opm requirements
    pip_cmd = os.path.join(VENV_DIR, "bin", "pip")
    # By installing Torch (cpu), it prevents TICO from pulling the large CUDA-enabled PyTorch package
    run_cmd(f"{pip_cmd} install torch --index-url https://download.pytorch.org/whl/cpu")
    run_cmd(f"{pip_cmd} install tico==0.1.0.dev251125")

    # 3. Prepare build directory for temporary worktree
    os.makedirs(BUILD_DIR, exist_ok=True)
    worktree_path = os.path.join(BUILD_DIR, PR_WORKTREE)

    # 4. Git worktree for PR and o2o extraction
    if not os.path.exists(worktree_path):
        # Fetch PR only if worktree doesn't exist
        try:
            run_cmd(f"git fetch https://github.com/Samsung/ONE.git {PR_REF}:{PR_BRANCH}")
        except subprocess.CalledProcessError:
            print("Fetch failed, possibly branch already exists. Continuing...")

        # Create worktree with no checkout
        run_cmd(f"git worktree add --no-checkout -f {worktree_path} {PR_BRANCH}")

        # Configure sparse checkout
        cwd = os.getcwd()
        try:
            os.chdir(worktree_path)
            run_cmd("git sparse-checkout init --cone")
            run_cmd(f"git sparse-checkout set tools/{O2O_DIR}")
            # Populate files
            run_cmd(f"git checkout {PR_BRANCH}")
        finally:
            os.chdir(cwd)

    # 5. Move o2o to current directory
    if not os.path.exists(O2O_DIR):
        src_o2o = os.path.join(worktree_path, "tools", O2O_DIR)
        if os.path.exists(src_o2o):
            print(f"Moving o2o tools to {O2O_DIR}...")
            shutil.move(src_o2o, O2O_DIR)
        else:
            print("o2o tools not found in worktree.")

    # 6. Remove temporary worktree
    if os.path.exists(worktree_path):
        print("Removing temporary worktree...")
        run_cmd(f"git worktree remove --force {worktree_path}")

    # 7. Make tools executable
    if os.path.exists(O2O_DIR):
        run_cmd(f"chmod +x {O2O_DIR}/*.py")

    print("opm init completed.")


if __name__ == "__main__":
    main()

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::Command;

const REPO_URL: &str = "https://github.com/Lemondeft/lrc-align.git";

pub fn run_self_update(force_gpu: bool, cpu_only: bool) -> Result<()> {
    let repo_dir = resolve_or_clone_repo()?;
    println!("Updating source repository at {}...", repo_dir.display());

    let before_rev = git_short_rev(&repo_dir);
    let pull = Command::new("git")
        .args(["pull", "--ff-only"])
        .current_dir(&repo_dir)
        .status()
        .context("Failed to run `git pull --ff-only` (is git installed?)")?;
    if !pull.success() {
        return Err(anyhow!(
            "`git pull --ff-only` failed in {}. Commit or stash local changes first.",
            repo_dir.display()
        ));
    }
    let after_rev = git_short_rev(&repo_dir);

    let feature: Option<&str> = if cpu_only {
        None
    } else if force_gpu {
        if cfg!(target_os = "macos") {
            Some("metal")
        } else {
            Some("cuda")
        }
    } else if cfg!(feature = "cuda") {
        Some("cuda")
    } else if cfg!(feature = "metal") {
        Some("metal")
    } else {
        None
    };

    let mut build_args = vec!["build", "--release"];
    if let Some(feat) = feature {
        build_args.push("--features");
        build_args.push(feat);
        println!("Rebuilding release binary with --features {feat}...");
    } else {
        println!("Rebuilding release binary (CPU out-of-the-box)...");
    }

    let build = Command::new("cargo")
        .args(&build_args)
        .current_dir(&repo_dir)
        .status()
        .context("Failed to run `cargo build --release` (is cargo in PATH?)")?;
    if !build.success() {
        return Err(anyhow!("`cargo build --release` failed"));
    }

    let built_bin = repo_dir.join("target/release/lrc-align");
    if !built_bin.is_file() {
        return Err(anyhow!(
            "Built binary not found at {}",
            built_bin.display()
        ));
    }

    let mut installed_paths = Vec::new();
    if let Ok(cur_exe) = std::env::current_exe() {
        let cur_canon = std::fs::canonicalize(&cur_exe).unwrap_or(cur_exe);
        let built_canon = std::fs::canonicalize(&built_bin).unwrap_or_else(|_| built_bin.clone());
        if cur_canon != built_canon {
            install_binary_atomic(&built_bin, &cur_canon)?;
            installed_paths.push(cur_canon);
        }
    }

    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        let local_bin = home.join(".local/bin/lrc-align");
        if local_bin.parent().map(|p| p.is_dir()).unwrap_or(false) {
            let already = installed_paths
                .iter()
                .any(|p| std::fs::canonicalize(p).ok() == std::fs::canonicalize(&local_bin).ok());
            if !already {
                install_binary_atomic(&built_bin, &local_bin)?;
                installed_paths.push(local_bin);
            }
        }
    }

    println!();
    match (before_rev, after_rev) {
        (Some(a), Some(b)) if a != b => println!("Updated lrc-align: {a} -> {b}"),
        (_, Some(b)) => println!("lrc-align rebuilt at commit {b}"),
        _ => println!("lrc-align update complete."),
    }
    if installed_paths.is_empty() {
        println!("Binary ready at {}", built_bin.display());
    } else {
        for p in installed_paths {
            println!("Installed binary -> {}", p.display());
        }
    }
    Ok(())
}

fn resolve_or_clone_repo() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if is_lrc_align_repo(&manifest_dir) {
        return Ok(manifest_dir);
    }
    if let Ok(cwd) = std::env::current_dir() {
        if is_lrc_align_repo(&cwd) {
            return Ok(cwd);
        }
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("Could not locate source repository or HOME directory"))?;
    let fallback = home.join(".local/share/lrc-align-src");
    if is_lrc_align_repo(&fallback) {
        return Ok(fallback);
    }

    println!("Cloning {REPO_URL} into {}...", fallback.display());
    if let Some(parent) = fallback.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let status = Command::new("git")
        .args(["clone", REPO_URL])
        .arg(&fallback)
        .status()
        .context("Failed to run `git clone`")?;
    if !status.success() {
        return Err(anyhow!("Failed to clone {REPO_URL}"));
    }
    Ok(fallback)
}

fn is_lrc_align_repo(dir: &Path) -> bool {
    dir.join(".git").exists() && dir.join("Cargo.toml").is_file()
}

fn git_short_rev(dir: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(dir)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn install_binary_atomic(src: &Path, dst: &Path) -> Result<()> {
    let parent = dst
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(".lrcalign_update_{}", uuid::Uuid::new_v4().simple()));
    std::fs::copy(src, &tmp)
        .with_context(|| format!("Failed to stage binary at {}", tmp.display()))?;
    // Atomic rename avoids ETXTBSY when replacing a currently running binary on Linux.
    if let Err(err) = std::fs::rename(&tmp, dst) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err).with_context(|| format!("Failed to install binary to {}", dst.display()));
    }
    Ok(())
}

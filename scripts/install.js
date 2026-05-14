const { spawnSync } = require("node:child_process");
const { existsSync } = require("node:fs");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "..");
const venvDir = path.join(repoRoot, ".venv");
const venvPython =
  process.platform === "win32"
    ? path.join(venvDir, "Scripts", "python.exe")
    : path.join(venvDir, "bin", "python");

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    stdio: "inherit",
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function findPythonLauncher() {
  const candidates =
    process.platform === "win32"
      ? [
          ["py", ["-3"]],
          ["python", []],
          ["python3", []],
        ]
      : [
          ["python3", []],
          ["python", []],
        ];

  for (const [command, args] of candidates) {
    const result = spawnSync(command, [...args, "--version"], {
      cwd: repoRoot,
      stdio: "ignore",
    });
    if (result.status === 0) {
      return { command, args };
    }
  }

  return null;
}

function ensureVenv() {
  if (existsSync(venvPython)) {
    return;
  }

  const launcher = findPythonLauncher();
  if (launcher === null) {
    console.error(
      "No usable Python interpreter was found. Install Python 3.10+ and rerun npm run install.",
    );
    process.exit(1);
  }

  run(launcher.command, [...launcher.args, "-m", "venv", ".venv"]);
}

function main() {
  ensureVenv();
  run(venvPython, ["-m", "pip", "install", "--upgrade", "pip"]);
  run(venvPython, ["-m", "pip", "install", "-e", ".[dev]"]);
}

main();
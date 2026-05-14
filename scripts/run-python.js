const { spawnSync } = require("node:child_process");
const { existsSync } = require("node:fs");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "..");
const venvPython =
  process.platform === "win32"
    ? path.join(repoRoot, ".venv", "Scripts", "python.exe")
    : path.join(repoRoot, ".venv", "bin", "python");

function main() {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.error("No Python command was provided.");
    process.exit(1);
  }

  if (!existsSync(venvPython)) {
    console.error(
      'Virtual environment not found. Run "npm run install" before using npm build/test/lint/typecheck scripts.',
    );
    process.exit(1);
  }

  const result = spawnSync(venvPython, args, {
    cwd: repoRoot,
    stdio: "inherit",
  });

  if (result.error) {
    throw result.error;
  }

  process.exit(result.status ?? 1);
}

main();
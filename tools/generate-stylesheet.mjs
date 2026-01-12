#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

function findRepoRoot(startDir) {
    let dir = path.resolve(startDir);
    while (true) {
        if (fs.existsSync(path.join(dir, "composer.json"))) return dir;
        const parent = path.dirname(dir);
        if (parent === dir) return path.resolve(startDir);
        dir = parent;
    }
}

function readFontNameFromComposer(repoRoot) {
    const composerPath = path.join(repoRoot, "composer.json");
    if (!fs.existsSync(composerPath)) {
        throw new Error(`composer.json not found at ${composerPath}`);
    }
    const raw = fs.readFileSync(composerPath, "utf8");
    let data;
    try {
        data = JSON.parse(raw);
    } catch (e) {
        throw new Error(`Invalid JSON in composer.json: ${e.message}`);
    }
    const name = data?.name;
    if (typeof name !== "string" || !name.includes("/")) {
        throw new Error(`composer.json "name" must be like "vendor/fontname", got: ${JSON.stringify(name)}`);
    }
    const [, fontname] = name.split("/", 2);
    if (!fontname) throw new Error(`Invalid composer.json "name": ${name}`);
    return fontname;
}

function parseArgs(argv) {
    const args = {
        // default becomes "auto": prefer scss, then sass
        inPath: null,
        outDir: "dist/fonts",
        style: "expanded",
        watch: false,
        fontname: null,
        loadPaths: [],
    };

    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];

        if (a === "--watch" || a === "-w") args.watch = true;
        else if (a === "--in") args.inPath = argv[++i];
        else if (a === "--outdir") args.outDir = argv[++i];
        else if (a === "--style") args.style = argv[++i];
        else if (a === "--fontname") args.fontname = argv[++i];
        else if (a === "--load-path") args.loadPaths.push(argv[++i]);
        else if (a === "-h" || a === "--help") {
            console.log(`
Usage:
  node tools/generate-stylesheet.mjs [options]

Options:
  --in <file>          Input SASS/SCSS file (default: auto-detect src/style.scss then src/style.sass)
  --outdir <dir>       Output directory (default: dist/fonts)
  --style <style>      expanded|compressed (default: expanded)
  --watch, -w          Watch for changes
  --fontname <name>    Override fontname (otherwise from composer.json vendor/fontname)
  --load-path <dir>    Extra load path for @use/@import (repeatable)
`);
            process.exit(0);
        }
    }

    if (args.style !== "expanded" && args.style !== "compressed") {
        throw new Error(`--style must be expanded or compressed, got: ${args.style}`);
    }

    return args;
}

function detectInputFile(repoRoot, explicitInPath) {
    if (explicitInPath) return path.resolve(repoRoot, explicitInPath);

    const scss = path.resolve(repoRoot, "src/style.scss");
    const sass = path.resolve(repoRoot, "src/style.sass");

    if (fs.existsSync(scss)) return scss;
    if (fs.existsSync(sass)) return sass;

    throw new Error(`No input file found. Expected ${scss} or ${sass} (or pass --in <file>).`);
}

function runSass({ repoRoot, inFile, outFile, style, watch, loadPaths }) {
    const isWin = process.platform === "win32";

    // Use the local install: node_modules/.bin/sass(.cmd)
    const sassBin = path.join(
        repoRoot,
        "node_modules",
        ".bin",
        isWin ? "sass.cmd" : "sass"
    );

    if (!fs.existsSync(sassBin)) {
        throw new Error(`sass not found at ${sassBin}. Did you run: npm i -D sass ?`);
    }

    const args = [];
    if (watch) args.push("--watch");
    args.push(`--style=${style}`);

    // Allow imports relative to input folder + any extra load paths
    args.push("--load-path", path.dirname(inFile));
    for (const p of loadPaths) args.push("--load-path", p);

    args.push(inFile, outFile);

    // ✅ Key fix: on Windows, .cmd shims often need a shell
    const child = spawn(sassBin, args, {
        stdio: "inherit",
        shell: isWin,
        windowsVerbatimArguments: isWin,
    });

    child.on("exit", (code) => process.exit(code ?? 0));
}

async function main() {
    const repoRoot = findRepoRoot(process.cwd());
    const args = parseArgs(process.argv.slice(2));

    const fontname = args.fontname ?? readFontNameFromComposer(repoRoot);
    const inFile = detectInputFile(repoRoot, args.inPath);

    if (!fs.existsSync(inFile)) throw new Error(`Input file not found: ${inFile}`);

    const outDir = path.resolve(repoRoot, args.outDir);
    fs.mkdirSync(outDir, { recursive: true });

    const outFile = path.join(outDir, `${fontname}.css`);

    runSass({
        repoRoot,
        inFile,
        outFile,
        style: args.style,
        watch: args.watch,
        loadPaths: args.loadPaths.map((p) => path.resolve(repoRoot, p)),
    });
}

main().catch((err) => {
    console.error(err?.stack || String(err));
    process.exit(1);
});

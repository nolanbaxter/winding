// The CHANGELOG section for one version, on stdout.
//
// Lives beside the workflow rather than in tools/, because tools/ is ignored
// -- it holds local harnesses, correctly -- and this is release
// infrastructure that CI cannot run without. It was in tools/ for one release
// and the job failed with MODULE_NOT_FOUND on a clean clone.
//
//   node .github/release-notes.js 0.7.4
//
// Two jobs, and the second is the reason this exists rather than a grep in a
// workflow file. It prints the notes, and it FAILS when there is no section to
// print -- which is what makes "the version was bumped" and "the release was
// written up" one event instead of two things to remember separately.
//
// A release with no changelog entry is not a release, it is a version number.

import { readFileSync } from 'node:fs';

const version = process.argv[2];
if (!version) {
  console.error('usage: node .github/release-notes.js <version>');
  process.exit(2);
}

const changelog = readFileSync(new URL('../CHANGELOG.md', import.meta.url), 'utf8')
  .replace(/\r\n/g, '\n');

// The heading this version's section starts at. Keep a Changelog's shape:
// `## [0.7.4] - 2026-09-23`, with the link definitions at the bottom of the
// file using the same `[0.7.4]:` spelling -- which is why the match is
// anchored to a line start and a bracket, not merely to the number appearing.
const heading = new RegExp(`^## \\[${version.replace(/\./g, '\\.')}\\](.*)$`, 'm');
const start = changelog.match(heading);

if (!start) {
  console.error(
    `CHANGELOG.md has no "## [${version}]" section.\n`
    + 'Add one before releasing: the version number is the cheap half.',
  );
  process.exit(1);
}

const from = start.index + start[0].length;
// The next version heading, or the link definitions if this is the newest.
const next = changelog.slice(from).search(/^## \[|^\[Unreleased\]:/m);
const body = (next === -1 ? changelog.slice(from) : changelog.slice(from, from + next)).trim();

if (body.length === 0) {
  console.error(`CHANGELOG.md has a "## [${version}]" heading with nothing under it.`);
  process.exit(1);
}

process.stdout.write(`${body}\n\n---\n\n\`\`\`bash\nnpm install winding-engine\n\`\`\`\n`);

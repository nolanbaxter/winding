# Security

## Supported versions

The latest release of Winding is supported: fixes land in the next patch of the current minor
version. Older versions are not patched.

## Reporting a vulnerability

Please report it privately, not in a public issue: on the repository's **Security** tab, choose
**Report a vulnerability**
([direct link](https://github.com/nolanbaxter/winding/security/advisories/new)). Say what is
affected, how to reproduce it, and what an attacker could do with it.

You will hear back within a week. A confirmed issue is fixed in a patch release and credited to
you in its changelog, unless you would rather it were not.

## What counts

Winding runs in the page that loads it, so most of what it does is bounded by the browser. Reports
that matter most:

- a crafted file -- a glTF, an `.hdr`, a `.cube`, a splat capture -- that runs code, reads data it
  should not, or loads URLs past the `fetch` an app passed to the loader;
- anything in the release workflow or the published package that could let someone other than
  the maintainer publish a version.

A malformed file that only throws, or hangs the tab, is a bug: open an issue for it.

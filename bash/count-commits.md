```bash
#!/usr/bin/env bash
# Get time PACKAGE_A was first introduced, then count PACKAGE_B commits since that time
# Exclude commits that only change the generator tools under PACKAGE_B/Tool:
#   PACKAGE_B/Tool/Tool1*
#   PACKAGE_B/Tool/Tool2*
# Uses process substitution to avoid a subshell for while.

set -euo pipefail

# find first commit that touched PACKAGE_A
creation_commit=$(git rev-list --reverse HEAD -- PACKAGE_A | head -n1)
if [ -z "$creation_commit" ]; then
  echo "No commits touch PACKAGE_A" >&2
  exit 1
fi

creation_time=$(git show -s --format=%at "$creation_commit")
since_date=$(date -d "@$creation_time" --iso-8601=seconds)

count=0

# iterate commits that touch PACKAGE_B since creation time
while read -r commit; do
  # list files changed by this commit
  mapfile -t files < <(git diff-tree --no-commit-id --name-only -r "$commit")

  # collect files under PACKAGE_B
  changed_in_birch=()
  for f in "${files[@]}"; do
    case "$f" in
      PACKAGE_B/*) changed_in_birch+=("$f") ;;
    esac
  done

  # skip commits that don't touch PACKAGE_B
  [ ${#changed_in_birch[@]} -eq 0 ] && continue

  # check whether ALL PACKAGE_B changes are within the two excluded paths
  only_excluded=1
  for f in "${changed_in_birch[@]}"; do
    case "$f" in
      PACKAGE_B/Tool/Tool1*|PACKAGE_B/Tool/Tool2*) ;;
      *) only_excluded=0; break ;;
    esac
  done

  [ $only_excluded -eq 0 ] && count=$((count+1))
done < <(git rev-list --since="$since_date" -- PACKAGE_B)

echo "PACKAGE_A created at: $since_date (epoch $creation_time)"
echo "Number of PACKAGE_B commits since then (filtered): $count"
```

## Short examples showing safe checks
```bash
# Example: reading changed files into an array safely
mapfile -t files < <(git diff-tree --no-commit-id --name-only -r "$commit")

# correct: check array length (number of elements)
if [ ${#files[@]} -eq 0 ]; then
  echo "no files"
else
  echo "have files: ${#files[@]}"
fi

# Example: reading changed files into a string
files="$(git diff-tree --no-commit-id --name-only -r "$commit")"

# correct: check non-empty string
if [ -n "$files" ]; then
  echo "non-empty string"
else
  echo "empty"
fi
```

## Explanations of key bash constructs

### set -euo pipefail
Use this at the top of bash scripts to make them fail-fast and avoid silent errors.

- -e: exit immediately if any command returns a non-zero status.
- -u: treat unset variables as an error and exit.
- -o pipefail: in a pipeline, return the status of the first failing command (if any).

Note: `pipefail` is a bash (and some other shells) extension; it's not available in all POSIX shells (e.g., dash). If you need portability, run your script with bash explicitly.

Example:
```bash
# Without -o pipefail, the pipeline exit status is from the last command
false | true
echo "exit: $?"  # may be 0

# With 'set -o pipefail' the failing command value is preserved
set -o pipefail
false | true
echo "exit: $?"  # non-zero
```

### [ -z "VAR" ] — test for empty string
`-z` returns true when the string length is zero. Commonly used to check whether a command produced output.

Example:
```bash
creation_commit=""
if [ -z "$creation_commit" ]; then
  echo "empty"
fi
```

### [ -n "VAR" ] — test for non-empty string
`-n` returns true when the string length is non-zero. Use it for string checks.

Example:
```bash
files="$(git diff-tree --no-commit-id --name-only -r "$commit")"
if [ -n "$files" ]; then
  echo "non-empty string"
else
  echo "empty"
fi
```

Important: Do NOT use `if "$var"; then` — that attempts to execute the content of `$var` as a command.

### ${#array[@]} — number of elements in an array
When `files` is an array (for example populated with `mapfile -t`), `${#files[@]}` expands to the element count. Use it to test whether an array is empty.

Example:
```bash
mapfile -t files < <(printf "%s\n" "a b" "c")
if [ "${#files[@]}" -eq 0 ]; then
  echo "no files"
else
  echo "have files: ${#files[@]}"
fi
```

Note: if `files` is a plain string, `${#files[@]}` will not behave as expected; use `-n`/`-z` for strings instead.

### Practical tips
- Prefer arrays (e.g., `mapfile -t files < <(...)`) to preserve filenames with whitespace.
- Use process substitution with `while` (e.g., `while read -r line; do ...; done < <(command)`) to keep the loop running in the current shell rather than a subshell created by a pipe.
# Bash

## Command

###  Use following command to count changes in OldPackage from Package create

#### It is used when port changes from OldPackage to Package
```bash
PKG_TIME=$(git log --diff-filter=A --follow --format=%aI --reverse -- "*Package*" | head -1)
echo "Package created at: $PKG_TIME"

filtered_commits=()
while read commit; do
  files=$(git diff-tree --no-commit-id --name-only -r $commit -- "*OldPackage*")
  if echo "$files" | grep -qvE 'OldPackage/Dir/(SubDir1|SubDir2)/'; then
    filtered_commits+=("$commit")
  fi
done < <(git log --since="$PKG_TIME" --format="%H" -- "*OldPackage*")

echo "Commits in OldPackage (excluding Dir/SubDir1 and Dir/SubDir2 only changes): ${#filtered_commits[@]}"
```

Or

```bash
PKG_TIME=$(git log --diff-filter=A --follow --format=%aI --reverse -- "*Package*" | head -1)
echo "Package created at: $PKG_TIME"

count=0
while read commit; do
  files=$(git diff-tree --no-commit-id --name-only -r $commit -- "*OldPackage*")
  if echo "$files" | grep -qvE 'OldPackage/Dir/(SubDir1|SubDir2)/'; then
    ((count++))
  fi
done < <(git log --since="$PKG_TIME" --format="%H" -- "*OldPackage*")

echo "Commits in OldPackage (excluding Dir/SubDir1 and Dir/SubDir2 only changes): $count
```

**Note 1:** `${#filtered_commits[@]}` is bash array syntax:

- `filtered_commits` - the array variable name
- `[@]` - means "all elements in the array"
- `${#...}` - the # gets the count/length

So `${#filtered_commits[@]}` returns the number of elements in the array.

**Note 2:** `echo "$files" | grep -qvE 'OldPackage/Dir/(SubDir1|SubDir2)/'`

- `echo "$files"` - Output the list of changed files
- `|` - Pipe to grep
- `grep` - Search for pattern
- `-q` - Quiet mode (no output, just exit code)
- `-v` - Invert match (show lines that DON'T match)
- `-E` - Extended regex
- `'OldPackage/Dir/(SubDir1|SubDir2)/'` - Pattern matching those two Tool directories

**Note 3:** `< <(...)` feeds input to the while loop running in the **current shell**.

#### Below is **error** command and should avoid do like this
The issue is that the `while read` loop runs in a subshell, so the `count` variable doesn't persist.
```bash
PKG_TIME=$(git log --diff-filter=A --follow --format=%aI --reverse -- "*Package*" | head -1)
echo "Package created at: $PKG_TIME"

count=0
git log --since="$PKG_TIME" --format="%H" -- "*OldPackage*" | while read commit; do
  files=$(git diff-tree --no-commit-id --name-only -r $commit -- "*OldPackage*")
  if echo "$files" | grep -qvE 'OldPackage/Dir/(SubDir1|SubDir2)/'; then
    ((count++))
  fi
done

echo "Commits in OldPackage (excluding Dir/SubDir1 and Dir/SubDir2 only changes): $count
```

#### Key note
- `command | while` - The pipe `|` creates a subshell for the right side (the while loop)
- `while ... < <(command)` - Process substitution `<(...)` creates a temporary file descriptor, and `<` redirects it as input. The `while` loop runs in the **current shell**

#### Prompt to generate the sample
```
Give me bash to use git to get time the Package create, then get the numbers of commits in OldPackage since that time. If change only in Dir/SubDir1 or Dir/SubDir2, filter out it. Avoid use `command | while` because The pipe `|` creates a subshell and `while` run in subshell, should use `while ... < <(command)` to make `while` run in current shell
```
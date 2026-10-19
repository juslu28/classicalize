#!/usr/bin/env bash
# Put Classicalize on GitHub and turn on GitHub Pages.
#   bash publish.sh                 (run it again any time to update the site)
#   REPO_NAME=my-name bash publish.sh   to use a different repository name
set -e
cd "$(dirname "$0")"
source "$HOME/miniforge3/bin/activate" classicalize
export PYTORCH_ENABLE_MPS_FALLBACK=1
REPO="${REPO_NAME:-classicalize}"
COLLABORATORS="christopherthorpe zsardisantos"

say() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }

# 1. tools: the GitHub CLI (gh) and git, from conda-forge (no Homebrew needed)
if ! command -v gh >/dev/null 2>&1 || ! git --version >/dev/null 2>&1; then
  say "Installing the GitHub command-line tool (one time)…"
  conda install -y -q -n classicalize -c conda-forge gh git
fi

# 2. log in to GitHub (opens your browser the first time)
if ! gh auth status >/dev/null 2>&1; then
  say "Log in to GitHub: copy the one-time code shown below, press Enter, and paste it in the browser."
  gh auth login --hostname github.com --git-protocol https --web --scopes "repo"
fi
gh auth setup-git >/dev/null 2>&1 || true
ME=$(gh api user -q .login)
say "Logged in as $ME"

# 3. build the static site (docs/) from the songs you have made
say "Building the website from your songs in output/ (about 15 s per song)…"
python scripts/build_site.py --repo "https://github.com/$ME/$REPO"

# 4. commit
[ -d .git ] || git init -q -b main
git config user.name >/dev/null 2>&1 || git config user.name "$ME"
git config user.email >/dev/null 2>&1 || git config user.email "$(gh api user -q .id)+$ME@users.noreply.github.com"
git add -A
git commit -q -m "Classicalize: app + GitHub Pages site" || echo "(nothing new to commit)"
git branch -M main

# 5. create the public repository (first time) and push
if gh repo view "$ME/$REPO" >/dev/null 2>&1; then
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$ME/$REPO.git"
  say "Pushing to github.com/$ME/$REPO…"
  git push -u origin main
else
  say "Creating the public repository github.com/$ME/$REPO…"
  gh repo create "$REPO" --public --source . --remote origin --push \
    --description "Pop songs played with classical instruments"
fi

# 6. GitHub Pages from the docs/ folder
say "Turning on GitHub Pages…"
if ! gh api "repos/$ME/$REPO/pages" >/dev/null 2>&1; then
  for i in 1 2 3 4 5; do
    gh api -X POST "repos/$ME/$REPO/pages" -f "source[branch]=main" -f "source[path]=/docs" >/dev/null 2>&1 && break
    sleep 4
  done
else
  gh api -X PUT "repos/$ME/$REPO/pages" -f "source[branch]=main" -f "source[path]=/docs" >/dev/null || true
fi

# 7. share with the course staff (they get an invitation email)
for c in $COLLABORATORS; do
  if gh api -X PUT "repos/$ME/$REPO/collaborators/$c" -f permission=pull >/dev/null 2>&1; then
    echo "  invited $c"
  else
    echo "  could not invite $c (add them under Settings → Collaborators)"
  fi
done

URL="https://$ME.github.io/$REPO/"
say "Done!"
echo "  Repository:   https://github.com/$ME/$REPO"
echo "  Website:      $URL   (the first build takes 1–2 minutes)"
echo "  Submit the website URL for the assignment."
( sleep 2 && open "https://github.com/$ME/$REPO/actions" ) >/dev/null 2>&1 &

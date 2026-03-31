# Start fresh
rm -rf /tmp/pub_clean
mkdir -p /tmp/pub_clean && cd /tmp/pub_clean
git init

# Copy from private (normal + dotfiles)
cp -r /c/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/flowcytometertoolprivatedev/* ./
cp -r /c/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/flowcytometertoolprivatedev/.[!.]* ./

# Delete .git
rm -rf .git

# Re-initialise and commit as a brand-new repo
git init
git add -A
git commit -m "Public release (history squashed)"
git branch -M main
git remote add origin git@github.com:CefasRepRes/flowcytometertool.git
git push --force origin main